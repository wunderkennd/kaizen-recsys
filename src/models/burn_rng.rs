//! Reproducible training for the burn-backed models (issue #100).
//!
//! Two things stood between a seed and a bit-identical model on the
//! `burn-ndarray` 0.21 backend:
//!
//! 1. **One process-global RNG.** `NdArray::seed` stores a `seed_from_u64`
//!    RNG in a static `Mutex<Option<Rng>>`, and every random op
//!    (`Tensor::random`, which every `Initializer` and `Dropout` goes
//!    through) takes that RNG out, draws, and puts it back. Within one
//!    thread, seed → init → train replays the same stream. Across threads
//!    it does not: concurrent trainings (rayon-parallel tuning trials in
//!    `tune_with`, or `cargo test` running model tests in parallel)
//!    interleave their draws and each `B::seed` resets the stream under the
//!    others' feet. That was the source of the #90 "init lottery" flakes.
//!    [`run_training`] fixes this with a process-wide lock held for the
//!    whole run, taken before the backend is seeded, so a run owns the RNG
//!    stream. Concurrent burn trainings therefore execute one after
//!    another; EASE draws nothing from burn and keeps ADR-0002's
//!    parallelism.
//!
//!    The lock must never be waited on from a rayon worker thread. burn's
//!    element-wise parallelism (`multi-threads`) is dispatched to rayon's
//!    global pool, so a training on an ordinary thread, holding the lock,
//!    waits on that pool; a second training parked on the lock *inside* a
//!    worker takes that worker away, and once every worker is parked the
//!    holder can never finish — a deadlock. That is why the tuning runner
//!    evaluates burn trials sequentially on the calling thread
//!    (`FoldEvaluator::needs_exclusive_backend`). Running a training
//!    *inside* a pool (`ThreadPool::install`) is fine on its own: the
//!    holder then executes its own parallel work on that pool. What must
//!    not happen is a second training parking a worker of the holder's
//!    pool on the lock.
//! 2. **Alignment-dependent SIMD reductions.** burn-ndarray's `simd`
//!    feature chunks a buffer by its alignment, so the order in which
//!    partial sums meet changes with the allocation address, and a run
//!    differed from the next in the last ULP even single-threaded with the
//!    RNG pinned; one Adam step amplifies that into a different model. The
//!    feature is disabled in `Cargo.toml`. burn's rayon parallelism
//!    (`multi-threads`) is per element and deterministic, so training keeps
//!    using every core.
//!
//! With both in place a training run is a pure function of its seed and
//! configuration regardless of thread count or scheduling. Inference never
//! draws, so only the `train_*` entry points go through here.

use burn::tensor::backend::Backend;
use std::cell::Cell;
use std::sync::{Mutex, MutexGuard, PoisonError};

static BURN_RNG_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    /// Whether this thread currently holds [`BURN_RNG_LOCK`]. `std::sync::Mutex`
    /// is not reentrant, and a training legitimately nests a lock-taking
    /// step (Two-Tower's `finalize` reloads the trained record onto the
    /// inference backend through [`load_under_rng_lock`]), so the lock is
    /// taken once per thread and nested guards ride the outer one.
    static HELD_BY_THIS_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Take the lock unless this thread already holds it. `None` means the
/// caller is nested inside an outer guard on this thread, which keeps the
/// RNG exclusive to it either way.
fn lock_for_this_thread() -> Option<MutexGuard<'static, ()>> {
    if HELD_BY_THIS_THREAD.with(Cell::get) {
        return None;
    }
    let guard = BURN_RNG_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    HELD_BY_THIS_THREAD.with(|held| held.set(true));
    Some(guard)
}

/// Run `train` as a reproducible burn training: under the process-wide RNG
/// lock, with `B` seeded to `seed`. Every burn `train_*` entry point goes
/// through here.
pub fn run_training<B, R, F>(device: &B::Device, seed: u64, train: F) -> R
where
    B: Backend,
    F: FnOnce() -> R,
{
    let _guard = seeded_training_guard::<B>(device, seed);
    train()
}

/// Build a model from a saved record under the RNG lock, with every lazy
/// parameter forced to initialize before the lock is released.
///
/// `Module::load_record` swaps the saved value in without running a
/// parameter's lazy initializer, but a parameter the record does not
/// cover (an architecture mismatch) would draw at its first forward, in
/// whatever thread happens to run it. Forcing initialization here, under
/// the lock, keeps a concurrent training's stream intact in that case too;
/// the draws themselves are discarded. Used by every `load_from`.
pub fn load_under_rng_lock<B, M, F>(build: F) -> M
where
    B: Backend,
    M: burn::module::Module<B>,
    F: FnOnce() -> M,
{
    let _guard = BurnRngGuard {
        _guard: lock_for_this_thread(),
    };
    let model = build();
    let _ = model.num_params();
    model
}

/// Exclusive access to the backend's process-global RNG, seeded.
///
/// Hold it for the whole training run (bind it to a local that lives until
/// the function returns). Dropping it releases the lock for the next run.
/// A guard taken while this thread already holds one (`_guard == None`)
/// is a no-op wrapper: the outer guard keeps the RNG exclusive.
#[must_use = "the guard must stay alive for the whole training run"]
pub struct BurnRngGuard {
    _guard: Option<MutexGuard<'static, ()>>,
}

impl Drop for BurnRngGuard {
    fn drop(&mut self) {
        if self._guard.is_some() {
            HELD_BY_THIS_THREAD.with(|held| held.set(false));
        }
    }
}

/// Take the process-wide burn RNG lock, then seed `B` on `device`.
///
/// A poisoned lock (a training panicked while holding it) is recovered
/// rather than propagated: the RNG is re-seeded right after, so nothing the
/// panicking run did can leak into this one.
pub fn seeded_training_guard<B: Backend>(device: &B::Device, seed: u64) -> BurnRngGuard {
    let guard = lock_for_this_thread();
    B::seed(device, seed);
    BurnRngGuard { _guard: guard }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::{Distribution, Tensor};

    type B = NdArray<f32>;

    /// Draw a few random tensors the way init + dropout would, with a
    /// deliberate pause between draws so a concurrent, unguarded run would
    /// interleave with (and corrupt) the stream.
    fn draw_stream(seed: u64) -> Vec<f32> {
        let device = NdArrayDevice::default();
        let _guard = seeded_training_guard::<B>(&device, seed);
        let mut out = Vec::new();
        for _ in 0..4 {
            let t = Tensor::<B, 1>::random([32], Distribution::Default, &device);
            out.extend(t.into_data().convert::<f32>().into_vec::<f32>().unwrap());
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        out
    }

    #[test]
    fn guarded_streams_are_a_function_of_the_seed_even_when_concurrent() {
        // Sequential references.
        let ref_a = draw_stream(11);
        let ref_b = draw_stream(22);
        assert_ne!(ref_a, ref_b, "different seeds must give different streams");
        assert_eq!(ref_a, draw_stream(11), "same seed replays bit-for-bit");

        // Many threads hammering the same global RNG with two different
        // seeds: every one must reproduce its seed's reference stream.
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let seed = if i % 2 == 0 { 11 } else { 22 };
                std::thread::spawn(move || (seed, draw_stream(seed)))
            })
            .collect();
        for h in handles {
            let (seed, stream) = h.join().unwrap();
            let expected = if seed == 11 { &ref_a } else { &ref_b };
            assert_eq!(&stream, expected, "seed {seed} diverged under concurrency");
        }
    }

    /// Two-Tower's training calls `load_under_rng_lock` from inside
    /// `run_training` (its `finalize` step); a non-reentrant lock would
    /// deadlock there. Run it on a helper thread so a regression fails the
    /// test instead of hanging it.
    #[test]
    fn lock_is_reentrant_on_the_holding_thread() {
        use burn::module::Module;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let device = NdArrayDevice::default();
            let n = run_training::<NdArray, _, _>(&device, 5, || {
                let inner = seeded_training_guard::<NdArray>(&device, 6);
                assert!(inner._guard.is_none(), "nested guard must not re-lock");
                drop(inner);
                assert!(
                    HELD_BY_THIS_THREAD.with(Cell::get),
                    "dropping the nested guard must leave the outer one held"
                );
                let model = load_under_rng_lock::<NdArray, _, _>(|| {
                    burn::nn::LinearConfig::new(3, 2).init::<NdArray>(&device)
                });
                model.num_params()
            });
            assert!(!HELD_BY_THIS_THREAD.with(Cell::get));
            tx.send(n).unwrap();
        });
        let n = rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("nested lock use must not deadlock");
        assert_eq!(n, 3 * 2 + 2);
        // The lock is free again for another thread.
        let device = NdArrayDevice::default();
        let _g = seeded_training_guard::<NdArray>(&device, 1);
    }

    #[test]
    fn guard_recovers_from_a_poisoned_lock() {
        let device = NdArrayDevice::default();
        let _ = std::thread::spawn(move || {
            let _g = seeded_training_guard::<B>(&device, 1);
            panic!("poison the burn RNG lock on purpose");
        })
        .join();
        // Must not panic, and must still seed deterministically.
        assert_eq!(draw_stream(5), draw_stream(5));
    }
}

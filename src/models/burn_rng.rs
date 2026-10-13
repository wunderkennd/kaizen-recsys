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
//!    Holding a lock across work that runs on a shared rayon pool is a
//!    deadlock waiting to happen: burn's element-wise parallelism
//!    (`multi-threads`) dispatches to the *current* rayon pool, which for
//!    an ordinary thread is the global one, so a training that held the
//!    lock while waiting on the global pool could never finish once every
//!    global worker was itself parked on the lock (concurrent tuning
//!    trials, or any `train_*` call made from inside a rayon job). The
//!    training body therefore runs on a private pool owned by this module
//!    ([`burn_pool`]): the lock holder only ever waits on workers that no
//!    other training can park, because only the lock holder's body is in
//!    that pool. A `train_*` call made from a rayon worker takes the lock
//!    on a plain helper thread instead (`install` from a worker would keep
//!    it busy with its own pool's jobs while waiting, which could start a
//!    second training inside the first), and the worker simply blocks on
//!    the result, which no lock holder depends on. The tuning runner still
//!    evaluates burn trials one at a time on the calling thread
//!    (`FoldEvaluator::needs_exclusive_backend`): they serialize on the
//!    lock anyway, so fanning them out would only park workers.
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
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

static BURN_RNG_LOCK: Mutex<()> = Mutex::new(());

/// The pool every training body runs on (see the module docs). Sized like
/// rayon's global pool (`RAYON_NUM_THREADS`, else the available
/// parallelism), so the usual knob still caps burn's parallelism.
static BURN_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

thread_local! {
    /// Whether this thread currently holds [`BURN_RNG_LOCK`]. `std::sync::Mutex`
    /// is not reentrant, and a training legitimately nests a lock-taking
    /// step (Two-Tower's `finalize` reloads the trained record onto the
    /// inference backend through [`load_under_rng_lock`]), so the lock is
    /// taken once per thread and nested guards ride the outer one.
    static HELD_BY_THIS_THREAD: Cell<bool> = const { Cell::new(false) };

    /// Set on every [`BURN_POOL`] worker. Only the lock holder's body runs
    /// there, so a guard requested from one of its workers is nested in
    /// that body and must not lock again.
    static IN_BURN_POOL: Cell<bool> = const { Cell::new(false) };
}

fn burn_pool() -> &'static rayon::ThreadPool {
    BURN_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("burn-train-{i}"))
            .start_handler(|_| IN_BURN_POOL.with(|flag| flag.set(true)))
            .build()
            .expect("failed to build the burn training thread pool")
    })
}

/// Take the lock unless this thread already holds it, or is a worker of
/// the pool the lock holder's body is running on. `None` means the caller
/// is nested inside an outer guard, which keeps the RNG exclusive to it
/// either way.
fn lock_for_this_thread() -> Option<MutexGuard<'static, ()>> {
    if HELD_BY_THIS_THREAD.with(Cell::get) || IN_BURN_POOL.with(Cell::get) {
        return None;
    }
    let guard = BURN_RNG_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    HELD_BY_THIS_THREAD.with(|held| held.set(true));
    Some(guard)
}

/// Run `train` as a reproducible burn training: under the process-wide RNG
/// lock, with `B` seeded to `seed`, on the private burn pool. Every burn
/// `train_*` entry point goes through here. Safe to call from any thread,
/// a rayon worker included.
pub fn run_training<B, R, F>(device: &B::Device, seed: u64, train: F) -> R
where
    B: Backend,
    F: FnOnce() -> R + Send,
    R: Send,
{
    if rayon::current_thread_index().is_some() {
        // Called from inside a rayon job: take the lock on a plain thread
        // and let the worker block on the result (see the module docs).
        std::thread::scope(|scope| {
            scope
                .spawn(|| run_training_locked::<B, R, F>(device, seed, train))
                .join()
                .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
        })
    } else {
        run_training_locked::<B, R, F>(device, seed, train)
    }
}

fn run_training_locked<B, R, F>(device: &B::Device, seed: u64, train: F) -> R
where
    B: Backend,
    F: FnOnce() -> R + Send,
    R: Send,
{
    let _guard = seeded_training_guard::<B>(device, seed);
    burn_pool().install(train)
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
/// A guard taken while this thread already holds one, or on a burn-pool
/// worker (`_guard == None`), is a no-op wrapper: the outer guard keeps
/// the RNG exclusive.
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
    use rayon::prelude::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

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
            std::thread::sleep(Duration::from_millis(5));
        }
        out
    }

    /// A random draw followed by a matmul: burn-ndarray runs matmul (like
    /// conv and pooling) inside `rayon::scope`, which from an ordinary
    /// thread injects into the global pool and waits. That is the wait a
    /// training does many times per step while the lock is held.
    fn pool_backed_op(device: &NdArrayDevice) -> f32 {
        let a = Tensor::<B, 2>::random([64, 64], Distribution::Default, device);
        let b = Tensor::<B, 2>::random([64, 64], Distribution::Default, device);
        a.matmul(b).sum().into_scalar()
    }

    /// Run `f` on a helper thread and fail, instead of hanging, if it has
    /// not finished within `secs`.
    fn finishes_within<T: Send + 'static>(
        secs: u64,
        what: &str,
        f: impl FnOnce() -> T + Send + 'static,
    ) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(Duration::from_secs(secs))
            .unwrap_or_else(|_| panic!("{what} did not finish within {secs}s (deadlock?)"))
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

    /// The training body runs on the burn pool while the caller holds the
    /// lock; a guard or `load_under_rng_lock` requested from inside the
    /// body (Two-Tower's `finalize` does the latter) must ride the outer
    /// guard instead of locking again.
    #[test]
    fn lock_is_reentrant_inside_a_training_body() {
        use burn::module::Module;
        let n = finishes_within(30, "nested lock use", || {
            let device = NdArrayDevice::default();
            let n = run_training::<NdArray, _, _>(&device, 5, || {
                assert!(IN_BURN_POOL.with(Cell::get), "body runs on the burn pool");
                let inner = seeded_training_guard::<NdArray>(&device, 6);
                assert!(inner._guard.is_none(), "nested guard must not re-lock");
                drop(inner);
                let model = load_under_rng_lock::<NdArray, _, _>(|| {
                    burn::nn::LinearConfig::new(3, 2).init::<NdArray>(&device)
                });
                model.num_params()
            });
            assert!(!HELD_BY_THIS_THREAD.with(Cell::get));
            n
        });
        assert_eq!(n, 3 * 2 + 2);
        // The lock is free again for another thread.
        let device = NdArrayDevice::default();
        let _g = seeded_training_guard::<NdArray>(&device, 1);
    }

    /// Contention test for the deadlock described in the module docs: an
    /// ordinary thread holds the lock and needs pool-backed burn ops while
    /// every worker of the global rayon pool is parked on the lock by a
    /// training call of its own. The holder's body runs on the private
    /// burn pool, so it finishes; the parked trainings then run in turn.
    #[test]
    fn trainings_from_rayon_workers_cannot_starve_the_lock_holder() {
        let n_workers = rayon::current_num_threads();
        let results = finishes_within(120, "trainings contending for the lock", move || {
            let device = NdArrayDevice::default();
            let holder_locked = Arc::new(AtomicBool::new(false));
            let workers_at_lock = Arc::new(AtomicUsize::new(0));

            let holder = {
                let holder_locked = Arc::clone(&holder_locked);
                let workers_at_lock = Arc::clone(&workers_at_lock);
                std::thread::spawn(move || {
                    run_training::<NdArray, _, _>(&device, 1, || {
                        holder_locked.store(true, Ordering::SeqCst);
                        // Wait until every global worker is about to block
                        // on the lock, then give them a moment to do so.
                        while workers_at_lock.load(Ordering::SeqCst) < n_workers {
                            std::thread::yield_now();
                        }
                        std::thread::sleep(Duration::from_millis(100));
                        pool_backed_op(&device)
                    })
                })
            };
            while !holder_locked.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }

            // One training call per global worker. Each item spins until all
            // of them have started, so they occupy distinct workers, then
            // calls `run_training` and blocks on the lock the holder has.
            let started = Arc::new(AtomicUsize::new(0));
            let from_workers: Vec<f32> = (0..n_workers)
                .into_par_iter()
                .with_min_len(1)
                .map(|i| {
                    started.fetch_add(1, Ordering::SeqCst);
                    while started.load(Ordering::SeqCst) < n_workers {
                        std::thread::yield_now();
                    }
                    assert!(rayon::current_thread_index().is_some());
                    workers_at_lock.fetch_add(1, Ordering::SeqCst);
                    run_training::<NdArray, _, _>(&device, 10 + i as u64, || {
                        pool_backed_op(&device)
                    })
                })
                .collect();

            let from_holder = holder.join().unwrap();
            (from_holder, from_workers)
        });

        let (from_holder, from_workers) = results;
        assert_eq!(from_workers.len(), n_workers);
        // Every run owned the RNG: same seed later gives the same value.
        let device = NdArrayDevice::default();
        assert_eq!(
            from_holder,
            run_training::<NdArray, _, _>(&device, 1, || pool_backed_op(&device))
        );
        for (i, v) in from_workers.iter().enumerate() {
            assert_eq!(
                *v,
                run_training::<NdArray, _, _>(&device, 10 + i as u64, || pool_backed_op(&device))
            );
        }
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

# kzn-recsys (FEASE) — developer commands.
# Run `just` to list recipes.

# Override per-machine: `JUST_PYTHON=python3.12 just test`
# pytest runs as `python -m pytest` (not the .venv/bin/pytest console script):
# the script bakes an absolute-path shebang at install time, so it breaks if the
# repo dir is renamed/moved. Invoking via the python binary is rename-immune.
python  := env_var_or_default("JUST_PYTHON",  ".venv/bin/python")
maturin := env_var_or_default("JUST_MATURIN", ".venv/bin/maturin")
pytest  := env_var_or_default("JUST_PYTEST",  python + " -m pytest")

_default:
    @just --list

# --- Build -----------------------------------------------------------------

# Compile Rust + install package into the active venv (dev mode).
develop:
    {{maturin}} develop

# Build release wheel into dist/ (pinned to {{python}} for tag consistency).
build:
    rm -rf dist
    {{maturin}} build --release --out dist --interpreter {{python}}

# --- Test layers -----------------------------------------------------------

# Layer 1: Rust unit tests (algorithmic correctness, fastest signal).
test-rust:
    cargo test --release

# Layer 2: Python unit tests. Rebuilds the .so first so you never test stale code.
test-python: develop
    {{pytest}} tests/ -v

# Run a single Python test, e.g. `just test-one tests/test_model.py::test_warm_user_prediction`.
test-one TEST: develop
    {{pytest}} {{TEST}} -v

# Layer 1 + 2 — pre-commit dev loop.
test: test-rust test-python

# Layer 3: Verify the built wheel ships kzn_recsys/ helpers.
test-wheel: build
    @echo "Inspecting dist/*.whl ..."
    @unzip -l dist/*.whl | grep -E "kzn_recsys/(_native|__init__|schemas|fease_wrapper)" \
        || (echo "FAIL: wheel missing kzn_recsys/ helpers" && exit 1)
    @echo "OK: wheel ships kzn_recsys/ helpers"

# Layer 4: Fresh-venv install + import smoke test (run from /tmp to avoid sys.path shadowing).
test-fresh: build
    #!/usr/bin/env bash
    set -euo pipefail
    TMPVENV=$(mktemp -d -t fease-fresh-XXXXXX)
    trap "rm -rf '$TMPVENV'" EXIT
    {{python}} -m venv "$TMPVENV"
    "$TMPVENV/bin/pip" install --quiet --upgrade pip
    "$TMPVENV/bin/pip" install --quiet dist/kzn_recsys-*.whl
    cd /tmp && "$TMPVENV/bin/python" -c "
    from kzn_recsys import (
        build_and_train, FeaseModel, ModelRegistry, SplitResult,
        EngagementSchema, random_split_safe,
    )
    print('OK: fresh-venv import works ->', FeaseModel)
    "

# All four install/build layers — full pre-release sanity check.
test-all: test-rust test-python test-wheel test-fresh

# --- Lint / format --------------------------------------------------------

fmt:
    cargo fmt --all

lint:
    cargo clippy --all-targets -- -D warnings

check: fmt lint test

# --- Release --------------------------------------------------------------

# Why tag origin/main instead of local HEAD: a release should always pin to
# what's on the remote — local main may be stale or have uncommitted edits
# (e.g. unrelated WIP). This recipe always fetches first, then tags the
# fetched origin/main SHA, so the release wheel content matches what merged.

# Tag origin/main as VERSION and push (e.g. `just release v0.1.0` or `just release v0.2.0-rc1`).
release VERSION:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! [[ "{{VERSION}}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-rc[0-9]+)?$ ]]; then
        echo "FAIL: VERSION must match vX.Y.Z or vX.Y.Z-rcN (got: {{VERSION}})" >&2
        exit 1
    fi
    git fetch origin --tags --prune
    if git rev-parse "{{VERSION}}" >/dev/null 2>&1; then
        echo "FAIL: tag {{VERSION}} already exists locally" >&2
        exit 1
    fi
    if git ls-remote --tags --exit-code origin "refs/tags/{{VERSION}}" >/dev/null 2>&1; then
        echo "FAIL: tag {{VERSION}} already exists on origin" >&2
        exit 1
    fi
    SHA=$(git rev-parse origin/main)
    echo "Tagging origin/main ($SHA) as {{VERSION}} ..."
    git tag -a "{{VERSION}}" "$SHA" -m "{{VERSION}}

    Release built from origin/main @ $SHA.
    Wheels published via .github/workflows/release.yml."
    git push origin "{{VERSION}}"
    echo
    echo "Tag pushed. Watch the build with: just release-watch"
    echo "Release page (populated when build succeeds):"
    echo "  https://github.com/wunderkennd/fease/releases/tag/{{VERSION}}"

# Watch the most recent release.yml run (useful right after `just release`).
release-watch:
    gh run watch $(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId') --interval 30

# --- Maintenance ----------------------------------------------------------

# Remove Rust + Python build artifacts. Does NOT touch .venv.
clean:
    cargo clean
    rm -rf dist/ build/ *.egg-info
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
    find kzn_recsys -maxdepth 2 -name "*.so" -delete

# Show what `just test-wheel` would inspect, without grepping.
wheel-contents: build
    unzip -l dist/*.whl

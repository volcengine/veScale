# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository scope

This repo open-sources one piece of ByteDance's internal veScale: a DTensor library with a new **`RaggedShard`** placement (asymmetric sharding over flattened tensor storage). The active code is `vescale/` (~5.7k lines of pure Python). `legacy/` is the old, much larger veScale library (FSDP, optimizers, examples) moved here for reference only — do not develop against or reformat it.

## Commands

Install (Python >= 3.11; pins `torch==2.7.1` cu121 and `numpy<2.0.0`):

```bash
pip3 install -r requirements.txt && pip3 install -e .
```

Run the full test suite (installs, then runs every `test_*.py` under `test/`):

```bash
./scripts/run_test.sh
```

Run a single test file / test (tests import `common_dtensor` as a top-level module, so you must run from `test/` with `PYTHONPATH` including it):

```bash
cd test
export PYTHONPATH=$(pwd):$PYTHONPATH
pytest dtensor/ragged_shard/test_norm.py -k test_ragged_shard_norm_dim0 -s
```

Lint/format (configured in `pyproject.toml`, line length 120; there is no pre-commit config or CI):

```bash
ruff check .
black .
```

## Testing conventions

- Tests subclass `DTensorTestBase` in `test/common_dtensor.py`, which extends torch's `MultiProcessTestCase`: each test **spawns `world_size` (4) processes** and requires the `@with_comms` decorator. Backend is NCCL if ≥2 CUDA GPUs are visible, else gloo on CPU (`test/dtensor/cpu_only/` tests are plain CPU-only `unittest` with a `ProcessPoolExecutor`).
- `scripts/run_test.sh` `pkill`s stray pytest processes between files because spawned distributed processes can hang the suite — if a local pytest run appears stuck, leftover workers are the likely cause.
- Multi-GPU tests are skipped (not failed) when GPUs are unavailable; a green CPU run does not exercise NCCL paths.

## Architecture

`vescale.dtensor` is a **minimally-patched fork of PyTorch 2.7's `torch.distributed.tensor`**, not a reimplementation. The guiding pattern throughout: subclass torch's class, override only what `RaggedShard` breaks, and delegate everything else back to torch. Each modified file documents its changes vs. torch in a module-level docstring comment — read those before editing.

The extension points:

- **`_api.py` — `DTensor(TorchDTensor)`**: carries a vescale `OpDispatcher` as the class attribute `_op_dispatcher`; `__torch_dispatch__` routes through it so all aten ops land in vescale code. Factory funcs (`distribute_tensor`, `from_local`, `redistribute`, `to_local`) are re-pointed to vescale's versions so they return vescale DTensors.
- **`_dispatch.py` — `OpDispatcher(TorchOpDispatcher)`**: merges torch's custom op handlers with vescale ones (fused AdamW/SGD handler, ragged norm handler, `found_inf` reduction for grad clipping). Adds `_cvt_dtensor` which runs before dispatch.
- **`_sharding_prop.py` — `ShardingPropagator(TorchShardingPropagator)`**: on init, merges torch propagator's `op_to_schema_info` / `op_to_rules` / `op_strategy_funcs` dicts with vescale's (vescale wins on conflict). Overrides `propagate_op_sharding_non_cached` to consult `is_ragged_shard` when adjusting shape/stride args.
- **`_redistribute.py` — `redistribute_local_tensor`**: fast path — if neither spec has a `RaggedShard`, calls torch's implementation directly. Otherwise handles three cases: ragged→ragged (single all-to-all via `RaggedShard._to_new_ragged_shard`), ragged→standard (all-gather via `_to_replicate_tensor`, then torch redistribute), standard→ragged (torch redistribute, then local `_split_tensor`). Ragged↔ragged is only supported within the same mesh dim.
- **`placement_types.py` — `RaggedShard(Placement)`**: shards a *contiguous flattened* tensor; `dims` = prefix dims collapsed together (must be `(0..k-1)`), `local_units` = relative element weights per rank (length == mesh dim size, sum must divide total numel). `_StridedRaggedShard` adds `split_factor` to compose with `Shard(0)` by recording a permutation (its `_to_replicate_tensor` is known-buggy per in-code TODO — avoid it in redistribute).
- **`vescale/utils/monkey_patch.py` — `patch_method`**: used to bolt `is_ragged_shard()` onto torch's own `Placement` base class so torch-side code can detect ragged placements without subclassing.

Supporting machinery:

- **`_ops/`** (pointwise/math/matrix/tensor): sharding rules registered via torch's `register_prop_rule` / `register_op_strategy` globals (re-exported in `_ops/utils.py`); vescale's `ShardingPropagator.__init__` picks them up, and `vescale/dtensor/__init__.py` force-imports `_ops` so registration happens on package import. Many files are near-copies of torch's, kept because the rules must output vescale `DTensorSpec`s and handle `RaggedShard`.
- **`vescale_utils/ragged_shard_utils.py`**: index arithmetic and spec helpers. Two idioms recur: `substitute_ragged_with_replicate` (swap the RaggedShard for `Replicate()` so torch's machinery handles the other mesh dims, then handle the ragged dim separately) and heavy `functools.lru_cache` on spec-derived computations (specs are hashable).
- **`vescale_utils/checkpoint.py`**: PyTorch DCP integration (`_break_ragged_box`, chunk/write-item builders) enabling communication-free checkpointing of ragged shards.

Constraints enforced across the codebase: exactly one `RaggedShard` per placements tuple; RaggedShard does not support SymInts; cross-mesh redistribute is unimplemented.

## Notes

- `setup.py` reads the version from `vescale/__init__.py` (`__version__`); importing `vescale` prints the version.
- The root `Dockerfile` is a placeholder; `.clang-format` is a leftover for the legacy tree. The new `vescale/` code is pure Python.

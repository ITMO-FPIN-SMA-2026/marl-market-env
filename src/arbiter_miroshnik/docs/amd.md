# PPO + GRU on the AMD RX 6600 XT

The original actor, critic, and PPO implementation in `arbiter_miroshnik` run on
the AMD RX 6600 XT through ROCm. The separate virtual environment is
`.deps/rocm-venv`; the CPU environment remains `.venv`.

## Tested installation

```bash
python3 -m venv .deps/rocm-venv
.deps/rocm-venv/bin/python -m pip install 'torch[device-gfx1032]==2.14.0+rocm10.1.0' \
  --index-url https://stable.repo.amd.com/rocm/whl-next/
.deps/rocm-venv/bin/python -m pip install -e '.[dev]'
.deps/rocm-venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The tested runtime reports PyTorch `2.14.0+rocm10.1.0`, HIP `7.16.26385`, and
`AMD Radeon RX 6600 XT`. PyTorch uses the device name `cuda` for ROCm too.
Installation references: [AMD TheRock](https://github.com/ROCm/TheRock/blob/main/RELEASES.md),
[GPU targets](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md), and
[PyTorch HIP semantics](https://docs.pytorch.org/docs/2.15/notes/hip.html).

GPU device files `/dev/kfd` and `/dev/dri/renderD128` are accessible on the host
but hidden inside the Codex sandbox. GPU commands require host execution there.

## Continue the saved PPO + GRU

The local GPU series is `runs/transferred_ppo_amd`, copied from the latest-team
series at 64 games per model. All three models completed eight additional games
on the GPU: **72 games and nine PPO updates per model**. Actor, critic, Adam
states, trajectories, and both seller populations are saved by the original
launcher. Model architecture, native reward, agents, and scenarios are unchanged.

```bash
.deps/rocm-venv/bin/python -m arbiter_miroshnik evaluate \
  --out runs/transferred_ppo_amd --episodes 20 --device cuda
.deps/rocm-venv/bin/python -m arbiter_miroshnik train \
  --out runs/transferred_ppo_amd --games 128 --device cuda
```

`--games` is the target total per model. Only one training process may write a
given output directory at a time. New checkpoints preserve earlier weights.

Other local series remain: `runs/transferred_ppo` at 128 games per model and
`runs/transferred_ppo_latest` at 64 games per model. The separately transferred
archives still contain the original 64-game and latest-team eight-game snapshots;
the new local checkpoints are excluded from Git.

Full qualification of **PPO + GRU** remains unproven. Its objective is a positive
welfare effect with at least 90% sanction precision, checked on independent
evaluation worlds and training replicas. Results from unrelated controllers
are excluded from this claim.

# CoDreamer

**Communication-Based Decentralised World Models**
CoCoMARL Workshop @ RLC 2024 · [arXiv:2406.13600](https://arxiv.org/abs/2406.13600)

An independent, from-scratch implementation of CoDreamer — a multi-agent extension
of DreamerV3 in which agents communicate **inside the world model**, not only in
their policies. The paper ships no official code; this repository is built from
the paper and its appendix.

Built on [`NM512/dreamerv3-torch`](https://github.com/NM512/dreamerv3-torch) (MIT).

---

## Overview

A world-model RL agent learns a latent model of its environment and trains its
policy inside that model. In the multi-agent setting, the straightforward
approach — **IDreamer** in the paper's terminology — gives every agent its own
copy of the model and treats the other agents as part of the environment. Each
agent is therefore blind to what its neighbours are doing until the consequences
show up in its own observations.

CoDreamer changes one thing: agents are nodes in a graph, and the world model
performs neighbourhood aggregation **while predicting**. Every agent's latent
state is built from its own history plus its neighbours' hidden states at the
same timestep, so communication shapes the model's predictions rather than
sitting on top of them.

Both variants share a single set of weights across agents (parameter sharing),
as specified by the paper. The only difference between them is whether the
graph message-passing layers are active.

---

## How communication is wired in

```
   t-1              t                t+1
  ┌────┬────┐    ┌────┬────┐     ┌────┬────┐
  │ h1 │ h2 │    │ h1 │ h2 │     │ h1 │ h2 │   latent states
  └────┴────┘    └────┴────┘     └────┴────┘
       ╲  ╱          ╲  ╱            ╲  ╱
        ╳  ◄── GNN exchanges here ──► ╳
       ╱  ╲          ╱  ╲            ╱  ╲
  ┌────┴────┐    ┌────┴────┐     ┌────┴────┐
  │ h1'│ h2'│ ──►│ h1'│ h2'│ ──► │ h1'│ h2'│
  └────┬────┘    └────┬────┘     └────┬────┘
       │              │               │
   RNN advances in time (directed)
   GNN exchanges within a timestep (undirected)
```

The two axes are orthogonal. Time is recurrent and directed; communication is
simultaneous and symmetric — every node reads its neighbours' **pre-update**
values, so a single layer has no ordering. Stacking *k* layers propagates
information *k* hops within one timestep.

**Three insertion points**, following the paper's tables:

| Location | What it changes |
|---|---|
| RSSM (`img_step`, after the GRU) | The prior, the posterior, and everything downstream of `get_feat` |
| Reward and continuation heads | Predicted reward and episode termination |
| Actor and critic | Behaviour learning and value estimation |

The decoder is deliberately left untouched, as the paper requires. The
insertion point in the RSSM is the one place that feeds both the prior and the
posterior, so a single layer there reaches every downstream head without extra
plumbing.

**The graph comes from the environment** (an edge exists when two agents are
within a communication radius), never from a learned predictor. During
imagination the graph from the trajectory's starting point is frozen and reused
for the whole rollout.

---

## Features

- **Two environments**
  - *Sequential Estimate Game* — the paper's appendix-B cooperative game, in
    pure NumPy. A deliberate stress test: an agent's target depends mostly on
    its neighbours' hidden state, so local information is provably insufficient.
  - *VMAS* — flocking, discovery and buzz-wire, wrapped to the same
    team-level interface.
- **Agent axis kept inside the sample** — data is shaped `(B, T, N, ·)` rather
  than flattened into the batch, because the GNN must aggregate over all *N*
  agents at the same timestep.
- **GATv2 message passing** with edge features, a skip connection against
  over-smoothing, and numerically safe handling of isolated agents.
- **One switch between variants** — `--comm_variant idreamer` / `codreamer`.
  At zero layers the aggregation module is a strict identity (it returns the
  input object itself), so IDreamer is exactly the same code path with
  communication off.
- **Configuration-driven** — environment spec, layer counts, observation mode
  and communication topology are all overridable from the command line.
- **Numerical tests instead of curve-watching** — structural bugs tend to train
  quietly rather than badly, so the critical paths are covered by assertions.

---

## Installation

```bash
conda create -n codreamer python=3.10
conda activate codreamer

# On some platforms the default PyPI wheel is CPU-only; pin the CUDA build.
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r third_party/dreamerv3-torch/requirements.txt
```

Verify the toolchain (Python, torch, CUDA, free memory, dependencies, and the
integrity of the vendored upstream files):

```bash
python scripts/check_env.py
```

Developed and tested on Ubuntu 22.04 with a single 12 GB GPU. The model and the
data pipeline are device-agnostic; VMAS runs its physics on CPU.

---

## Quick start

```bash
# Fast end-to-end sanity check (a few minutes)
bash scripts/run_smoke.sh

# Windows
powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1
```

---

## Usage

### Training

```bash
cd third_party/dreamerv3-torch

# IDreamer — no communication between agents
python dreamer.py --configs codreamer --comm_variant idreamer \
    --seg_graph_obs True --logdir ~/runs/idreamer

# CoDreamer — one GNN layer in the RSSM, the prediction heads, and actor-critic
python dreamer.py --configs codreamer --comm_variant codreamer \
    --seg_graph_obs True --logdir ~/runs/codreamer
```

`--comm_variant` is the only switch you normally need. It drives all three
insertion points at once, which prevents the most common mistake: enabling the
RSSM layer while leaving the heads and actor-critic off, which yields neither
the paper's CoDreamer nor IDreamer.

For single-point ablations, `--gnn_layers`, `--gnn_head_layers` and
`--gnn_behavior_layers` override the variant individually.

### VMAS

```bash
python dreamer.py --configs codreamer --comm_variant codreamer \
    --task vmas_flocking --vmas_graph_obs True --logdir ~/runs/flocking
```

The task name selects the scenario (`vmas_flocking`, `vmas_discovery`,
`vmas_buzz_wire`); per-scenario agent counts and parameters live in
`envs/vmas_env.py`.

### Monitoring and plots

```bash
python scripts/monitor.py --log ~/runs/codreamer/main.log
python scripts/plot_metrics.py --logdir ~/runs/codreamer --out figures/curve.png
python scripts/compare_runs.py --run "IDreamer=~/runs/idreamer" \
                               --run "CoDreamer=~/runs/codreamer" --out figures/ab.png
```

### Evaluation

```bash
python scripts/eval_policy.py --checkpoint ~/runs/codreamer/latest.pt
```

---

## Testing

Learning curves are a poor correctness signal: they are noisy, and a structural
bug often trains *quietly* rather than visibly badly. The critical paths are
therefore covered by numerical assertions.

```bash
cd third_party/dreamerv3-torch
python -m envs.seg            # reward semantics, adjacency, reproducibility
python -m envs.seg_gym        # gym spaces, agent axis, graph structure
python -m envs.vmas_env       # VMAS wrapper: shapes, seeds, graph validity
python -m tools_marl          # multi-agent collection, team grouping, sampling
python -m networks_gnn        # GATv2: isolated nodes, smoothing, gradients

cd ../.. && python scripts/test_graph_pipeline.py
```

`test_graph_pipeline.py` asserts a **numerical identity**: with the same data,
weights and optimizer configuration, the graph-batch path and the flattened path
must produce identical losses. With communication disabled the aggregation is an
exact identity, so the relative error must sit at float32 machine precision
(~1e-7). Any axis mix-up or off-by-one in the agent dimension shows up orders of
magnitude larger than that.

---

## Project structure

```
CoDreamer/
├── scripts/
│   ├── check_env.py                # toolchain and upstream-integrity check
│   ├── run_smoke.{sh,ps1}          # end-to-end sanity run
│   ├── test_graph_pipeline.py      # numerical equivalence of the data path
│   ├── eval_policy.py              # checkpoint evaluation
│   ├── eval_world_model.py         # open-loop world-model error
│   ├── plot_metrics.py, compare_runs.py, monitor.py
├── third_party/
│   ├── THIRD_PARTY.md              # provenance and modification policy
│   └── dreamerv3-torch/            # vendored base (MIT)
│       ├── networks_gnn.py         # GATv2 message passing
│       ├── envs/seg.py             # Sequential Estimate Game
│       ├── envs/seg_gym.py         # team-level gym wrapper
│       ├── envs/vmas_env.py        # VMAS adapter
│       ├── tools_marl.py           # multi-agent collection and graph sampling
│       └── UPSTREAM_MANIFEST.sha256
├── docs/                           # developer notes
└── README.md
```

Files added on top of the vendored base are not listed in
`UPSTREAM_MANIFEST.sha256`, so **"manifest mismatch ⟺ upstream was modified"**
holds as an invariant: any change to the base can be detected mechanically.

---

## Status

Implemented and working:

- End-to-end multi-agent world-model training (world model, actor, critic) with
  parameter sharing.
- Graph message passing in all three locations described by the paper, with a
  single switch between IDreamer and CoDreamer.
- The paper's Sequential Estimate Game, and a VMAS adapter covering flocking,
  discovery and buzz-wire.
- Self-checks for the environment, the data pipeline and the GNN module, plus a
  numerical-equivalence test for the graph batching path.
- Checkpointing, resumable evaluation, learning-curve tooling and open-loop
  world-model diagnostics.

Known limitations:

- **This is not a verified bit-exact reproduction of the paper's reported
  numbers.** The paper provides no reference implementation, and several design
  choices it does not pin down — most notably the message-aggregation operator —
  are implementation decisions made here.
- VMAS scenario details follow the installed VMAS version rather than the
  version the paper used, so observation and action dimensions can differ.
- Tuning targets a single 12 GB GPU; the paper's full batch configuration does
  not fit in that budget and is scaled down accordingly.

---

## Acknowledgements

The base codebase is [`NM512/dreamerv3-torch`](https://github.com/NM512/dreamerv3-torch),
distributed under the MIT licence, copyright NM512 (2023). The original licence
is retained inside `third_party/dreamerv3-torch/`; provenance is documented in
[`third_party/THIRD_PARTY.md`](third_party/THIRD_PARTY.md). Everything else in
this repository is the reproduction work.

## License

The vendored base in `third_party/dreamerv3-torch/` is MIT-licensed, copyright
NM512 (2023); its original licence file is kept in that directory. Everything
else in this repository is the reproduction work and carries no separate licence
file.

## Citation

```bibtex
@inproceedings{codreamer2024,
  title         = {CoDreamer: Communication-Based Decentralised World Models},
  booktitle     = {CoCoMARL Workshop at RLC},
  year          = {2024},
  eprint        = {2406.13600},
  archivePrefix = {arXiv}
}
```

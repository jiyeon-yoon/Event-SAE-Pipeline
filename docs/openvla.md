# openVLA pipeline

Closed-loop interpretability pipeline for the openVLA backbone on the
LIBERO simulation suites.

## Pipeline sanity check (5-trial scale)

This is the public minimal sweep (LIBERO-Spatial, top-5 features
per ranking, α = 0, 5 trials per task, seed 0), not the paper's
full-scale experiment. The goal is to verify this codebase
reproduces the qualitative ranking order observed in the reference
research code at this sample size; small numerical gaps are
expected.

| Configuration                  | SAE       | Feature lists  | Baseline SR | Δ event-aligned | Δ window-mean | Δ task-mean | Δ random-alive |
|---|---|---|---:|---:|---:|---:|---:|
| Reference                      | original  | original       | 80.0%       | −28.4           | −7.2          | −8.0        | −6.8           |
| This codebase, both reused     | original  | original       | 82.0%       | −26.8           | −9.2          | −9.6        | −9.2           |
| This codebase, SAE only reused | original  | this codebase  | 80.0%       | −25.2           | −6.4          | −8.0        | −8.8           |

ΔSR is in percentage points relative to each row's own baseline run.

In the third row, only the SAE is held fixed, and every other step
runs through this repository. Small gaps from the reference are
expected: 50 rollouts per condition leave a few pp of sampling
noise, and float32 / CUDA versions can also shift SR run-to-run.

## Installation

### Step 1: Conda environment

```bash
conda env create -f environment-openvla.yml
conda activate event-sae-openvla
```

Optional flash-attn for faster inference (openVLA falls back to `sdpa`
if skipped):

```bash
pip install flash-attn==2.5.5 --no-build-isolation
```

### Step 2: External libraries

Three libraries installed editable into the conda env. Clone under
`external/` at the repo root (already gitignored):

```bash
cd external
```

**(a) LIBERO** — sim benchmark:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
touch libero/__init__.py libero/lifelong/models/modules/__init__.py   # missing in upstream
pip install -e .
pip install robosuite==1.4.0 bddl==1.0.1 robomimic==0.2.0 mujoco \
            gym==0.25.2 easydict==1.9 cloudpickle==2.1.0 future
cd ..
```

**(b) `dictionary_learning`** — SAE training library:

```bash
git clone https://github.com/saprmarks/dictionary_learning.git
pip install -e ./dictionary_learning
```

Tested against commit `60ec6bf`. If upstream breaks, pin it:
`(cd dictionary_learning && git checkout 60ec6bf)`.

**(c) AWE** — kinematic keyframe extraction (fork with packaging
fixes; see the fork's NOTICE):

```bash
git clone https://github.com/xc-j/awe.git
pip install -e ./awe
```

## Usage

The pipeline has four stages: **(1)** SAE training, **(2)** kinematic
keyframe extraction, **(3)** event clustering with VLM annotation,
**(4)** closed-loop intervention. Between (3) and (4) a feature
ranking step picks candidate features to intervene on.

Every command below chains off a single rollout. Step (a) creates a
timestamped run directory under `logs/openvla/`; later commands
reference it via `$EVAL_RUN` (the export is shown after step (a)).
Later commands also reference `$SAE_CKPT` — set it in step (b) to
either a freshly-trained checkpoint or a pre-trained one.

## Phase 1 — SAE training

Roll out openVLA on LIBERO, save the residual-stream activations, then
train a BatchTopK SAE on them.

### (a) Collect openVLA activations during a LIBERO rollout

Edit `configs/examples/openvla/collect_libero_spatial.yaml` (or copy
and adapt). Requires GPU and LIBERO assets (`LIBERO_CONFIG_PATH`).

```bash
python scripts/openvla/collect_activations.py \
    --config configs/examples/openvla/collect_libero_spatial.yaml
```

This creates a timestamped run directory. Export its name so later
steps can derive their paths:

```bash
export EVAL_RUN=EVAL-libero_spatial-openvla-<DATE_TIME>   # name of the dir under logs/openvla/
```

Outputs under `logs/openvla/$EVAL_RUN/sae_activations/`:
- dense `.pt` shards — input to step (b)
- `activation_index.jsonl` — input to step (h)

### (b) Train an SAE on collected shards

Edit `configs/examples/openvla/train_sae_layer31.yaml` and point
`data_dir` at the shard directory from step (a). Set `wandb_project`
to log to wandb, or leave empty to disable.

```bash
python scripts/train_sae.py \
    --config configs/examples/openvla/train_sae_layer31.yaml \
    --save-dir logs/openvla/sae/libero_spatial_layer31
```

Output: `ae.pt` + `config.json` under
`logs/openvla/sae/libero_spatial_layer31/trainer_0/`.

Or skip step (b) and use the paper's four pre-trained SAEs (one per
LIBERO suite, each at openVLA layer 31, BatchTopK k=64) from the
[Hugging Face Hub](https://huggingface.co/mr-cabbage/event-sae-openvla-libero).
Set `$SAE_CKPT` to either the HF download or the local training
output:

```bash
# Pretrained, e.g. LIBERO-Spatial:
SAE_CKPT=$(hf download mr-cabbage/event-sae-openvla-libero libero_spatial/ae.pt)

# Or locally trained:
SAE_CKPT=logs/openvla/sae/libero_spatial_layer31/trainer_0/ae.pt
```

All subsequent commands in this doc reference `--sae-checkpoint
$SAE_CKPT`.

### (b2) Evaluate offline SAE fidelity

This is an offline pass over existing dense shards; it does not run OpenVLA or
create new rollouts. The command reports global FVE, alive-feature percentage,
element-wise reconstruction MSE, and average L0. Omit `--max-rows` for the
final full-shard result.

```bash
python scripts/evaluate_sae.py \
    --data-dir logs/openvla/$EVAL_RUN/sae_activations/post_mlp_residual \
    --sae-checkpoint "$SAE_CKPT" \
    --output logs/openvla/sae/libero_spatial_layer31/offline_fidelity.json \
    --device cuda:0
```

The final checkpoint already operates at the raw activation scale. Do not
divide evaluation shards by the training norm factor again. If the same
shards were used for training, the result is in-sample fidelity.

## Phase 2 — Kinematic keyframe extraction

Pick a small number of waypoints per episode from the end-effector
trajectory. These waypoints anchor the events used in Phase 3 and are
independent of the SAE.

### (c) Extract AWE kinematic keyframes from rollout trajectories

CPU-only. Defaults (`pos_only`, error budget η = 0.05) are baked
into the CLI.

```bash
python scripts/extract_keyframes.py \
    --trajectory-records-path logs/openvla/$EVAL_RUN/trajectory_records.jsonl
```

Output: `waypoint_summary.json` under
`logs/openvla/keyframes/$EVAL_RUN/dp_pos_only_err0p05/`.

## Phase 3 — Event clustering with VLM annotation

Group waypoint windows into per-task event clusters, then ask Gemini
to label each cluster with a short phrase and one of six phase tags
(`pre_grasp`, `immobilization`, `contact`, `detach`, `post_grasp`,
`transition`).

### (d) Render 5-frame bundles around each keyframe

Save 5 PNG frames per waypoint at offsets `-4, -2, 0, 2, 4` plus a
short MP4 over the same window. Requires step (a) to have saved
rollout videos (`logging.save_video: true`).

```bash
python scripts/extract_keyframe_media.py \
    --waypoint-summary-path logs/openvla/keyframes/$EVAL_RUN/dp_pos_only_err0p05/waypoint_summary.json
```

Outputs under `logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/`:
- per-sample PNG frames — input to step (e)
- per-sample MP4 clips — for human inspection
- `samples.jsonl` — sample manifest

### (e) Build vision embeddings + state vectors per sample

Encode each 5-frame bundle through a frozen vision encoder (default
SigLIP), L2-normalize, then concatenate the end-effector pose at the
waypoint. Requires GPU.

```bash
python scripts/build_event_features.py \
    --samples-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/samples.jsonl
```

Output: `event_features.jsonl` next to `samples.jsonl`.

### (f) Task-local agglomerative clustering of event features

Cluster samples per task by cosine-distance agglomerative clustering
on the weighted [vision, state, progress] descriptor. CPU-only, runs
in seconds. Defaults: cosine threshold 0.18, weights 1.0 / 0.5 / 0.4,
5 exemplars per cluster.

```bash
python scripts/cluster_events.py \
    --event-features-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/event_features.jsonl
```

Outputs under `clusters/` next to `event_features.jsonl`:
- `cluster_assignments.jsonl` — sample → cluster_id map
- `clusters.jsonl` — per-cluster members + exemplars
- `summary.json` — overall stats

### (g) Annotate clusters with Gemini

Send each cluster's representative 5-frame sequences to Gemini and
parse a `{phrase, phase}` JSON response. `phase` is one of the six
tags from the Phase 3 intro. Default model: `gemini-2.5-flash`
(override with `--model`). The paper used a stronger Gemini model;
this default keeps annotation cost low for reproduction. Cluster
labels are descriptive only and do not affect feature ranking or
the intervention results.

```bash
export GEMINI_API_KEY=<your-key>
python scripts/annotate_clusters.py \
    --clusters-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/clusters/clusters.jsonl
```

Output: `gemini-2_5-flash_cluster_annotations.jsonl` next to
`clusters.jsonl`.

## Feature ranking (bridge between Phase 3 and Phase 4)

Encode the saved activations through the trained SAE, then score each
event cluster against the SAE features. The result is a ranked list of
candidate features for Phase 4.

### (h) Top-k SAE encoding (offline)

Apply the SAE to the dense shards from step (a) and write sparse
top-k shards. Encoding is decoupled from collection, so re-encoding
with a different SAE or layer needs no fresh rollout.

```bash
python scripts/extract_topk.py \
    --dense-dir logs/openvla/$EVAL_RUN/sae_activations/post_mlp_residual \
    --sae-checkpoint $SAE_CKPT \
    --layer-idx 31 \
    --output-dir logs/openvla/$EVAL_RUN/topk_activations
```

Output: top-k shards + `manifest.json` under
`logs/openvla/$EVAL_RUN/topk_activations/`.

To skip step (h) entirely, use the **online top-k mode** in step
(a): set `sae_collect.mode: "topk"` and `sae_collect.sae_checkpoint:
<path>` in the YAML and the rollout writes sparse shards directly.
This requires an SAE checkpoint already available (from a prior
training run or the Hugging Face Hub).

### (i) Event-feature score matrix

For each VLM-labeled cluster, score every SAE feature on how
strongly its activation lines up with that cluster's events. The
score is the max projection onto three temporal templates — pulse,
step-up, step-down — inside a ±5-step window around each event,
averaged across episodes. CPU-only.

```bash
python scripts/score_cluster_features.py \
    --topk-run-dir logs/openvla/$EVAL_RUN/topk_activations \
    --prompt-records-path logs/openvla/$EVAL_RUN/prompt_records.jsonl \
    --event-features-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/event_features.jsonl \
    --cluster-assignments-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/clusters/cluster_assignments.jsonl \
    --cluster-annotations-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/clusters/gemini-2_5-flash_cluster_annotations.jsonl \
    --output-path logs/openvla/scores/$EVAL_RUN/event_feature_scores.pt
```

Output: one `.pt` payload — `(num_clusters, dict_size)` `matrix` plus
`row_keys`, `row_results`, `templates`, `selection_counts`,
`selected_events`, `source`.

### (j) Build candidate feature lists

Surface the top-K features under four ranking strategies. CPU-only.

```bash
python scripts/build_feature_rankings.py \
    --scores-pt logs/openvla/scores/$EVAL_RUN/event_feature_scores.pt \
    --topk-run-dir logs/openvla/$EVAL_RUN/topk_activations \
    --output-dir logs/openvla/rankings/$EVAL_RUN \
    --top-k 5
```

The four rankings:

- **event-aligned** — mean of the score matrix across canonical cluster
  rows.
- **window-mean** — per-row window-mean vectors weighted by event count,
  restricted to the same canonical cluster rows.
- **task-mean** — per-task feature means weighted by per-task step count.
- **random-alive** — uniform sample over alive features, excluding any
  feature already chosen by the three informed rankings.

Outputs under `--output-dir`:

- per-ranking JSONL: `event_aligned.jsonl`, `window_mean.jsonl`,
  `task_mean.jsonl`, `random_alive.jsonl`
- `candidates.jsonl` — flat list of `4 × K` `(ranking, rank,
  feature_id, score)` rows that feeds step (k)

## Phase 4 — Closed-loop intervention

Edit one SAE feature at inference time and check how the policy's
success rate changes. For selected feature `i` and scaling factor
`α`:

    z'_i = α · z_i        # selected feature, scaled
    z'_j = z_j            # all other features unchanged (j ≠ i)
    x'   = x + Dec(z') − Dec(z)

`α = 0` zeros the feature out, `α = 1` leaves the hidden state
unchanged, intermediate values give partial suppression, `α > 1`
amplifies. The SAE reconstruction error on the un-edited code is
preserved.

### (k) Run a single-feature intervention on LIBERO

The paper's main OpenVLA intervention sweep reuses the Section 4.1 budget:
50 trials per task, or 500 rollouts per feature across the 10-task suite.
Run a no-hook baseline under the same config,
deduplicate feature IDs across the four rankings, and resume completed features
rather than launching the flat 20-row candidate list blindly.

For one manually selected feature, the validated low-level command is:

```bash
python scripts/openvla/intervene.py \
    --config configs/reproduction/openvla/libero_spatial_intervention_layer31.yaml \
    --sae-checkpoint $SAE_CKPT \
    --layer-idx 31 \
    --feature-id <FEATURE_ID> \
    --alpha 0.0 \
    --expected-rollouts 500 \
    --expected-code-revision $EVENT_SAE_COMMIT \
    --result-output logs/openvla/intervention/feature-<FEATURE_ID>.json
```

For the baseline + deduplicated sweep, use
`scripts/openvla/run_intervention_sweep.py`; the exact command is in
[reproduce_libero_spatial_500.md](reproduce_libero_spatial_500.md).

For each ranking, take the mean of `SR_hook − SR_baseline` across
its K features — this is how much zeroing that ranking's features
hurts the policy.

Each intervention result records aggregate hook counts and feature activity.
Its JSONL evidence file contains one lightweight row per environment step;
offline fidelity remains the source for numerical reconstruction metrics.

## Frozen environment snapshot

`environment-openvla.lock.yml` is a pinned record of the conda + pip
package versions on our working machine. It is **not a working
installer** — editable external libraries are not included. Use it
to cross-check versions when Step 1 / Step 2 produces a different
env than expected.

# Event-SAE

Codebase for "Event-Grounded Sparse Autoencoders for
Vision-Language-Action Policies". The project scales sparse-autoencoder
feature labeling by anchoring candidate features in SAE-independent
kinematic events from closed-loop rollouts, ranking them against
VLM-labeled event clusters, and validating each ranking with
residual-preserving zero-out interventions.

**Paper:** https://arxiv.org/abs/2605.17204

Two VLA backbones are covered: **openVLA** and **openpi (π₀.₅)**, on
the LIBERO simulation suites.

## Pipeline

Four stages plus a ranking bridge, 11 numbered steps in total:

| Stage | Steps | Produces |
|---|---|---|
| 1. SAE training | a, b | activation shards → trained SAE |
| 2. Kinematic keyframes | c | AWE waypoints per episode |
| 3. Event clustering + VLM annotation | d–g | labeled event clusters |
| Feature ranking (bridge) | h, i, j | top-K candidate features per ranking |
| 4. Closed-loop intervention | k | per-feature ΔSR |

Stages 1–4 are shared across backbones; activation collection (a) and
the intervention hook (k) are backbone-specific. Per-backbone guides:

- **openVLA** — [docs/openvla.md](docs/openvla.md)
- **현재 Spatial-500 재현** — [docs/reproduce_libero_spatial_500.md](docs/reproduce_libero_spatial_500.md)
- **openpi (π₀.₅)** — [docs/openpi.md](docs/openpi.md)

Each guide includes installation, the full pipeline (steps a–k),
pretrained SAE checkpoints from the paper (on the Hugging Face Hub),
and a reproducibility check against the original research artifacts.

## Repository layout

Step letters in parentheses map to the `a`–`k` pipeline table above.

```
event_sae/                     core library
  sae.py                       BatchTopK SAE
  train.py                     SAE training (b)
  keyframes/extract.py         AWE kinematic keyframes (c)
  events/                      event clustering + VLM annotation (d–g)
    extract_media.py           5-frame bundles (d)
    build_features.py          vision + state embeddings (e)
    cluster.py                 task-local clustering (f)
    annotate.py, prompts.py    Gemini cluster annotation (g)
  scoring/                     feature scoring + ranking (h–j)
    score_matrix.py            event-feature score matrix (i)
    rankings.py                four ranking strategies (j)
  evaluate.py                  offline SAE fidelity (FVE, MSE, alive, L0)
  openvla/                     openVLA backbone: collection (a) + intervention (k)
  openpi/                      openpi backbone: collection (a) + intervention (k)
scripts/                       CLI entry points, one per step
  train_sae.py                          (b)
  evaluate_sae.py                       offline SAE fidelity
  extract_keyframes.py                  (c)
  extract_keyframe_media.py             (d)
  build_event_features.py               (e)
  cluster_events.py                     (f)
  annotate_clusters.py                  (g)
  extract_topk.py                       (h)
  score_cluster_features.py             (i)
  build_feature_rankings.py             (j)
  openvla/{collect_activations,intervene}.py   (a, k — openVLA)
  openpi/{serve_policy,eval_libero}.py         (a, k — openpi)
configs/examples/{openvla,openpi}/   example YAML configs
docs/{openvla,openpi}.md             per-backbone runbooks
environment-openvla.yml              conda env (openVLA)
environment-{openvla,openpi}.lock.yml  pinned snapshots
```

## License

MIT (see `LICENSE`). External libraries cloned under `external/` at
install time retain their own licenses.

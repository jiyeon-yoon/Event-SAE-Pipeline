# 확장 LIBERO 데이터 수집

기존 `collect_activations.py`는 수정하거나 호출하지 않는다. 새 진입점은
`scripts/openvla/collect_extended_dataset.py`이며, rollout·모델 로딩·Layer 31
hook·파일 writer는 `extended_collection/`에 별도로 구현되어 있다.

이 수집기는 Event-SAE baseline 재현 이후의 실패·subgoal 연구용이다.
기존 500-rollout 재현 데이터를 대체하지 않는다.

## 저장하는 데이터

- OpenVLA Layer 31 dense activation
- 실제 OpenVLA 입력 RGB와 rollout MP4
- initial state 원본·seed·SHA-256
- action 전/후 robot·EEF·gripper·object·fixture 상태
- MuJoCo integration state(`time/qpos/qvel/act/mocap`), control, force, sensor
  배열과 이름/index schema
- 모든 contact와 contact force, object별 grasp 판정, 소유 geom 기반 충돌 후보
- reward, success, BDDL goal predicate별 상태와 충족 비율
- 7D raw OpenVLA action과 LIBERO에 실제 전달한 action
- 7개 action 차원 각각의 전체 vocabulary entropy·top probability·margin
- OpenVLA 256개 action-token probability mass와 조건부 entropy·probability·margin

전체 logits, depth, segmentation, 다른 layer activation은 저장하지 않는다.
BDDL predicate는 각각 저장하지만, 순서가 정의된 semantic subgoal로 재해석하지 않는다.

## 1. RunPod에서 최신 코드 고정

기존 runtime image를 그대로 사용한다. 새 Python 의존성은 없으므로
Docker image를 다시 build할 필요는 없다. GitHub에 push한 실험용
40자 commit을 고정 checkout한다.

```text
ghcr.io/jiyeon-yoon/event-sae-runtime@sha256:ef895731ffd73985e1183c964b479bdcc7e97dfeabea6c8777b903c54ade1405
```

모델과 remote code는 config의 OpenVLA revision·code revision으로 고정되어
있다. 필수 telemetry preflight가 실패하거나 episode 중 예외의 발생하면
`strict_preflight=true`, `fail_fast=true`에 따라 수집을 중단한다.

```bash
event-sae-init
event-sae-verify --require-gpu

export EVENT_SAE_COMMIT=GITHUB에_PUSH한_40자_COMMIT
test "${#EVENT_SAE_COMMIT}" -eq 40 || { echo "COMMIT_REQUIRED"; exit 1; }

cd /workspace
test -d Event-SAE-Pipeline/.git || \
  git clone https://github.com/jiyeon-yoon/Event-SAE-Pipeline.git
git -C Event-SAE-Pipeline fetch origin main
test -z "$(git -C Event-SAE-Pipeline status --porcelain)" || \
  { echo "DIRTY_SOURCE"; exit 1; }
git -C Event-SAE-Pipeline checkout --detach "$EVENT_SAE_COMMIT"
test "$(git -C Event-SAE-Pipeline rev-parse HEAD)" = "$EVENT_SAE_COMMIT" || exit 1

cd /workspace/Event-SAE-Pipeline
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PIPELINE_COMMIT=$(git rev-parse HEAD)
echo "PIPELINE_COMMIT=$PIPELINE_COMMIT"
```

## 2. 코드 테스트

```bash
TQDM_DISABLE=1 PYTHONPATH=. pytest -q
```

테스트가 실패하면 GPU 수집을 시작하지 않는다.

## 3. 1-episode smoke test

1 episode로 파일 형식·LIBERO API·Layer 31 hook을 먼저 검증한다. 전체
수집 전에 반드시 한 번 실행하며, 전체 실험을 대체하지는 않는다.

```bash
tmux new -s extended-libero

cd /workspace/Event-SAE-Pipeline
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /workspace/results/extended-libero-smoke
SMOKE_LOG=/workspace/results/extended-libero-smoke/collect.log
set -o pipefail

python scripts/openvla/collect_extended_dataset.py \
  --config configs/research/openvla/collect_libero_spatial_extended_layer31.yaml \
  --override env.task_ids=0 \
  --override env.num_trials_per_task=1 \
  --override output.root_dir=/workspace/results/extended-libero-smoke \
  2>&1 | tee "$SMOKE_LOG"

SMOKE_STATUS=${PIPESTATUS[0]}
echo "SMOKE_EXIT_CODE=$SMOKE_STATUS"
test "$SMOKE_STATUS" -eq 0 || exit 1
```

정상 종료 시 `EXTENDED_COLLECTION_OK: ...`에 출력된 경로를 검증한다.

```bash
RUN_DIR=$(find /workspace/results/extended-libero-smoke -mindepth 1 -maxdepth 1 \
  -type d -name 'EXTENDED-libero_spatial-openvla-*' | sort | tail -1)
cp "$SMOKE_LOG" "$RUN_DIR/collect.log"
python scripts/openvla/validate_extended_dataset.py --run-dir "$RUN_DIR"
python -c "import json; m=json.load(open('$RUN_DIR/manifest.json')); assert m['code']['commit']=='$EVENT_SAE_COMMIT' and not m['code']['dirty']; print('SOURCE_OK:', m['code']['commit'])"
du -sh "$RUN_DIR"
```

`EXTENDED_DATASET_OK`가 반드시 출력되어야 한다. Smoke 결과의 용량을
확인한 후에만 전체 Pod의 disk를 결정한다. 확장 수집은 dense activation
외의 simulator·RGB 데이터도 저장하므로 기존 281 GB만으로 전체
용량을 예산하면 안 된다.

Smoke validator 요약에서는 다음을 확인한다.

```text
episodes: 1
policy_steps: 1 이상
activation_index_records: policy_steps × 7
activation_shards: 1 이상
```

## 4. 전체 LIBERO-Spatial 분할 수집

전체 범위는 10 tasks × 50 rollouts = 500 episodes다. 5개 Pod를 사용하며
각 Pod에서 1–2번의 checkout·test를 반복한 뒤 `tmux new -s extended-libero`를
실행한다. 그 tmux에서 아래 변수 한 줄만 자신의 Pod에 맞게
실행한다.

```bash
# Pod 1
PAIR=0-1; TASK_IDS=0,1

# Pod 2
PAIR=2-3; TASK_IDS=2,3

# Pod 3
PAIR=4-5; TASK_IDS=4,5

# Pod 4
PAIR=6-7; TASK_IDS=6,7

# Pod 5
PAIR=8-9; TASK_IDS=8,9
```

그 다음 각 Pod의 **같은 tmux 화면**에서 아래 공통 블록을 실행한다.

```bash
cd /workspace/Event-SAE-Pipeline
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RUN_ROOT="/workspace/results/extended-libero-tasks-$PAIR"
mkdir -p "$RUN_ROOT"
LOG_FILE="$RUN_ROOT/collect.log"
set -o pipefail

python scripts/openvla/collect_extended_dataset.py \
  --config configs/research/openvla/collect_libero_spatial_extended_layer31.yaml \
  --override env.task_ids="$TASK_IDS" \
  --override env.num_trials_per_task=50 \
  --override output.root_dir="$RUN_ROOT" \
  2>&1 | tee "$LOG_FILE"

COLLECT_STATUS=${PIPESTATUS[0]}
echo "COLLECT_EXIT_CODE=$COLLECT_STATUS"
test "$COLLECT_STATUS" -eq 0 || exit 1
```

RunPod 연결이 끊겨도 tmux 작업은 계속된다. 재접속할 때는
`tmux attach -t extended-libero`를 사용한다.

## 5. 로컬 산출물 검증

각 Pod에서 수집 종료 후 실행한다.

```bash
RUN_DIR=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 \
  -type d -name 'EXTENDED-libero_spatial-openvla-*' | sort | tail -1)
cp "$LOG_FILE" "$RUN_DIR/collect.log"
python scripts/openvla/validate_extended_dataset.py --run-dir "$RUN_DIR"
python -c "import json; m=json.load(open('$RUN_DIR/manifest.json')); assert m['code']['commit']=='$EVENT_SAE_COMMIT' and not m['code']['dirty']; print('SOURCE_OK:', m['code']['commit'])"
```

`EXTENDED_DATASET_OK`는 episode별 initial-state checksum, action 전/후 simulator
배열, RGB, video, uncertainty, activation-index의 **구조·shape·finite value·연결**을
검증했다는 의미다. 이 문구가 없으면 업로드하지 않는다.

## 6. Hugging Face 업로드·원격 검증

각 Pod의 `PAIR` 값에 따라 별도 dataset repository로 업로드한다. Token은
image에 넣지 않고 Pod에서 `hf auth login`으로 등록한다.

```bash
hf auth whoami || hf auth login
HF_REPO="jiyeony/event-sae-libero-spatial-extended-tasks-$PAIR"
export HF_REPO RUN_DIR

python -c "import os; from huggingface_hub import HfApi; HfApi().create_repo(repo_id=os.environ['HF_REPO'], repo_type='dataset', private=True, exist_ok=True)"
huggingface-cli upload-large-folder "$HF_REPO" "$RUN_DIR" --repo-type dataset

python scripts/openvla/verify_extended_dataset_upload.py \
  --run-dir "$RUN_DIR" \
  --repo-id "$HF_REPO"
```

원격 검증은 모든 로컬 파일의 상대경로·파일 크기를 Hugging Face와
비교한다. `REMOTE_EXTENDED_DATASET_OK` 출력을 확인한 후에만 Pod를
terminate한다.

## 7. 파일 정렬 기준

한 정책 step은 다음처럼 저장된다.

```text
pre state + raw OpenVLA action + executed LIBERO action → post state/reward/done
```

`episode_num`, `task_id`, `task_episode_idx`, `step_in_episode`가 RGB, simulator
NPZ, uncertainty, activation index를 연결하는 공통 키다. `manifest.json`과
`schemas/task_XX.json`에는 배열 이름·index와 수집 가능 여부가 기록된다.

핵심 출력 구조는 다음과 같다.

```text
RUN_DIR/
├── manifest.json
├── summary.json
├── prompt_records.jsonl
├── episode_results.jsonl
├── trajectory_records.jsonl
├── action_records.jsonl
├── policy_uncertainty.jsonl
├── initial_states/
├── sim_state/
├── vision/
├── videos/
├── schemas/
└── sae_activations/post_mlp_residual/
    ├── activation_index.jsonl
    └── layer_31_shard_*.pt
```

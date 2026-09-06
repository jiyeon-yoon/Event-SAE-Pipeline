# Paired normal / 강제-release 수집

이 수집기는 기존 Event-SAE 수집기를 수정하거나 호출하지 않는다. 같은
LIBERO initial state와 seed에서 `normal`을 먼저 실행한 뒤, 안정적인
grasp+lift가 확인된 시점에 gripper만 강제로 여는 `forced_release`를 실행한다.

한 번의 수집으로 normal/premature release 비교, Layer 31 raw-hidden probe,
SAE Top-K와 시간 지표, object/contact/reward/predicate 기반 실패 분석에 필요한
원천 데이터를 만든다. SAE Top-K는 dense Layer 31과 checkpoint에서 나중에
계산하므로 재수집하지 않는다. 정책 실행 중 feature를 직접 억제하는 closed-loop
intervention만 새 rollout이 필요하다.

각 task는 고정 횟수를 무조건 끝까지 실행하지 않는다.
`env.num_trials_per_task`를 최대 시도 횟수로 사용하며, 설정한 task별
`valid`와 `primary` 목표를 모두 채우면 해당 run을 즉시 종료한다.

- `valid`: 동일 초기 상태·prefix와 강제 detach가 기술적으로 검증된 pair
- `primary`: valid이면서 normal이 성공하고 자연 release도 관측된 주 분석용 pair

## 수집 데이터와 필요한 이유

| 수집 데이터 | 필요한 이유 |
|---|---|
| pair/task/seed, initial state 파일·hash, `normal`/`forced_release`, `t_cmd`·`t_detach`·`t_obs` | 같은 시작점의 정상 release와 조기 release를 정확히 정렬·비교 |
| object·fixture pose와 선/각속도, robot joint·EEF·gripper 상태 | 접근→grasp→lift→drop→placement의 물리 변화와 실패 위치 분석 |
| 활성 MuJoCo contact·contact force·geom owner, object별 grasp flag | 접촉·grasp·detach를 주관적 라벨 없이 판정 |
| reward·done·success·info, BDDL goal predicate의 step 전/후 만족 여부 | 성공 여부와 목표 진행 상태를 simulator 기준으로 판정 |
| qpos·qvel·act·ctrl·force·sensor 등 지정된 simulator vector의 step 전/후 값 | 두 조건의 intervention 이전 동역학이 같은지 검증하고 후처리 분석 |
| raw OpenVLA action, LIBERO 변환 action, 실제 실행 action, gripper override | 강제-release에서 gripper만 바뀌었는지 확인하고 행동 변화 추적 |
| token ID와 entropy·top-1 probability·margin 등 요약값 | full logits를 저장하지 않고도 정책 불확실성 분석 |
| model-input RGB와 rollout MP4 | 물리 이벤트를 실제 장면 및 시간축과 함께 확인 |
| Layer 31 dense hidden과 episode/condition/step/forward→shard index | raw-hidden probe, SAE Top-K, event 주변 feature와 시간 지표를 재수집 없이 계산 |

`subgoal`은 순서가 붙은 별도 라벨이 아니라 **BDDL predicate별 만족 상태**로
저장한다. 현재 수집하지 않는 항목은 추가 카메라 원본, depth, segmentation,
full logits, Layer 31 이외 activation, 순서화된 semantic subgoal label이다.
SAE Top-K는 Layer 31 dense hidden과 SAE checkpoint로 후처리하며, closed-loop
feature intervention 결과는 정책을 다시 실행해야 하므로 별도 rollout이 필요하다.

## 최근 추가한 수집·검증 로직

| 로직 | 추가한 이유 |
|---|---|
| task 하나당 별도 run | 한 task 실패 때문에 이미 완료한 다른 task까지 다시 수집하지 않기 위해 |
| task별 `valid`·`primary` 목표 | 일부 task의 쓸 수 있는 pair가 0개인데 전체 합계만 통과하는 문제 방지 |
| 최대 시도 횟수 + 목표 달성 시 조기 종료 | 실패가 많은 task의 무한 실행을 막고, 충분히 모이면 GPU 비용 절감 |
| `valid`와 `primary` 분리 | 기술적으로 올바른 pair와 실제 주 분석에 적합한 pair를 구분 |
| summary·pair·episode 원본 기록 교차검증 | 잘못된 집계나 조건 기록으로 quota가 거짓 통과하는 문제 방지 |
| 목표 미달 시 실패 상태·완료 마커 미생성 | 불완전 데이터를 정상 완료본으로 업로드·사용하지 않기 위해 |
| 전체 10개 task smoke test | 본 수집 전에 task별 trigger, normal 성공·자연 release, 저장 구조 문제 확인 |

현재 본 수집의 task별 `valid=20`, `primary=20`은 논문이 보장한 표본 수가
아니라 **분석 가능한 최소 cohort를 확보하기 위한 품질 gate**다. 목표 충족 시
조기 종료하는 adaptive cohort이므로 원 정책의 unbiased 성공률 추정에는 쓰지 않는다.

## 1. RunPod 준비

OpenVLA runtime image를 사용한다. 수집은 RTX 4090으로 가능하다. paired 데이터는
아직 실측 용량이 없으므로 두 task/Pod에는 Container Disk 300 GB를 권장한다.
Smoke test의 실제 사용량을 먼저 확인하고, 수집 중 `df -h /workspace`를 감시한다.
수집기는 시작 시 여유 80 GiB 미만이면 실행하지 않고, 실행 중 20 GiB 미만이면
불완전 상태로 중단한다. 값은 YAML에서 조정할 수 있지만 Smoke 실측 없이 낮추지
않는다.

```bash
set -euo pipefail
event-sae-init
event-sae-verify --require-gpu

export EVENT_SAE_COMMIT="여기에_git_rev_parse_HEAD로_확인한_40자_commit"
test "${#EVENT_SAE_COMMIT}" -eq 40

cd /workspace
if [ -d /workspace/Event-SAE-Pipeline/.git ]; then
  git -C /workspace/Event-SAE-Pipeline fetch origin main
else
  git clone https://github.com/jiyeon-yoon/Event-SAE-Pipeline.git
fi

cd /workspace/Event-SAE-Pipeline
git checkout --detach "$EVENT_SAE_COMMIT"
test "$(git rev-parse HEAD)" = "$EVENT_SAE_COMMIT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q \
  tests/test_controlled_release.py \
  tests/test_paired_pair_quota.py \
  tests/test_paired_release_config.py \
  tests/test_paired_release_validator.py
```

## 2. Smoke test

한 run에는 task 하나만 넣는다. 아래 `TASK_ID`를 0~9로 바꿔 각 task를
한 번씩 확인한다. 5개 Pod를 쓴다면 Pod마다 두 task를 순차 실행한다.

```bash
tmux new -s paired-release-smoke
```

`tmux` 화면이 열린 뒤 아래 블록을 실행한다.

```bash
cd /workspace/Event-SAE-Pipeline
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
TASK_ID=0
SMOKE_ROOT="/workspace/results/paired-release-smoke-task-$TASK_ID"
mkdir -p "$SMOKE_ROOT"
set -uo pipefail

python scripts/openvla/collect_paired_release_dataset.py \
  --config configs/research/openvla/collect_libero_spatial_paired_release_layer31.yaml \
  --override env.task_ids="$TASK_ID" \
  --override env.num_trials_per_task=5 \
  --override paired_release.target_valid_pairs_per_task=1 \
  --override paired_release.target_primary_pairs_per_task=1 \
  --override output.root_dir="$SMOKE_ROOT" \
  2>&1 | tee "$SMOKE_ROOT/collect.log"

PIPE_STATUSES=("${PIPESTATUS[@]}")
echo "COLLECT_EXIT_CODE=${PIPE_STATUSES[0]} TEE_EXIT_CODE=${PIPE_STATUSES[1]}"
test "${PIPE_STATUSES[0]}" -eq 0 || { echo "COLLECTION_FAILED"; exit 1; }
test "${PIPE_STATUSES[1]}" -eq 0 || { echo "LOG_WRITE_FAILED"; exit 1; }
```

완료 후 검증한다.

```bash
RUN_DIR=$(find "$SMOKE_ROOT" -mindepth 1 -maxdepth 1 \
  -type d -name 'EXTENDED-libero_spatial-paired-release-openvla-*' \
  | sort | tail -1)
test -n "$RUN_DIR"
cp "$SMOKE_ROOT/collect.log" "$RUN_DIR/collect.log"

python scripts/openvla/validate_paired_release_dataset.py \
  --run-dir "$RUN_DIR"
```

10개 task 모두 `PAIRED_RELEASE_DATASET_OK`가 나와야 전체 수집을 시작한다.
5회 안에 valid 1개와 primary 1개를 만들지 못한 task는 trigger와
target/destination뿐 아니라 normal 정책 실패와 자연 release 부족도 점검한다.
수집 종료 시에도 같은 semantic validator가 자동 실행되며, 통과한 run에만
`COLLECTION_COMPLETE`가 생성된다.

## 3. 전체 수집과 업로드

10개 task를 5개 Pod에 나누되 **각 task를 별도 run으로 순차 실행**한다.
두 번째 task가 실패해도 첫 번째 task의 검증·업로드 결과는 보존된다.

각 Pod에서 먼저 `tmux`를 열고, 열린 화면 안에서 자기 할당을 입력한다.

```bash
tmux new -s paired-release
```

```bash
# Pod 1: TASK_IDS=(0 1)
# Pod 2: TASK_IDS=(2 3)
# Pod 3: TASK_IDS=(4 5)
# Pod 4: TASK_IDS=(6 7)
# Pod 5: TASK_IDS=(8 9)
TASK_IDS=(0 1)  # 이 줄만 현재 Pod 할당으로 바꾼다.
```

그 직후 같은 `tmux` 화면에서 아래 공통 블록을 한 번 실행한다.

```bash
cd /workspace/Event-SAE-Pipeline
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
set -uo pipefail
hf auth whoami || { echo "HF_LOGIN_REQUIRED"; exit 1; }

for TASK_ID in "${TASK_IDS[@]}"; do
  RUN_ROOT="/workspace/results/paired-release-task-$TASK_ID"
  mkdir -p "$RUN_ROOT"

  set +e
  python scripts/openvla/collect_paired_release_dataset.py \
    --config configs/research/openvla/collect_libero_spatial_paired_release_layer31.yaml \
    --override env.task_ids="$TASK_ID" \
    --override env.num_trials_per_task=50 \
    --override paired_release.target_valid_pairs_per_task=20 \
    --override paired_release.target_primary_pairs_per_task=20 \
    --override output.root_dir="$RUN_ROOT" \
    2>&1 | tee "$RUN_ROOT/collect.log"

  PIPE_STATUSES=("${PIPESTATUS[@]}")
  set -e
  test "${PIPE_STATUSES[0]}" -eq 0 || { echo "TASK_${TASK_ID}_FAILED"; exit 1; }
  test "${PIPE_STATUSES[1]}" -eq 0 || { echo "LOG_WRITE_FAILED"; exit 1; }

  RUN_DIR=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 \
    -type d -name 'EXTENDED-libero_spatial-paired-release-openvla-*' \
    | sort | tail -1)
  test -n "$RUN_DIR"
  cp "$RUN_ROOT/collect.log" "$RUN_DIR/collect.log"

  python scripts/openvla/validate_paired_release_dataset.py \
    --run-dir "$RUN_DIR" \
    --min-valid-pairs-per-task 20 \
    --min-primary-pairs-per-task 20

  RUN_TAG=$(basename "$RUN_DIR" | sed -E 's/^.*openvla-//; s/_/-/g')
  HF_REPO="jiyeony/event-sae-libero-spatial-pr-task-${TASK_ID}-${RUN_TAG}"
  python -c "from huggingface_hub import HfApi; HfApi().create_repo(repo_id='$HF_REPO', repo_type='dataset', private=True, exist_ok=True)"
  huggingface-cli upload-large-folder "$HF_REPO" "$RUN_DIR" --repo-type dataset
  python scripts/openvla/verify_extended_dataset_upload.py \
    --run-dir "$RUN_DIR" \
    --repo-id "$HF_REPO"
done
```

각 task마다 `REMOTE_EXTENDED_DATASET_OK`가 나와야 한다. 최대 50회 안에 valid
20개와 primary 20개를 채우지 못하면 해당 task만 실패하며 완료 마커와 업로드가
생기지 않는다. 부분 run은 이어서 실행할 수 없으므로 원인을 고친 뒤 그 task만
다시 수집한다. adaptive cohort는 pair 분석용이며 원 정책의 unbiased 성공률
추정에는 사용하지 않는다.

## 핵심 출력

- `pair_results.jsonl`: pair 유효성, trigger, t_cmd/t_detach/t_obs
- `summary.json`: task별 시도·valid·primary 수와 목표 달성 여부
- `trajectory_records.jsonl`: object·robot·contact·grasp·reward·predicate
- `action_records.jsonl`: raw, policy, 실제 action과 gripper override
- `policy_uncertainty.jsonl`: entropy, top probability, margin
- `initial_states/`, `sim_state/`, `vision/`, `videos/`
- `sae_activations/post_mlp_residual/`: Layer 31 dense activation과 step index
- `COLLECTION_COMPLETE`: 정상 완료된 run에만 생성

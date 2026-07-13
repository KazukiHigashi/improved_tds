#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ROOT=${1:?run output directory is required}
PYTHON="$PROJECT_ROOT/.venv/bin/python"
TOTAL_TIMESTEPS=1000000
CHECKPOINT_FREQ=100000
STALE_SECONDS=7200
MIN_FREE_KIB=$((20 * 1024 * 1024))

mkdir -p "$RUN_ROOT"
SUPERVISOR_LOG="$RUN_ROOT/supervisor.log"

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$SUPERVISOR_LOG"
}

run_task() {
    local task=$1
    local seed=$((1000 + task))
    local task_dir="$RUN_ROOT/exp_scissor${task}"
    mkdir -p "$task_dir/checkpoints"
    printf 'running\n' > "$task_dir/status.txt"
    (
        cd "$PROJECT_ROOT" || exit 90
        export OMP_NUM_THREADS=3
        export MKL_NUM_THREADS=3
        export OPENBLAS_NUM_THREADS=3
        /usr/bin/time -v -o "$task_dir/resource_usage.txt" \
            "$PYTHON" -m improved_tds.experiments.train_sb3 \
            --env-id "ExpScissor${task}-v0" \
            --algorithm sac \
            --total-timesteps "$TOTAL_TIMESTEPS" \
            --seed "$seed" \
            --device cpu \
            --output "$task_dir/model" \
            --dataset "$task_dir/training_candidates.npz" \
            --checkpoints-dir "$task_dir/checkpoints" \
            --checkpoint-freq "$CHECKPOINT_FREQ" \
            --dataset-save-frequency 100 \
            > "$task_dir/train.log" 2>&1
    )
    local exit_code=$?
    printf '%s\n' "$exit_code" > "$task_dir/exit_code"
    if [[ $exit_code -eq 0 ]] && "$PYTHON" -c '
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
s = json.loads((d / "model_training_summary.json").read_text())
assert s["total_timesteps"] == 1_000_000
assert s["env_id"] == sys.argv[2]
assert (d / "model.zip").is_file()
' "$task_dir" "ExpScissor${task}-v0"; then
        printf 'completed\n' > "$task_dir/status.txt"
        return 0
    fi
    printf 'failed\n' > "$task_dir/status.txt"
    return 1
}

monitor_wave() {
    local pids=("$@")
    while :; do
        local alive=0
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
            fi
        done
        [[ $alive -eq 0 ]] && return 0

        local free_kib
        free_kib=$(df --output=avail "$RUN_ROOT" | tail -1)
        if (( free_kib < MIN_FREE_KIB )); then
            log "停止: disk空き容量が20 GiB未満"
            kill "${pids[@]}" 2>/dev/null || true
            return 1
        fi

        local now
        now=$(date +%s)
        for task_dir in "$RUN_ROOT"/exp_scissor*; do
            [[ -f "$task_dir/status.txt" ]] || continue
            [[ $(<"$task_dir/status.txt") == running ]] || continue
            if grep -Eqi '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$task_dir/train.log" 2>/dev/null; then
                log "停止: $task_dir のlogでNaN/Infを検出"
                kill "${pids[@]}" 2>/dev/null || true
                return 1
            fi
            local latest_mtime
            latest_mtime=$(find "$task_dir" -type f -printf '%T@\n' | sort -nr | head -1)
            latest_mtime=${latest_mtime%.*}
            if [[ -n "$latest_mtime" ]] && (( now - latest_mtime > STALE_SECONDS )); then
                log "停止: $task_dir の更新が2時間停止"
                kill "${pids[@]}" 2>/dev/null || true
                return 1
            fi
        done
        sleep 300
    done
}

run_wave() {
    local tasks=("$@")
    local pids=()
    log "wave開始: tasks=${tasks[*]}"
    for task in "${tasks[@]}"; do
        run_task "$task" &
        pids+=("$!")
    done
    local monitor_status=0
    monitor_wave "${pids[@]}" || monitor_status=$?
    local wave_status=$monitor_status
    for pid in "${pids[@]}"; do
        wait "$pid" || wave_status=1
    done
    if [[ $wave_status -ne 0 ]]; then
        log "wave失敗: tasks=${tasks[*]}。後続waveを開始しない"
        return 1
    fi
    log "wave完了: tasks=${tasks[*]}"
}

log "学習開始: 5 tasks, 100 epochs/task, 100 episodes/epoch, 100 steps/episode"
run_wave 1 2 || exit 1
run_wave 3 4 || exit 1
run_wave 5 || exit 1
log "全task学習完了"

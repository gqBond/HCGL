#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/main}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/cache}"
DEVICE="${DEVICE:-cuda:0}"
METHOD="${METHOD:-hero}"
DATASETS="${DATASETS:-DBLP IMDB Freebase}"
SEEDS="${SEEDS:-0 1 2 3 4}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}/logs" "${CACHE_DIR}"
export CGL_DATA_ROOT="${DATA_ROOT}"
export CGL_CACHE_ROOT="${CACHE_DIR}"

for dataset in ${DATASETS}; do
    case "${dataset}" in
        DBLP) num_tasks=2 ;;
        IMDB) num_tasks=5 ;;
        Freebase) num_tasks=7 ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            exit 2
            ;;
    esac

    dataset_dir="${DATA_ROOT}/${dataset}"
    for required_file in node.dat link.dat label.dat label.dat.test; do
        if [[ ! -f "${dataset_dir}/${required_file}" ]]; then
            echo "Missing dataset file: ${dataset_dir}/${required_file}" >&2
            exit 2
        fi
    done

    for seed in ${SEEDS}; do
        run_name="${dataset}_${METHOD}_seed${seed}"
        run_args=()
        if [[ "${dataset}" == "DBLP" ]]; then
            run_args+=(--cls-balance true)
        fi
        if [[ "${METHOD}" == "metacl_graph" ]]; then
            run_args+=(--patience 200)
        fi
        if [[ "${METHOD}" == "ftf_er" ]]; then
            run_args+=(--data-path "${CACHE_DIR}/ftf_er")
        fi
        echo "Running ${run_name}"
        "${PYTHON_BIN}" main.py \
            --dataset "${dataset}" \
            --method "${METHOD}" \
            --num-tasks "${num_tasks}" \
            --seed "${seed}" \
            --device "${DEVICE}" \
            --data-root "${DATA_ROOT}" \
            --output-dir "${OUTPUT_DIR}" \
            "${run_args[@]}" \
            "$@" 2>&1 | tee "${OUTPUT_DIR}/logs/${run_name}.log"
    done
done

"${PYTHON_BIN}" scripts/aggregate_results.py \
    --input-dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/summary.csv"

echo "Main experiment finished. Summary: ${OUTPUT_DIR}/summary.csv"

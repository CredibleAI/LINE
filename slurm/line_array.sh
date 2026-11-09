#!/bin/bash
#SBATCH --job-name LINE
#SBATCH --time 0-24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G
#SBATCH --output ./slurm/logs/line_array=%A_%a.log
#SBATCH --array 0-51%10

# Explain random neurons of the paper's (model, layer) pairs with LINE, as a SLURM array.
#
# Each pair is split into ceil(N_NEURONS / CHUNK_SIZE) tasks, so --array must cover
# len(JOBS) * ceil(N_NEURONS / CHUNK_SIZE) tasks: 26 * 2 = 52 (0-51) with the defaults.
# Update it when changing JOBS, N_NEURONS or CHUNK_SIZE. The %10 caps concurrent tasks.
#
# Run from the repository root:
#   mkdir -p slurm/logs
#   sbatch slurm/line_array.sh                                         # SDXL (paper default)
#   T2I_MODEL=black-forest-labs/FLUX.1-dev sbatch slurm/line_array.sh  # FLUX.1-dev

set -euo pipefail

if [ -f .env ]; then
  # Variables set at submission (e.g. `SAVE_DIR=... sbatch ...`) take precedence over .env.
  SUBMIT_ENV=$(export -p)
  set -a
  source .env
  set +a
  eval "${SUBMIT_ENV}"
else
  echo "Warning: .env file not found, HF_TOKEN may not be set."
fi

T2I_MODEL="${T2I_MODEL:-stabilityai/stable-diffusion-xl-base-1.0}"
LANGUAGE_MODEL="${LANGUAGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
N_NEURONS="${N_NEURONS:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
NEURON_SEED="${NEURON_SEED:-42}"
NUMBER_OF_IMAGES="${NUMBER_OF_IMAGES:-5}"
NUMBER_OF_ITERATIONS="${NUMBER_OF_ITERATIONS:-10}"
H_SIZE="${H_SIZE:-0}"
T2I_BATCH_SIZE="${T2I_BATCH_SIZE:-1}"
DATA_DIR="${DATA_DIR:-./data}"
SAVE_DIR="${SAVE_DIR:-results}"

# SDXL-Turbo is a one-step model; every other model uses its default number of steps.
if [[ "${T2I_MODEL}" == *sdxl-turbo* ]]; then
  NUM_INF_STEPS="${NUM_INF_STEPS:-1}"
else
  NUM_INF_STEPS="${NUM_INF_STEPS:-}"
fi

# Each entry: "<explained model> <layer>"
JOBS=(
  "resnet18 layer1"
  "resnet18 layer2"
  "resnet18 layer3"
  "resnet18 layer4"
  "resnet18 avgpool"
  "resnet18|imagenet layer1"
  "resnet18|imagenet layer2"
  "resnet18|imagenet layer3"
  "resnet18|imagenet layer4"
  "resnet18|imagenet avgpool"
  "resnet50 layer1"
  "resnet50 layer2"
  "resnet50 layer3"
  "resnet50 layer4"
  "resnet50 avgpool"
  "resnet50|imagenet layer1"
  "resnet50|imagenet layer2"
  "resnet50|imagenet layer3"
  "resnet50|imagenet layer4"
  "resnet50|imagenet avgpool"
  "vit_b_16 encoder.layers.encoder_layer_6"
  "vit_b_16 encoder.layers.encoder_layer_8"
  "vit_b_16 encoder.layers.encoder_layer_10"
  "vit_b_16 encoder.layers.encoder_layer_11.ln_1"
  "vit_b_16 encoder.ln"
  "dinov3-vitb16 norm"
)

N_CHUNKS=$(( (N_NEURONS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
JOB_IDX=$(( SLURM_ARRAY_TASK_ID / N_CHUNKS ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % N_CHUNKS ))
if [ "${JOB_IDX}" -ge "${#JOBS[@]}" ]; then
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} is out of range (${#JOBS[@]} pairs x ${N_CHUNKS} chunks)."
  exit 1
fi
read -r EXPLAINED_MODEL LAYER <<< "${JOBS[$JOB_IDX]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${EXPLAINED_MODEL} / ${LAYER}, chunk ${CHUNK_IDX}/${N_CHUNKS}, T2I ${T2I_MODEL}"
echo "Host: $(hostname), GPU(s): ${CUDA_VISIBLE_DEVICES:-none}"

NEURONS_SPEC=$(uv run sample_neurons.py \
  --explained_model "${EXPLAINED_MODEL}" \
  --layer "${LAYER}" \
  --n "${N_NEURONS}" \
  --seed "${NEURON_SEED}" \
  --chunk_size "${CHUNK_SIZE}" \
  --chunk_index "${CHUNK_IDX}")
echo "Neurons: ${NEURONS_SPEC}"

EXTRA_ARGS=()
if [ -n "${NUM_INF_STEPS}" ]; then
  EXTRA_ARGS+=(--num_inf_steps "${NUM_INF_STEPS}")
fi

uv run main.py \
  --t2i_model "${T2I_MODEL}" \
  --language_model "${LANGUAGE_MODEL}" \
  --imagenet_data_path "${DATA_DIR}/ImageNet-1K" \
  --control_activations_path "${DATA_DIR}/control_activations" \
  --explained_model "${EXPLAINED_MODEL}" \
  --neurons "${NEURONS_SPEC}" \
  --num_img "${NUMBER_OF_IMAGES}" \
  --n_iters "${NUMBER_OF_ITERATIONS}" \
  --h_size "${H_SIZE}" \
  --t2i_batch_size "${T2I_BATCH_SIZE}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --save_histories \
  --save_images \
  --save_dir "${SAVE_DIR}/${EXPLAINED_MODEL}/${T2I_MODEL##*/}"

echo "Finished task ${SLURM_ARRAY_TASK_ID}"

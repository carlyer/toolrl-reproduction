#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.cache" "$ROOT/logs" "$ROOT/results"
exec 9>"$ROOT/.cache/gpu-training.lock"
flock 9
source "$ROOT/configs/models.sh"
MODE="${1:-smoke}"
if [[ "$MODE" != smoke && "$MODE" != full ]]; then echo 'usage: train.sh [smoke|full] [Hydra overrides...]'; exit 2; fi
shift || true
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/.cache/datasets}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export EXPERIMENT_NAME="toolrl-qwen2.5-3b-$MODE-$(date -u +%Y%m%dT%H%M%SZ)"
export WANDB_MODE=offline
for var in WITHLENGTH REFINEDREWARD COARSEREWARD STRICTMATCH CORRECTMAX1 MAX1STEP30MAX3 SCHEDULEREWARD SCHEDULELENGTH INTERMEDIATEREWARD; do export "$var=0"; done
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Run scripts/setup.sh first'; exit 1; }
cd "$ROOT/upstream"
extra=()
if [[ "$MODE" == smoke ]]; then extra=(data.train_batch_size=8 data.val_batch_size=8 actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size=8 trainer.total_training_steps=1 +trainer.val_before_train=False trainer.save_freq=-1 trainer.save_final_checkpoint=False trainer.test_freq=-1); fi
"$ROOT/.venv/bin/python" -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo data.train_files="$ROOT/upstream/dataset/rlla_4k/train.parquet" data.val_files="$ROOT/upstream/dataset/rlla_4k/test.parquet" \
 data.train_batch_size=512 data.val_batch_size=128 data.max_prompt_length="${MAX_PROMPT_LENGTH:-4608}" data.max_response_length=1024 \
 actor_rollout_ref.model.path="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}" actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_mini_batch_size=128 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.grad_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True algorithm.kl_ctrl.kl_coef=0.0 trainer.critic_warmup=0 \
 "trainer.logger=['console']" trainer.project_name=ToolRL trainer.experiment_name="$EXPERIMENT_NAME" \
 trainer.n_gpus_per_node="${N_GPUS:-4}" trainer.nnodes=1 trainer.save_freq=-1 trainer.save_final_checkpoint=True trainer.test_freq=5 trainer.total_epochs=15 \
 trainer.default_local_dir="$ROOT/results/checkpoints/$EXPERIMENT_NAME" "${extra[@]}" "$@" \
 2>&1 | tee "$ROOT/logs/$EXPERIMENT_NAME-$(date -u +%Y%m%dT%H%M%SZ).log"

# --- log directory: every stage below tees its output here ---
LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$LOGDIR"
echo "logs -> $LOGDIR"

cd Online-Diffusion-Planning/Pretrain

#pretrain planner
CUDA_VISIBLE_DEVICES=0 python pretrain_script4.py 2>&1 | tee "$LOGDIR/1_pretrain.log"

#train reward
CUDA_VISIBLE_DEVICES=0 python train_reward_script.py 2>&1 | tee "$LOGDIR/2_reward.log"

#train kernel
CUDA_VISIBLE_DEVICES=0 python train_kernel_script.py 2>&1 | tee "$LOGDIR/3_kernel.log"

cd Online-Diffusion-Planning/Finetuning
#train critic
CUDA_VISIBLE_DEVICES=0 python train_critic_script.py 2>&1 | tee "$LOGDIR/4_critic.log"

#warm up the critic 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  --multi_gpu --num_processes=4  train_critic_script2.py 2>&1 | tee "$LOGDIR/5_critic_warmup.log"


#finetune 
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_TIMEOUT=1000000
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

accelerate launch --multi_gpu --num_processes=4   finetune_script2.py 2>&1 | tee output.txt "$LOGDIR/6_finetune.log"


#rollout
python Rollout2.py 2>&1 | tee "$LOGDIR/7_rollout2.log"

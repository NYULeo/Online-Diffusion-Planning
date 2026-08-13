cd Online-Diffusion-Planning/Pretrain

#pretrain planner
CUDA_VISIBLE_DEVICES=0 python pretrain_script4.py

#train reward
CUDA_VISIBLE_DEVICES=0 python train_reward_script.py

#train kernel
CUDA_VISIBLE_DEVICES=0 python train_kernel_script.py

cd Online-Diffusion-Planning/Finetuning
#train critic
CUDA_VISIBLE_DEVICES=0 python train_critic_script.py

#warm up the critic 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  --multi_gpu --num_processes=4  train_critic_script2.py


#finetune 
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_TIMEOUT=1000000
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

accelerate launch --multi_gpu --num_processes=4   finetune_script2.py | tee output.txt


#rollout
python Rollout2.py

#!/bin/bash  

# env variables for DDP training
# These lines correctly set defaults if the variables are not already set by the environment
[ -z "${MLP_WORKER_0_HOST}" ] && MLP_WORKER_0_HOST=127.0.0.1 # master IP address
[ -z "${MLP_WORKER_0_PORT}" ] && MLP_WORKER_0_PORT=12345 # master port number
[ -z "${MLP_ROLE_INDEX}" ] && MLP_ROLE_INDEX=0 # current worker index
[ -z "${MLP_WORKER_NUM}" ] && MLP_WORKER_NUM=1 # total number of workers (Default to 1 if not set)
[ -z "${MLP_WORKER_GPU_NUM}" ] && MLP_WORKER_GPU_NUM=$(nvidia-smi -L | wc -l) # automatically get local GPU count

# Assume MLP_WORKER_NUM is always set to at least 1 now
if (( MLP_WORKER_NUM == 1 ))
then
  # Single node, single GPU or multiple GPUs on one node
  # master_port is only strictly needed for multi-node, but harmless for single-node multi-GPU
  DISTRIBUTED_ARGS="--nproc_per_node $MLP_WORKER_GPU_NUM \
                    --master_port $MLP_WORKER_0_PORT" # Master port is often good practice even for single node multi-GPU
else
  # Multi-node distributed training
  DISTRIBUTED_ARGS="--nproc_per_node $MLP_WORKER_GPU_NUM \
                    --nnodes $MLP_WORKER_NUM \
                    --node_rank $MLP_ROLE_INDEX \
                    --master_addr $MLP_WORKER_0_HOST \
                    --master_port $MLP_WORKER_0_PORT" # Add master_port for consistency and robustness
fi
echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"

# export WANDB_API_KEY=$YOUR WANDB API KEY$ # replace with your own wandb api key
export WANDB_PROJECT="Enformer_Pretrain"

[ -z "${learning_rate}" ] && learning_rate=5e-4
[ -z "${shift_aug}" ] && shift_aug=True
[ -z "${rc_aug}" ] && rc_aug=True
[ -z "${seqlen}" ] && seqlen=131072

max_steps=150_000
weight_decay=1e-2
per_device_batch_size=8
gradient_accumulation_steps=1
dataloader_num_workers=16
report_to="wandb"
total_batch_size=$(($per_device_batch_size * $MLP_WORKER_GPU_NUM * $MLP_WORKER_NUM * $gradient_accumulation_steps)) # this is only suitable for single node trainings
run_name="Enformer_lr_${learning_rate}_bs_${total_batch_size}_len_${seqlen}"
echo "${run_name}"
output_path="/vepfs-mlp/mlp-public/user/yangzhao/outputs/enformer/${run_name}" # replace with your own output path

torchrun ${DISTRIBUTED_ARGS} train_enformer.py \
    --run_name ${run_name} \
    --save_steps 5000 \
    --eval_steps 5000 \
    --logging_steps 100 \
    --max_steps ${max_steps} \
    --warmup_steps 5000 \
    --report_to ${report_to} \
    --output_dir ${output_path} \
    --weight_decay ${weight_decay} \
    --optim adamw_torch \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type cosine \
    --dataloader_num_workers ${dataloader_num_workers} \
    --save_strategy steps \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --remove_unused_columns False \
    --per_device_train_batch_size ${per_device_batch_size} \
    --per_device_eval_batch_size ${per_device_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --save_safetensors False \
    --shift_aug ${shift_aug} \
    --rc_aug ${rc_aug} \
    --seqlen ${seqlen} \
    --bf16 

nohup python ~/oc_gpu.py > ~/oc_gpu.log 2>&1 &
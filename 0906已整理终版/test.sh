

CKPTs=("")

values=(1.00)
target_height="0"
terrain="Rural" # Suburban, DenseUrban, Rural, OrdinaryUrban
epochs=350 
rerank_force=64
lr=2e-4
train_percent=0.8
num_sample=400
ddim=30
single_height=True

DATASET_ROOT="/data/dataset/chenghaotong/SpectrumNet"
ACCELERATE_CONFIG="./acc_config/test.yaml"
MODEL_CONFIG="./configs/SpectrumNet.yml"

for i in "${CKPTs[@]}"; do
    echo "Start Testing!"

    CUDA_VISIBLE_DEVICES=1,3 accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        test.py \
        --ckpt_path "${i}" \
        --desc "(Eval)epoch${epoch}_LR${lr}_${terrain}_DDIM${ddim}_numsample${num_sample}_force${rerank_force}" \
        --config_file "${MODEL_CONFIG}" \
        \
        MODEL.DIFFUSION_STEPS 1000 \
        MODEL.DDIM_STEPS ${ddim} \
        MODEL.GLOBAL_TIME_DIM 256 \
        MODEL.SINGLE_HEIGHT ${single_height} \
        \
        INPUT.DATASET_ROOT "${DATASET_ROOT}" \
        INPUT.HYBRID_GEO "['${terrain}']" \
        INPUT.SAMPLE_EACH_HEIGHT_SEPARATELY False \
        INPUT.NUM_SAMPLE ${num_sample} \
        \
        SOLVER.EPOCHS ${epochs} \
        SOLVER.WARMUP_EPOCHS 5 \
        SOLVER.BATCH_SIZE 32 \
        SOLVER.LOG_ITER 10 \
        \
        CONDITION.TARGET_HEIGHT "'${target_height}'" \
        CONDITION.TARGET_FREQUENCY "f03" \
        \
        TEST.BATCH_SIZE 1 \
        TEST.RERANK_FORCE ${rerank_force}
done

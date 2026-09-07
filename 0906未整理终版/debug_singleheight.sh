values=(1.00)
target_height="0"
terrain="Rural"
epochs=350
rerank_force=1 #64
lr=2e-4
batch_size=32
train_percent=0.8
num_sample=500 #200

DATASET_ROOT="/data/dataset/chenghaotong/SpectrumNet"
ACCELERATE_CONFIG="./acc_config/train.yaml"
MODEL_CONFIG="./configs/SpectrumNet.yml"

for i in "${values[@]}"; do
echo "Start Training!"

CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    train.py \
    --desc "DEBUG_SingleHeight_epoch${epochs}_LR${lr}_${terrain}" \
    --config_file "${MODEL_CONFIG}" \
    \
    MODEL.DIFFUSION_STEPS 1000 \
    MODEL.DDIM_STEPS 10 \
    MODEL.GLOBAL_TIME_DIM 256 \
    \
    INPUT.DATASET_ROOT "${DATASET_ROOT}" \
    INPUT.HYBRID_GEO "['${terrain}']" \
    INPUT.SAMPLE_EACH_HEIGHT_SEPARATELY False \
    INPUT.TRAIN_PERCENT ${train_percent} \
    INPUT.NUM_SAMPLE ${num_sample} \
    \
    SOLVER.EPOCHS ${epochs} \
    SOLVER.WARMUP_EPOCHS 5 \
    SOLVER.BATCH_SIZE ${batch_size} \
    SOLVER.LR ${lr} \
    \
    CONDITION.TARGET_HEIGHT "'${target_height}'" \
    CONDITION.TARGET_FREQUENCY "f03" \
    \
    TEST.BATCH_SIZE 1 \
    TEST.RERANK_FORCE ${rerank_force} \
    \
    DEBUG.OVERFIT False \
    DEBUG.SINGLE_HEIGHT_BASELINE True
done

# NUM_SAMPLE = 200
# 200 / (3 * 128 * 128) ≈ 0.4%
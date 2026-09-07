values=(1.00)
terrain="Rural"
epochs=2000
rerank_force=64
lr=2e-4
train_percent=0.8
num_sample=200

DATASET_ROOT="/data/dataset/chenghaotong/SpectrumNet"
ACCELERATE_CONFIG="./acc_config/train.yaml"
MODEL_CONFIG="./configs/SpectrumNet.yml"

for i in "${values[@]}"; do
    echo "Start Training!"

    CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        train.py \
        --desc "OVERFIT8_H0_epoch${epochs}_LR${lr}_${terrain}" \
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
        SOLVER.BATCH_SIZE 8 \
        SOLVER.NUM_WORKERS 0 \
        SOLVER.LR ${lr} \
        \
        CONDITION.TARGET_HEIGHT "Hybrid" \
        CONDITION.TARGET_FREQUENCY "f03" \
        \
        DEBUG.OVERFIT True \
        DEBUG.NUM_SAMPLES 8 \
        DEBUG.HEIGHT_IDX 0 \
        DEBUG.SAMPLE_SEED 20260905 \
        \
        TEST.BATCH_SIZE 1 \
        TEST.RERANK_FORCE ${rerank_force}
done

# NUM_SAMPLE = 200, # 200 / (3 * 128 * 128) ~ 0.4 %
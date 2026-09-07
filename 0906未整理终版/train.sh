
values=(1.00)
terrain="Rural" # Suburban, DenseUrban, Rural, OrdinaryUrban
epochs=350 # best epoch setting
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
        --desc "epoch${epochs}_LR${lr}_Rerank${rerank_force}_${terrain}" \
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
        SOLVER.BATCH_SIZE 32 \
        SOLVER.LR ${lr} \
        \
        CONDITION.TARGET_HEIGHT "Hybrid" \
        CONDITION.TARGET_FREQUENCY "f03" \
        \
        TEST.BATCH_SIZE 1 \
        TEST.RERANK_FORCE ${rerank_force}
done

# NUM_SAMPLE = 50, # 50 / (3 * 128 * 128) ~ 0.1 %


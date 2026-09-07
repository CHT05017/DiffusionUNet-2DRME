

CKPTs=("/home/chenghaotong/RadioMap2/_CKPTs/2026-0906-03-44-29-DEBUG_SingleHeight_epoch350_LR2e-4_Rural/latest.pth")

values=(1.00)
terrain="Rural" # Suburban, DenseUrban, Rural, OrdinaryUrban
epochs=350 # best epoch setting
rerank_force=64
lr=2e-4
train_percent=0.8
num_sample=20
ddim=30

DATASET_ROOT="/data/dataset/chenghaotong/SpectrumNet"
ACCELERATE_CONFIG="./acc_config/test.yaml"
MODEL_CONFIG="./configs/SpectrumNet.yml"

for i in "${CKPTs[@]}"; do
    echo "Start Testing!"

    CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        test.py \
        --ckpt_path "${i}" \
        --desc "(Eval)epoch${epoch}_LR${lr}_${terrain}_DDIM${ddim}_numsample${num_sample}_force${rerank_force}" \
        --config_file "${MODEL_CONFIG}" \
        \
        MODEL.DIFFUSION_STEPS 1000 \
        MODEL.DDIM_STEPS ${ddim} \
        MODEL.GLOBAL_TIME_DIM 256 \
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
        CONDITION.TARGET_HEIGHT "'0'" \
        CONDITION.TARGET_FREQUENCY "f03" \
        \
        TEST.BATCH_SIZE 1 \
        TEST.RERANK_FORCE ${rerank_force} \
        \
        DEBUG.OVERFIT False \
        DEBUG.SINGLE_HEIGHT_BASELINE True
done

# NUM_SAMPLE = 50, # 50 / (3 * 128 * 128) ~ 0.1 %



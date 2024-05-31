python training_benchmark.py \
    --device xpu \
    --models=gcn \
    --datasets=Reddit \
    --num-workers=0 \
    --batch-sizes=512 \
    --num-layers=2 \
    --num-hidden-channels=64 --num-steps=50
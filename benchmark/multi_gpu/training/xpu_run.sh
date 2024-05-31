# export CCL_WORKER_COUNT=1

mpirun -n 2 \
    -genv OMP_NUM_THREADS=55 \
    python training_benchmark_xpu.py --dataset Reddit --model gcn --num-epochs 1

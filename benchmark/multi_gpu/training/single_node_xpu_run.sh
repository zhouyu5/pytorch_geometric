set -x

np=2
OMP_NUM_THREADS=48

mpirun \
    -genv OMP_NUM_THREADS=$OMP_NUM_THREADS \
    -n $np \
    -print-rank-map \
    -verbose \
    python training_benchmark_xpu.py --dataset Reddit --model gcn --num-epochs 1


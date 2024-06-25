set -x

hosts=vsr215,vsr214
np=4
ppn=2
OMP_NUM_THREADS=48
master_addr=`hostname`
master_port=11111

mpirun \
    -genv OMP_NUM_THREADS=$OMP_NUM_THREADS \
    -genv MASTER_ADDR=$master_addr \
    -genv MASTER_PORT=$master_port \
    -n $np -ppn $ppn \
    -hosts $hosts \
    -print-rank-map \
    -verbose \
    python training_benchmark_xpu.py --dataset Reddit --model gcn --num-epochs 1

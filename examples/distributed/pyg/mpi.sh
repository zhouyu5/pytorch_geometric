set -x

master_addr=`hostname`
# hosts format: hostname1,hostname2,...
hosts=vsr215,vsr214
hosts_num=2
OMP_NUM_THREADS=48
ppn=1

PYG_WORKSPACE=/work/examples/distributed/pyg
EXEC_SCRIPT=${PYG_WORKSPACE}/"node_ogb_cpu_2.py"
DATASET=ogbn-products
DATASET_ROOT_DIR="/work/data/partitions/${DATASET}/${hosts_num}-parts"
LOG_PATH=${PYG_WORKSPACE}/data/log_${DATASET}_${hosts_num}_$(date +%H%M)
CMD="python ${EXEC_SCRIPT} --dataset=${DATASET} \
     --num_epochs=1 --batch_size=1024 --num_neighbors=5,5,5 \
     --num_workers=2 --concurrency=4 --ddp_port=11111 \
     --ddp_backend=gloo \
     --dataset_root_dir=${DATASET_ROOT_DIR} --master_addr=${master_addr}"

mpirun \
    -genv OMP_NUM_THREADS=$OMP_NUM_THREADS \
    -n $hosts_num -ppn $ppn \
    -hosts $hosts \
    -print-rank-map \
    -verbose \
    ${CMD}

    
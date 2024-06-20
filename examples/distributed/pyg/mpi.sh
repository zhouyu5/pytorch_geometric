set -x

master_addr=`hostname`
# hosts format: hostname1,hostname2,...
hosts=vsr215,vsr214
hosts_num=2
OMP_NUM_THREADS=48
ppn=1

PYG_WORKSPACE=/work/examples/distributed/pyg
EXEC_SCRIPT="node_ogb_cpu_2.py"
DATASET=ogbn-products
DATASET_ROOT_DIR="/work/data/partitions/${DATASET}/${hosts_num}-parts"
LOG_PATH=${PYG_WORKSPACE}/data/log_${DATASET}_${hosts_num}_$(date +%H%M)
CMD="cd ${PYG_WORKSPACE}; python ${EXEC_SCRIPT} --dataset=${DATASET} --dataset_root_dir=${DATASET_ROOT_DIR} --master_addr=${master_addr}"

mpirun \
    -genv OMP_NUM_THREADS=$OMP_NUM_THREADS \
    -n $hosts_num -ppn $ppn \
    -hosts $hosts \
    -print-rank-map \
    -verbose \
    ${CMD} | tee $LOG_PATH 

    
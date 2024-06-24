#!/bin/bash

rm dist_cpu-node0.txt

PYG_WORKSPACE=/work/examples/distributed/pyg
USER=root
SSH_PROT=2222
EXEC_SCRIPT="node_ogb_cpu.py"
# source /opt/intel/oneapi/setvars.sh --force; 
CMD="cd ${PYG_WORKSPACE}; python ${EXEC_SCRIPT} --ddp_backend=gloo"

# Node number:
NUM_NODES=2

# Dataset name:
DATASET=ogbn-products

# Dataset folder:
DATASET_ROOT_DIR="/work/data/partitions/${DATASET}/${NUM_NODES}-parts"

# Number of epochs:
NUM_EPOCHS=1

# The batch size:
BATCH_SIZE=1024

# Fanout per layer:
NUM_NEIGHBORS="5,5,5"

# Number of workers for sampling:
NUM_WORKERS=2
CONCURRENCY=4

# DDP Port
DDP_PORT=11111

# IP configuration path:
IP_CONFIG=${PYG_WORKSPACE}/data/ip_config.yaml

# Folder and filename to place logs:
logdir="data"
mkdir -p $logdir
logname=log_${DATASET}_${NUM_PARTS}_$(date +%H%M)
echo "stdout stored in ${PYG_WORKSPACE}/${logdir}/${logname}"
set -x

# stdout stored in `/logdir/logname.out`.
python launch.py --workspace ${PYG_WORKSPACE} --ssh_port ${SSH_PROT} --ip_config ${IP_CONFIG} --ssh_username ${USER} --num_nodes ${NUM_NODES} --num_neighbors ${NUM_NEIGHBORS} --dataset_root_dir ${DATASET_ROOT_DIR} --dataset ${DATASET}  --num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --concurrency ${CONCURRENCY} --ddp_port ${DDP_PORT} "${CMD}" |& tee ${logdir}/${logname} &
pid=$!
echo "started launch.py: ${pid}"
# kill processes at script exit (Ctrl + C)
trap "kill -2 $pid" SIGINT
wait $pid
set +x

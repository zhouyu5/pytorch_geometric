import os
import sys
import argparse


def excute_command(command, message):
    os.system(command)
    print(command)
    print('#' * 70)
    print(message)
    print('#' * 70)
    sys.stdout.flush()
    

def get_predefined_args() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(
        'GNN distributed (DDP) training benchmark')
    add = argparser.add_argument
    add('--hosts', nargs='+')
    return argparser


def main():
    dataset_list = ['Reddit', 'ogbn-products']

    for host in args.hosts:
        for dataset in dataset_list:
            command = f"""
            ssh {host} \
            'source /opt/intel/oneapi/setvars.sh --force && \
            cd /workspace/pyg-dev/benchmark/multi_gpu/training/ && \
            mpirun \
                -np 4 \
                -verbose -prepend-rank -print-rank-map \
                python training_benchmark_xpu.py \
                --dataset {dataset} --model sage \
                --batch-size 128 \
                --num-layers 2 \
                --num-hidden-channels 64 \
                --num-neighbors 10 \
                --num-epochs 10 \
                --num-workers 16 \
                --test \
                --device cpu'
            """
            message = f'np 4 on {host}, dataset {dataset}'
            excute_command(command, message)

    for host in args.hosts:
        for dataset in dataset_list:
            command = f"""
            ssh {host} \
            'source /opt/intel/oneapi/setvars.sh --force && \
            cd /workspace/pyg-dev/benchmark/multi_gpu/training/ && \
            I_MPI_OFI_PROVIDER=tcp FI_TCP_IFACE=bond0 \
            mpirun \
                -genv MASTER_ADDR={host} \
                -genv MASTER_PORT=11111 \
                -hosts {args.hosts} \
                -np 8 -ppn 4  \
                -verbose -prepend-rank -print-rank-map \
                python training_benchmark_xpu.py \
                --dataset {dataset} --model sage \
                --batch-size 64  \
                --num-layers 2 \
                --num-hidden-channels 64 \
                --num-neighbors 10 \
                --num-epochs 10 \
                --num-workers 16 \
                --test \
                --device cpu'
            """
            message = f'np 8 on {host}, dataset {dataset}'
            excute_command(command, message)


if __name__ == "__main__":
    argparser = get_predefined_args()
    args = argparser.parse_args()
    main()

# python cpu-exp.py --hosts x1001c4s0b0n0 x1001c3s2b0n0 2>&1 | tee -a cpu.log

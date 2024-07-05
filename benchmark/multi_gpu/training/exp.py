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
    add('--task', nargs='+', choices=['single', 'multi'], default=['single', 'multi'])
    add('--devices', nargs='+', choices=['cpu', 'xpu'], default=['cpu', 'xpu'])
    add('--np', default=4, type=int)
    return argparser


def main():
    dataset_list = ['Reddit', 'ogbn-products']
    batch_size = 512 // args.np

    if 'single' in args.task:
        for device in args.devices:
            for dataset in dataset_list:
                command = f"""
                mpirun \
                    -np {args.np} \
                    -verbose -prepend-rank -print-rank-map \
                    python training_benchmark_xpu.py \
                    --dataset {dataset} --model sage \
                    --batch-size {batch_size} \
                    --num-layers 2 \
                    --num-hidden-channels 64 \
                    --num-neighbors 10 \
                    --num-epochs 10 \
                    --num-workers 16 \
                    --test \
                    --device {device}
                """
                message = f'device {device} np {args.np} on dataset {dataset}'
                excute_command(command, message)

    if 'multi' in args.task:
        hosts = ','.join(args.hosts)
        ppn = args.np // len(args.hosts)

        for device in args.devices:
            for dataset in dataset_list:
                command = f"""
                I_MPI_OFI_PROVIDER=tcp FI_TCP_IFACE=bond0 \
                mpirun \
                    -genv MASTER_ADDR={args.hosts[0]} \
                    -genv MASTER_PORT=11111 \
                    -hosts {hosts} \
                    -np {args.np} -ppn {ppn}  \
                    -verbose -prepend-rank -print-rank-map \
                    python training_benchmark_xpu.py \
                    --dataset {dataset} --model sage \
                    --batch-size {batch_size}  \
                    --num-layers 2 \
                    --num-hidden-channels 64 \
                    --num-neighbors 10 \
                    --num-epochs 10 \
                    --num-workers 16 \
                    --test \
                    --device {device}
                """
                message = f'device {device} np {args.np} on {device}, dataset {dataset}'
                excute_command(command, message)


if __name__ == "__main__":
    argparser = get_predefined_args()
    args = argparser.parse_args()
    main()



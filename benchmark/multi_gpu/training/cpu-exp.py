import os
import sys



if __name__ == "__main__":
    dataset_list = ['Reddit', 'ogbn-products']

    for dataset in dataset_list:
        command = f"""
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
            --device cpu
        """
        os.system(command)
        print('#' * 70)
        print(command)
        print('#' * 70)
        sys.stdout.flush()

    for dataset in dataset_list:
        command = f"""
        I_MPI_OFI_PROVIDER=tcp FI_TCP_IFACE=bond0 \
        mpirun \
            -genv MASTER_ADDR=x1002c1s4b0n0 \
            -genv MASTER_PORT=11111 \
            -hosts x1001c7s1b0n0,x1002c1s4b0n0 \
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
            --device cpu
        """
        os.system(command)
        print('#' * 70)
        print(command)
        print('#' * 70)
        sys.stdout.flush()

# python cpu-exp.py 2>&1 | tee -a cpu.log

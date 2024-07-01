import os
import sys



if __name__ == "__main__":
    np_list = [1, 4]
    dataset_list = ['Reddit', 'ogbn-products']
    batch_size_list = [128, 512]
    # num_workers_list = [0, 2, 4, 8, 16]
    num_workers_list = [0, 16]

    for np in np_list:
        for dataset in dataset_list:
            for batch_size in batch_size_list:
                for num_workers in num_workers_list:
                    command = f"""
                        mpirun \
                        -np {np} \
                        -verbose -prepend-rank -print-rank-map \
                        python training_benchmark_xpu.py \
                        --dataset {dataset} --model sage \
                        --batch-size {batch_size}  \
                        --num-layers 2 \
                        --num-hidden-channels 64 \
                        --num-neighbors 10 \
                        --num-epochs 10 \
                        --num-workers {num_workers} \
                        --evaluate
                    """
                    os.system(command)
                    print('#' * 70)
                    print(f'np = {np}, dataset = {dataset}, batch_size = {batch_size}, num_workers = {num_workers} is done!')
                    print(command)
                    print('#' * 70)

# python compare.py 2>&1 | tee -a compare.log

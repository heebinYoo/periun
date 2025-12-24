setups=(1 2 3 4)
model_seeds=(0 1 2)
unlearn_seeds=(0 1 2)
method_idxs=(0 1 2 3 4 5)
methods=("randomlabel" "finetune" "finetune_l1" "neggrad" "GAGD" "randomlabel_salun")
epochs=(10 10 10 5 5 10)

gpu=0
max_gpu=5
for setup in "${setups[@]}"; do
for method_idx in "${method_idxs[@]}"; do
for model_seed in "${model_seeds[@]}"; do
for unlearn_seed in "${unlearn_seeds[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python -m train_unlearn \
        --setup $setup --method ${methods[$method_idx]} --epochs ${epochs[$method_idx]} \
        --save_dir  assets/optimal_basic_unlearn_model \
        --model_seed $model_seed --unlearn_seed $unlearn_seed &
    ((gpu++))
    if [ $gpu -ge $max_gpu ]; then
        wait
        gpu=0
    fi
done
done
done
done



setups=(3 4)
model_seeds=(0 1 2)
decreasing_lr=("0" "0")
epochs=(80 100)
lr=0.1
gpu=0
max_gpu=6
for setup in "${setups[@]}"; do
for model_seed in "${model_seeds[@]}"; do
    setup_index=$((setup - 3))
    CUDA_VISIBLE_DEVICES=$gpu python -m microscopic_evaluation_expr.train_orig --setup $setup --epochs ${epochs[$setup_index]} --lr $lr --decreasing_lr "${decreasing_lr[$setup_index]}" --save_dir RUM/microscopic_evaluation_expr/assets/orig_low_train_expr --model_seed $model_seed &
    ((gpu++))
    if [ $gpu -ge $max_gpu ]; then
        wait
        gpu=0
    fi
done
done




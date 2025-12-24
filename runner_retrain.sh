setups=(1 2 3 4)
model_seeds=(0 1 2)
unlearn_seeds=(0 1 2)
decreasing_lr=("0" "60,120" "0" "0")
epochs=(30 150 80 100)
lr=0.1
gpu=0
max_gpu=6
for setup in "${setups[@]}"; do
for model_seed in "${model_seeds[@]}"; do
for unlearn_seed in "${unlearn_seeds[@]}"; do
    setup_index=$((setup - 1))
    CUDA_VISIBLE_DEVICES=$gpu python -m train_retrain \
    --setup $setup --epochs ${epochs[$setup_index]} --lr $lr --decreasing_lr "${decreasing_lr[$setup_index]}" \
     --save_dir assets/retrain_model --model_seed $model_seed --unlearn_seed $unlearn_seed &
    ((gpu++))
    if [ $gpu -ge $max_gpu ]; then
        wait
        gpu=1
    fi
done
done
done
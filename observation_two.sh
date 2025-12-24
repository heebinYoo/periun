setups=(1 2 3 4)
unlearn_seeds=(0 1 2)

gpu=0
max_gpu=6
for setup in "${setups[@]}"; do
for unlearn_seed in "${unlearn_seeds[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python -m observation_two.calc \
        --setup $setup --unlearn_seed $unlearn_seed &
    ((gpu++))
    if [ $gpu -ge $max_gpu ]; then
        # wait
        gpu=0
    fi
done
done



retrain
setups=(1 2 3 4)
unlearn_seeds=(0 1 2)

gpu=0
max_gpu=6
for unlearn_seed in "${unlearn_seeds[@]}"; do
for setup in "${setups[@]}"; do
    echo CUDA_VISIBLE_DEVICES=$gpu python -m mia_eval.forget_mia --setup $setup --unlearn_seed $unlearn_seed &
    ((gpu++))
    if [ $gpu -ge $max_gpu ]; then
        # wait
        gpu=0
    fi
done
done

setups=(1 2 3 4)
unlearn_seeds=(0 1 2)
methods=("finetune" "finetune_l1" "neggrad" "GAGD" "randomlabel" "randomlabel_salun")
targets=("basic" "sailency")

gpu=0
max_gpu=6
for target in "${targets[@]}"; do
for method in "${methods[@]}"; do
for unlearn_seed in "${unlearn_seeds[@]}"; do
for setup in "${setups[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python -m mia_eval.forget_mia_unlearn --setup $setup --unlearn_seed $unlearn_seed --method $method --target $target &
    ((gpu++))
    if [ $gpu -ge $max_gpu ]; then
        # wait
        gpu=0
    fi
done
done
wait
done
done
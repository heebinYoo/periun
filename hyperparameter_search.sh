# Salun
lr_values=( $(python -c "import numpy as np; print(' '.join(map(str, np.linspace(0.01, 0.05, num=10))))") )
data_saliency=(0.1 0.3 0.5 0.7)
alpha_values=(0.3 0.4 0.5 0.6 0.7)

setups=(1 2 3 4)
model_seeds=(0)
unlearn_seeds=(0)
method_idxs=(0)
methods=("randomlabel_salun")
epochs=(10)

gpu=0
max_gpu=6
for setup in "${setups[@]}"; do
for model_seed in "${model_seeds[@]}"; do
for unlearn_seed in "${unlearn_seeds[@]}"; do
for method_idx in "${method_idxs[@]}"; do
    for lr in "${lr_values[@]}"; do
    for alpha in "${alpha_values[@]}"; do
    for ds in "${data_saliency[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python -m confidence_saliency.train_unlearn_data_saliency \
            --setup $setup --method ${methods[$method_idx]} --epochs ${epochs[$method_idx]} \
            --save_dir assets/high_conf_fgt \
            --model_seed $model_seed --unlearn_seed $unlearn_seed --lr $lr --unlearn_param $alpha --saliency_ratio $ds &
        ((gpu++))
        if [ $gpu -ge $max_gpu ]; then
            wait
            gpu=0
        fi
    done
    done
    done
done
done
done
done






# finetune L1
alpha_values=( $(python -c "import numpy as np; print(' '.join(map(str, np.logspace(-6, -4, num=10))))") )
lr_values=( $(python -c "import numpy as np; print(' '.join(map(str, np.logspace(-5, -1, num=10))))") )
data_saliency=(0.1 0.3 0.5 0.7)

setups=(1 2 3 4)
model_seeds=(0)
unlearn_seeds=(0)
method_idxs=(0)
methods=("finetune_l1")
epochs=(10)

gpu=0
max_gpu=6
for setup in "${setups[@]}"; do
for model_seed in "${model_seeds[@]}"; do
for unlearn_seed in "${unlearn_seeds[@]}"; do
for method_idx in "${method_idxs[@]}"; do
    for ds in "${data_saliency[@]}"; do
    for lr in "${lr_values[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python -m confidence_saliency.train_unlearn_data_saliency \
            --setup $setup --method ${methods[$method_idx]} --epochs ${epochs[$method_idx]} \
            --save_dir assets/unlearn_hyperparam_search \
            --model_seed $model_seed --unlearn_seed $unlearn_seed --lr $lr --unlearn_param $alpha --saliency_ratio $ds &
        ((gpu++))
        if [ $gpu -ge $max_gpu ]; then
            wait
            gpu=0
        fi
    done
    done
    done
done
done
done
done


# GAGD
alpha_values=( $(python -c "import numpy as np; print(' '.join(map(str, np.linspace(0.85, 0.99, num=10))))") )
lr_values=( $(python -c "import numpy as np; print(' '.join(map(str, np.logspace(-5, -1, num=10))))") )
data_saliency=(0.1 0.3 0.5 0.7)

setups=(1 2 3 4)
model_seeds=(0)
unlearn_seeds=(0)
method_idxs=(0)
methods=("GAGD")
epochs=(5)

gpu=0
max_gpu=6
for setup in "${setups[@]}"; do
for model_seed in "${model_seeds[@]}"; do
for unlearn_seed in "${unlearn_seeds[@]}"; do
for method_idx in "${method_idxs[@]}"; do
    for ds in "${data_saliency[@]}"; do
    for lr in "${lr_values[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python -m confidence_saliency.train_unlearn_data_saliency \
            --setup $setup --method ${methods[$method_idx]} --epochs ${epochs[$method_idx]} \
            --save_dir assets/unlearn_hyperparam_search \
            --model_seed $model_seed --unlearn_seed $unlearn_seed --lr $lr --unlearn_param $alpha --saliency_ratio $ds &
        ((gpu++))
        if [ $gpu -ge $max_gpu ]; then
            wait
            gpu=0
        fi
    done
    done
    done
done
done
done
done



# finetune
lr_values=( $(python -c "import numpy as np; print(' '.join(map(str, np.logspace(-5, -1, num=10))))") )
data_saliency=(0.1 0.3 0.5 0.7)

setups=(1 2 3 4)
model_seeds=(0)
unlearn_seeds=(0)
method_idxs=(0)
methods=("finetune")
epochs=(10)

gpu=0
max_gpu=6
for setup in "${setups[@]}"; do
for model_seed in "${model_seeds[@]}"; do
for unlearn_seed in "${unlearn_seeds[@]}"; do
for method_idx in "${method_idxs[@]}"; do
    for ds in "${data_saliency[@]}"; do
    for lr in "${lr_values[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python -m confidence_saliency.train_unlearn_data_saliency \
            --setup $setup --method ${methods[$method_idx]} --epochs ${epochs[$method_idx]} \
            --save_dir assets/unlearn_hyperparam_search \
            --model_seed $model_seed --unlearn_seed $unlearn_seed --lr $lr --saliency_ratio $ds &
        ((gpu++))
        if [ $gpu -ge $max_gpu ]; then
            wait
            gpu=0
        fi
    done
    done
done
done
done
done

#Neggrad

lr_values=( $(python -c "import numpy as np; print(' '.join(map(str, np.logspace(-5, -1, num=10))))") )
data_saliency=(0.1 0.3 0.5 0.7)
setups=(1 2 3 4)
model_seeds=(0)
unlearn_seeds=(0)
method_idxs=(0)
methods=("neggrad")
epochs=(5)

gpu=0
max_gpu=6
for setup in "${setups[@]}"; do
for model_seed in "${model_seeds[@]}"; do
for unlearn_seed in "${unlearn_seeds[@]}"; do
for method_idx in "${method_idxs[@]}"; do
    for ds in "${data_saliency[@]}"; do
    for lr in "${lr_values[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python -m confidence_saliency.train_unlearn_data_saliency \
            --setup $setup --method ${methods[$method_idx]} --epochs ${epochs[$method_idx]} \
            --save_dir assets/unlearn_hyperparam_search \
            --model_seed $model_seed --unlearn_seed $unlearn_seed --lr $lr --saliency_ratio $ds &
        ((gpu++))
        if [ $gpu -ge $max_gpu ]; then
            wait
            gpu=0
        fi
    done
    done
done
done
done
done






# Random Label

lr_values=( $(python -c "import numpy as np; print(' '.join(map(str, np.linspace(0.01, 0.1, num=10))))") )
data_saliency=(0.1 0.3 0.5 0.7)
setups=(1 2 3 4)
model_seeds=(0)
unlearn_seeds=(0)
method_idxs=(0)
methods=("randomlabel")
epochs=(10)

gpu=0
max_gpu=6
for setup in "${setups[@]}"; do
for model_seed in "${model_seeds[@]}"; do
for unlearn_seed in "${unlearn_seeds[@]}"; do
for method_idx in "${method_idxs[@]}"; do
    for lr in "${lr_values[@]}"; do
    for ds in "${data_saliency[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python -m confidence_saliency.train_unlearn_data_saliency \
            --setup $setup --method ${methods[$method_idx]} --epochs ${epochs[$method_idx]} \
            --save_dir assets/unlearn_hyperparam_search \
            --model_seed $model_seed --unlearn_seed $unlearn_seed --lr $lr --saliency_ratio $ds &
        ((gpu++))
        if [ $gpu -ge $max_gpu ]; then
            wait
            gpu=0
        fi
    done
    done
done
done
done
done
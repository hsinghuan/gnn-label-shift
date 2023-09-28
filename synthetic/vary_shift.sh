#!/bin/bash

seed_list=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")
shift_list=("-0.2" "-0.3" "-0.4" "-0.5" "-0.6")
for seed in "${seed_list[@]}"; do
    for shift in "${shift_list[@]}"; do
        python wo_adapt_exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_shift \
                           --exp_param "$shift"\
                           --gnn_dim_list 2 64 64 \
                           --mlp_dim_list 64 16 2 \
                           --gnn_dr_list 0.5 \
                           --mlp_dr_list 0.5 \
                           --learning_rate 5e-3 \
                           --seed "$seed"
    done
done
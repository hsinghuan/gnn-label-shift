#!/bin/bash

seed_list=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")
inter_edge_prob_list=("0.003" "0.006" "0.009" "0.012" "0.015")
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python wo_adapt_exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob"\
                           --gnn_dim_list 2 64 64 \
                           --mlp_dim_list 64 16 2 \
                           --gnn_dr_list 0.5 \
                           --mlp_dr_list 0.5 \
                           --seed "$seed"
    done
done
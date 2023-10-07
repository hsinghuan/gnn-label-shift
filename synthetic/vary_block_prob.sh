#!/bin/bash

seed_list=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")
inter_edge_prob_list=("0.003" "0.0045" "0.006" "0.0075" "0.009" "0.0105" "0.012" "0.0135" "0.015" "0.0165" "0.018" "0.0195")

# GCN
## 1 layer
#for seed in "${seed_list[@]}"; do
#    for prob in "${inter_edge_prob_list[@]}"; do
#        python wo_adapt_exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_block_prob \
#                           --exp_param "$prob"\
#                           --model gcn \
#                           --gnn_dim_list 2 64 \
#                           --mlp_dim_list 64 64 2 \
#                           --mlp_dr_list 0.5 \
#                           --learning_rate 5e-3 \
#                           --seed "$seed"
#    done
#done


#Logistic Regression
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python wo_adapt_exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob"\
                           --model logreg \
                           --seed "$seed"
    done
done
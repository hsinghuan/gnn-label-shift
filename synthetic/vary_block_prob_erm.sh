#!/bin/bash

seed_list=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")
inter_edge_prob_list=("0.003" "0.0045" "0.006" "0.0075" "0.009" "0.0105" "0.012" "0.0135" "0.015" "0.0165" "0.018" "0.0195")

# GCN
## 1 layer
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob" \
                           --method erm \
                           --model gcn \
                           --model_gnn_dim_list 2 64 \
                           --model_mlp_dim_list 64 64 2 \
                           --model_mlp_dr_list 0.5 \
                           --model_learning_rate 5e-3 \
                           --seed "$seed"
    done
done

## 2 layer
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob" \
                           --method erm \
                           --model gcn \
                           --model_gnn_dim_list 2 64 64 \
                           --model_mlp_dim_list 64 64 2 \
                           --model_gnn_dr_list 0.5 \
                           --model_mlp_dr_list 0.5 \
                           --model_learning_rate 5e-3 \
                           --seed "$seed"
    done
done

## 3 layer
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob"\
                           --method erm \
                           --model gcn \
                           --model_gnn_dim_list 2 32 32 64 \
                           --model_mlp_dim_list 64 64 2 \
                           --model_gnn_dr_list 0.5 0.5 \
                           --model_mlp_dr_list 0.5 \
                           --model_learning_rate 5e-3 \
                           --seed "$seed"
    done
done

# LinGCN
## 1 layer
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob" \
                           --method erm \
                           --model lingcn \
                           --model_gnn_dim_list 2 2 \
                           --model_learning_rate 3e-2 \
                           --seed "$seed"
    done
done

## 2 layer
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob"\
                           --method erm \
                           --model lingcn \
                           --model_gnn_dim_list 2 2 2 \
                           --model_learning_rate 3e-2 \
                           --seed "$seed"
    done
done

## 3 layer
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob" \
                           --method erm \
                           --model lingcn \
                           --model_gnn_dim_list 2 2 2 2 \
                           --model_learning_rate 3e-2 \
                           --seed "$seed"
    done
done

# GAT
## 1 layer
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob" \
                           --method erm \
                           --model gat \
                           --model_gnn_dim_list 2 64 \
                           --model_mlp_dim_list 64 64 2 \
                           --model_mlp_dr_list 0.5 \
                           --model_learning_rate 5e-3 \
                           --seed "$seed"
    done
done

## 2 layer
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob" \
                           --method erm \
                           --model gat \
                           --model_gnn_dim_list 2 64 64 \
                           --model_mlp_dim_list 64 64 2 \
                           --model_gnn_dr_list 0.5 \
                           --model_mlp_dr_list 0.5 \
                           --model_learning_rate 5e-3 \
                           --seed "$seed"
    done
done

## 3 layer
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob" \
                           --method erm \
                           --model gat \
                           --model_gnn_dim_list 2 32 32 64 \
                           --model_mlp_dim_list 64 64 2 \
                           --model_gnn_dr_list 0.5 0.5 \
                           --model_mlp_dr_list 0.5 \
                           --model_learning_rate 1e-2 \
                           --seed "$seed"
    done
done



#Logistic Regression
for seed in "${seed_list[@]}"; do
    for prob in "${inter_edge_prob_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_block_prob \
                           --exp_param "$prob"\
                           --method erm \
                           --model logreg \
                           --seed "$seed"
    done
done
#!/bin/bash

seed_list=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")

# GCN
## 2 layer, mlp as blackbox
for seed in "${seed_list[@]}"; do
      python main.py --data_dir ~/data/ogb-data/ \
                         --method iw-erm \
                         --estimator bbse \
                         --blackbox gcn \
                         --model gcn \
                         --blackbox_gnn_dim_list 128 64 64 \
                         --blackbox_mlp_dim_list 64 64 40 \
                         --blackbox_gnn_dr_list 0.5 \
                         --blackbox_mlp_dr_list 0.5 \
                         --blackbox_learning_rate 1e-2 \
                         --blackbox_epochs 400 \
                         --model_gnn_dim_list 128 64 64 \
                         --model_mlp_dim_list 64 64 40 \
                         --model_gnn_dr_list 0.5 \
                         --model_mlp_dr_list 0.5 \
                         --model_learning_rate 1e-2 \
                         --model_epochs 400 \
                         --seed "$seed"
done

## 2 layer, mlp as blackbox
#for seed in "${seed_list[@]}"; do
#      python main.py --data_dir ~/data/ogb-data/ \
#                         --method iw-erm \
#                         --estimator bbse \
#                         --blackbox mlp \
#                         --model gcn \
#                         --blackbox_mlp_dim_list 128 128 40 \
#                         --blackbox_mlp_dr_list 0.5 \
#                         --blackbox_learning_rate 5e-3 \
#                         --blackbox_epochs 400 \
#                         --model_gnn_dim_list 128 64 64 \
#                         --model_mlp_dim_list 64 64 40 \
#                         --model_gnn_dr_list 0.5 \
#                         --model_mlp_dr_list 0.5 \
#                         --model_learning_rate 1e-2 \
#                         --model_epochs 400 \
#                         --seed "$seed"
#done


#Multi-layer Perceptron
#for seed in "${seed_list[@]}"; do
#      python main.py --data_dir ~/data/ogb-data/ \
#                         --method iw-erm \
#                         --estimator bbse \
#                         --blackbox mlp \
#                         --model mlp \
#                         --blackbox_mlp_dim_list 128 128 40 \
#                         --blackbox_mlp_dr_list 0.5 \
#                         --blackbox_learning_rate 5e-3 \
#                         --blackbox_epochs 400 \
#                         --model_mlp_dim_list 128 128 40 \
#                         --model_mlp_dr_list 0.5 \
#                         --model_learning_rate 5e-3 \
#                         --model_epochs 400 \
#                         --seed "$seed"
#done

#Logistic Regression
#for seed in "${seed_list[@]}"; do
#      python main.py --data_dir ~/data/ogb-data/ \
#                         --method iw-erm \
#                         --estimator bbse \
#                         --blackbox logreg \
#                         --model logreg \
#                         --seed "$seed"
#done
#!/bin/bash

seed_list=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")



# GCN
## 2 layer
for seed in "${seed_list[@]}"; do
      python main.py --data_dir ~/data/ogb-data/ \
                         --method erm \
                         --model gcn \
                         --model_gnn_dim_list 128 64 64 \
                         --model_mlp_dim_list 64 64 40 \
                         --model_gnn_dr_list 0.5 \
                         --model_mlp_dr_list 0.5 \
                         --model_learning_rate 1e-2 \
                         --model_epochs 400 \
                         --seed "$seed"
done


#Multi-layer Perceptron
#for seed in "${seed_list[@]}"; do
#      python main.py --data_dir ~/data/ogb-data/ \
#                         --method erm \
#                         --model mlp \
#                         --model_mlp_dim_list 128 128 40 \
#                         --model_mlp_dr_list 0.5 \
#                         --model_learning_rate 5e-3 \
#                         --model_epochs 400 \
#                         --seed "$seed"
#done

#Logistic Regression
#for seed in "${seed_list[@]}"; do
#      python main.py --data_dir ~/data/ogb-data/ \
#                         --method erm \
#                         --model logreg \
#                         --seed "$seed"
#done
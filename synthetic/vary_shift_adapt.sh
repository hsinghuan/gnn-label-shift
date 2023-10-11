#!/bin/bash

seed_list=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")
shift_list=("-0.0" "-0.1" "-0.2" "-0.3" "-0.4" "-0.5" "-0.6")


# GCN
## 1 layer
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift"\
#                           --estimator "bbse"\
#                           --model gcn \
#                           --gnn_dim_list 2 64 \
#                           --mlp_dim_list 64 64 2 \
#                           --mlp_dr_list 0.5 \
#                           --learning_rate 5e-3 \
#                           --seed "$seed"
#    done
#done


# LinGCN
## 1 layer
for seed in "${seed_list[@]}"; do
    for shift in "${shift_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_shift \
                           --exp_param "$shift"\
                           --estimator "bbse"\
                           --model lingcn \
                           --gnn_dim_list 2 2 \
                           --learning_rate 3e-2 \
                           --seed "$seed"
    done
done

## 1 layer, logreg as black box
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift"\
#                           --estimator "bbse-logreg"\
#                           --model lingcn \
#                           --gnn_dim_list 2 2 \
#                           --learning_rate 3e-2 \
#                           --seed "$seed"
#    done
#done



#Multi-layer Perceptron
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift"\
#                           --estimator "bbse"\
#                           --model mlp \
#                           --mlp_dim_list 2 64 64 2 \
#                           --mlp_dr_list 0.5 0.5 \
#                           --seed "$seed"
#    done
#done


#Logistic Regression
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift"\
#                           --estimator "bbse"\
#                           --model logreg \
#                           --seed "$seed"
#    done
#done
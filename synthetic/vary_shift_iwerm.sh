#!/bin/bash

seed_list=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")
shift_list=("-0.0" "-0.1" "-0.2" "-0.3" "-0.4" "-0.5" "-0.6")


# GCN
## 1 layer
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift" \
#                           --method iw-erm \
#                           --estimator bbse \
#                           --blackbox gcn \
#                           --model gcn \
#                           --blackbox_gnn_dim_list 2 64 \
#                           --blackbox_mlp_dim_list 64 64 2 \
#                           --blackbox_mlp_dr_list 0.5 \
#                           --blackbox_learning_rate 5e-3 \
#                           --model_gnn_dim_list 2 64 \
#                           --model_mlp_dim_list 64 64 2 \
#                           --model_mlp_dr_list 0.5 \
#                           --model_learning_rate 5e-3 \
#                           --seed "$seed"
#    done
#done

## 1 layer, logreg as blackbox
for seed in "${seed_list[@]}"; do
    for shift in "${shift_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_shift \
                           --exp_param "$shift" \
                           --method iw-erm \
                           --estimator bbse \
                           --blackbox logreg \
                           --model gcn \
                           --model_gnn_dim_list 2 64 \
                           --model_mlp_dim_list 64 64 2 \
                           --model_mlp_dr_list 0.5 \
                           --model_learning_rate 5e-3 \
                           --seed "$seed"
    done
done


# LinGCN
## 1 layer
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift" \
#                           --method iw-erm \
#                           --estimator bbse \
#                           --blackbox lingcn \
#                           --model lingcn \
#                           --blackbox_gnn_dim_list 2 2 \
#                           --blackbox_learning_rate 3e-2 \
#                           --model_gnn_dim_list 2 2 \
#                           --model_learning_rate 3e-2 \
#                           --seed "$seed"
#    done
#done

## 2 layer
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift" \
#                           --method iw-erm \
#                           --estimator bbse \
#                           --blackbox lingcn \
#                           --model lingcn \
#                           --blackbox_gnn_dim_list 2 2 2 \
#                           --blackbox_learning_rate 3e-2 \
#                           --model_gnn_dim_list 2 2 2 \
#                           --model_learning_rate 3e-2 \
#                           --seed "$seed"
#    done
#done

## 3 layer
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift" \
#                           --method iw-erm \
#                           --estimator bbse \
#                           --blackbox lingcn \
#                           --model lingcn \
#                           --blackbox_gnn_dim_list 2 2 2 2 \
#                           --blackbox_learning_rate 3e-2 \
#                           --model_gnn_dim_list 2 2 2 2 \
#                           --model_learning_rate 3e-2 \
#                           --seed "$seed"
#    done
#done


## 1 layer, logreg as black box
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift"\
#                           --method iw-erm \
#                           --estimator bbse \
#                           --blackbox logreg \
#                           --model lingcn \
#                           --model_gnn_dim_list 2 2 \
#                           --model_learning_rate 3e-2 \
#                           --seed "$seed"
#    done
#done

## 2 layer, logreg as black box
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift" \
#                           --method iw-erm \
#                           --estimator bbse \
#                           --blackbox logreg \
#                           --model lingcn \
#                           --model_gnn_dim_list 2 2 2 \
#                           --model_learning_rate 3e-2 \
#                           --seed "$seed"
#    done
#done

## 3 layer, logreg as black box
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift" \
#                           --method iw-erm \
#                           --estimator bbse \
#                           --blackbox logreg \
#                           --model lingcn \
#                           --model_gnn_dim_list 2 2 2 2 \
#                           --model_learning_rate 3e-2 \
#                           --seed "$seed"
#    done
#done

# GAT
## 1 layer
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift" \
#                           --method iw-erm \
#                           --estimator bbse \
#                           --blackbox gat \
#                           --model gat \
#                           --blackbox_gnn_dim_list 2 64 \
#                           --blackbox_mlp_dim_list 64 64 2 \
#                           --blackbox_mlp_dr_list 0.5 \
#                           --blackbox_learning_rate 5e-3 \
#                           --model_gnn_dim_list 2 64 \
#                           --model_mlp_dim_list 64 64 2 \
#                           --model_mlp_dr_list 0.5 \
#                           --model_learning_rate 5e-3 \
#                           --seed "$seed"
#    done
#done

## 1 layer, logreg as blackbox
for seed in "${seed_list[@]}"; do
    for shift in "${shift_list[@]}"; do
        python exp.py --data_dir ~/data/sbm_ls/ \
                           --exp_name vary_shift \
                           --exp_param "$shift" \
                           --method iw-erm \
                           --estimator bbse \
                           --blackbox logreg \
                           --model gat \
                           --model_gnn_dim_list 2 64 \
                           --model_mlp_dim_list 64 64 2 \
                           --model_mlp_dr_list 0.5 \
                           --model_learning_rate 5e-3 \
                           --seed "$seed"
    done
done

#Multi-layer Perceptron
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift"\
#                           --method iw-erm \
#                           --estimator bbse \
#                           --blackbox mlp \
#                           --model mlp \
#                           --blackbox_mlp_dim_list 2 64 64 2 \
#                           --blackbox_mlp_dr_list 0.5 0.5 \
#                           --model_mlp_dim_list 2 64 64 2 \
#                           --model_mlp_dr_list 0.5 0.5 \
#                           --seed "$seed"
#    done
#done


#Logistic Regression
#for seed in "${seed_list[@]}"; do
#    for shift in "${shift_list[@]}"; do
#        python exp.py --data_dir ~/data/sbm_ls/ \
#                           --exp_name vary_shift \
#                           --exp_param "$shift" \
#                           --method iw-erm \
#                           --estimator bbse \
#                           --blackbox logreg \
#                           --model logreg \
#                           --seed "$seed"
#    done
#done
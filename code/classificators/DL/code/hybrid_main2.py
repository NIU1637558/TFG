# 1. CONFIG
from functions.config import parse_arguments, initialize_wandb, setup_preprocessing

# 2. LOAD DATA
from functions.load_data import load_and_balance_data

# 3. PREPROCESS DATA
from functions.preprocess_data import prepare_features

# 4. TRAIN/EVALUATION
from functions.training import single_training_loop, evaluate_and_save_best_model_outputs, save_single_metrics, hybrid_training_loop_multiinput, evaluate_and_save_best_model_outputs_multiinput, joint_training_loop_multiinput

# 5. MODELS
from model_arquitectres.models import MLP2, HybridAttentionMLP

# others
import pandas as pd 
import numpy as np
import sys
import wandb
import os
import torch

# main.py
def model_factory_fn(input_dim, cat_dim):
    return HybridAttentionMLP(input_dim, cat_dim, dropout=0.5)


def main():
    ## ------------- 1. Initialize WandB --------------------##
    sys.stderr.write(" -------------------- 1. Initialize WandB ---------------------------\n")
    name = 'Hybrid2_HUMAN'
    args = parse_arguments()
    nepoch = 200
    n_models = 2 # num models per type, max 8
    extra = 0
    initialize_wandb(args, name, nepoch, extra)

    ## ------------- 2. Load and preprocess data --------------------##
    sys.stderr.write(" -------------------- 2. Load and preprocess data --------------------\n")
    if 'RPA' in name:
        data_type = 'RPA'
    else:
        data_type = 'human'
    original_data, balanced_data, target_col = load_and_balance_data(extra, data_type)
    setup_preprocessing()

    # # # sample balanced data
    # balanced_data = balanced_data.sample(n=1000, random_state=42)
    
    class_names = ['False', 'True']

    # get embeddings for each model
    macroX = {}
    macro_txtshape = {}
    embedders = ['w2v','d2v','BERT', 'AE1BERT', 'distBERT', 'AE1BERTtuned']
    for embedder in embedders:
        print(f'Using {embedder} embedding...')
        macroX[embedder], y, categorical_cols, text_cols, og_indices, macro_txtshape[embedder] = prepare_features(balanced_data, original_data, embedder=embedder)

    sys.stderr.write('---------------------------------------------------------------------\n')

    ## ------------- 4. Training and Testing--------------------##
    sys.stderr.write(" -------------------- 3. Training  ----------------------------------\n")
    if 'att' in name:
        att = True
        print('Attention Activated...')
    # No cargues modelos preentrenados, crea desde cero
    model, test_loader, y_test, metrics = joint_training_loop_multiinput(
        macroX, y,
        model_factory_fn=model_factory_fn,
        macro_txtshape=macro_txtshape,
        nepoch=nepoch
    )

    sys.stderr.write('---------------------------------------------------------------------\n')
    
    ## ------------- 5. Save metrics and predictions --------------------##
    sys.stderr.write(" -------------------- 4. Save metrics and predictions --------------------\n")
    model_path = evaluate_and_save_best_model_outputs_multiinput(
        model, test_loader, y_test,
        balanced_data, original_data,
        class_names, target_col, name_model=name
    )


    csv_path = save_single_metrics(metrics, class_names, name)

    # save code
    code_path = '/fhome/amir/TFG/code/classificators/DL/code/hibrid_main2.py'
    wandb.save(code_path)

    sys.stderr.write('----------------------------------------------------------------------\n')
    
    ## ------------- 6. Finalize WandB --------------------##
    sys.stderr.write(" -------------------- 5. Finalize WandB --------------------\n")
    wandb.save(os.path.join(os.path.dirname(__file__), 'hibrid_main2.py'))
    wandb.finish()
    sys.stderr.write(f"\nBest model saved to {model_path}\n")
    sys.stderr.write(f"Metrics saved to {csv_path}\n")
    if args.save_fails:
        sys.stderr.write(f"Failed predictions saved to failed_predictions/ directory\n")
    sys.stderr.write("All done!\n")
    sys.stderr.write('----------------------------------------------------------------------\n')

if __name__ == "__main__":
    main()
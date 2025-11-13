#!/usr/bin/env python3
"""
SLM (Small Language Model) Training Pipeline
Uses DistilBERT for Dense Passage Retrieval on MSMARCO dataset
"""

import os
import sys
import logging
import pandas as pd
from datasets import load_dataset
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

# ============================================================================
# SETUP
# ============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Create necessary directories
os.makedirs("content/data/msmarco", exist_ok=True)
os.makedirs("drive/MyDrive/trained_models/pretrained", exist_ok=True)
os.makedirs("drive/MyDrive/results", exist_ok=True)
os.makedirs("drive/MyDrive/indices", exist_ok=True)

# ============================================================================
# STEP 1: DOWNLOAD AND PREPARE MSMARCO DATA
# ============================================================================

def download_msmarco_data():
    """Download MSMARCO training triples and validation qrels"""
    print("=== Downloading MSMARCO ===")
    print("Downloading MSMARCO training triples...")
    
    dataset = load_dataset("thilina/negative-sampling")["train"]
    qrels = load_dataset("BeIR/msmarco-qrels")["validation"]
    
    print("Dataset loaded. Sample:")
    print(dataset[0])
    
    print("Saving dataset to disk...")
    dataset.to_csv("content/data/msmarco/msmarco-train.tsv", sep="\t", index=False)
    qrels.to_csv("content/data/msmarco/qrels.tsv", sep="\t", index=False)
    
    print("Done.")
    print("=== MSMARCO download complete ===")

# ============================================================================
# STEP 2: TRAIN THE MODEL
# ============================================================================

def train_retrieval_model():
    """Train DPR model using DistilBERT on MSMARCO data"""
    
    train_data_path = "content/data/msmarco/msmarco-train.tsv"
    
    # Load training data
    print("\nLoading training data...")
    train_data = pd.read_csv(train_data_path, sep="\t")
    print(f"Training data shape: {train_data.shape}")
    print(f"Columns: {train_data.columns.tolist()}")
    
    # Configure training arguments
    model_args = RetrievalArgs()
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.use_cached_eval_features = False
    model_args.include_title = False
    model_args.max_seq_length = 256
    model_args.num_train_epochs = 5
    model_args.train_batch_size = 16
    model_args.use_hf_datasets = True
    model_args.learning_rate = 1e-6
    model_args.warmup_steps = 5000
    model_args.save_steps = 300000
    model_args.evaluate_during_training = False
    model_args.save_model_every_epoch = False
    model_args.wandb_project = "Negative Sampling Multilingual - Pretrain"
    model_args.hard_negatives = False
    model_args.n_gpu = 1
    model_args.data_format = "beir"
    model_args.output_dir = "drive/MyDrive/trained_models/pretrained/DPR-distilbert-msmarco"
    model_args.wandb_kwargs = {"name": "DPR-distilbert-msmarco"}
    
    print("\n=== Starting Model Training ===")
    print(f"Model Args: {model_args}")
    
    # Initialize retrieval model with DistilBERT
    model = RetrievalModel(
        model_type="custom",
        model_name=None,
        context_encoder_name="distilbert-base-multilingual-cased",
        query_encoder_name="distilbert-base-multilingual-cased",
        args=model_args,
    )
    
    # Train model
    print("Training started...")
    model.train_model(train_data=train_data, eval_set="dev")
    
    print("=== Training Complete ===")
    return model

# ============================================================================
# STEP 3: EVALUATE THE MODEL
# ============================================================================

def evaluate_retrieval_model():
    """Evaluate trained model on test set"""
    
    dataset = "english"
    data_path = "content/data/msmarco"
    model_path = "drive/MyDrive/trained_models/pretrained/DPR-distilbert-msmarco"
    results_dir = "drive/MyDrive/results/msmarco"
    indices_dir = "drive/MyDrive/indices/msmarco"
    
    # Retrieval model arguments
    args = RetrievalArgs()
    args.reprocess_input_data = True
    args.overwrite_output_dir = True
    args.retrieve_n_docs = 100
    args.max_seq_length = 256
    args.eval_batch_size = 100
    args.n_gpu = 1
    args.data_format = "beir"
    args.output_dir = indices_dir
    
    print("\n=== Starting Model Evaluation ===")
    
    # Initialize model from checkpoint
    model = RetrievalModel(
        model_type="custom",
        model_name=None,
        context_encoder_name=f"{model_path}/context_encoder",
        query_encoder_name=f"{model_path}/query_encoder",
        args=args,
    )
    
    # Evaluate
    print(f"Evaluating on {dataset} dataset...")
    report = model.eval_model(
        data_path,
        save_as_experiment=True,
        experiment_name=results_dir,
        dataset_name=dataset,
        model_name=os.path.basename(model_path),
        eval_set="test",
        pytrec_eval_metrics=["recip_rank", "recall_100", "ndcg_cut_10"],
    )
    
    # Print results
    print("\n#############################")
    print(f"Dataset: {dataset}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Results:\n{report}")
    print("#############################")
    
    return report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Step 1: Download data
    download_msmarco_data()
    
    # Step 2: Train model
    train_retrieval_model()
    
    # Step 3: Evaluate model
    evaluate_retrieval_model()
    
    print("\n=== Pipeline Complete ===")
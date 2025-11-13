import logging
import pandas as pd
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
from multiprocessing import set_start_method

# Set logging level
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Path to your local training data
train_data_path = "../../data/msmarco/msmarco-train.tsv"

# Load the training data
train_data = pd.read_csv(train_data_path, sep="\t")

# Optional: If you have a dev set, load it similarly, else set None
eval_data = None

# Configure the model arguments
model_args = RetrievalArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_cached_eval_features = False
model_args.include_title = False
model_args.max_seq_length = 256
model_args.num_train_epochs = 3  # Change epochs as needed
model_args.train_batch_size = 8
model_args.use_hf_datasets = True
model_args.learning_rate = 1e-6
model_args.warmup_steps = 500
model_args.save_steps = 1000
model_args.evaluate_during_training = False
model_args.save_model_every_epoch = False
model_args.hard_negatives = True
model_args.n_gpu = 1

model_args.data_format = "beir"
model_args.ance_training = False

# Model info: use multilingual BERT for context and question encoder
model_type = "custom"
model_name = None
context_name = "bert-base-multilingual-cased"
question_name = "bert-base-multilingual-cased"

# Output directory for the trained model
model_args.output_dir = "../../trained_models/pretrained/DPR-BM-msmarco-local"

import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

if __name__ == "__main__":
    # Set multiprocessing start method
    set_start_method("spawn")

    # Initialize the model
    model = RetrievalModel(
        model_type=model_type,
        model_name=model_name,
        context_encoder_name=context_name,
        question_encoder_name=question_name,
        args=model_args,
    )

    # Train the model
    model.train_model(
        train_data,
        eval_set=eval_data,
    )

import torch

use_cuda = torch.cuda.is_available()
print(f"CUDA available? {use_cuda}")

model = RetrievalModel(
    model_type=model_type,
    model_name=model_name,
    context_encoder_name=context_name,
    question_encoder_name=question_name,
    args=model_args,
    use_cuda=use_cuda,
)

import multiprocessing

if __name__ == "__main__":
    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method != "spawn":
        multiprocessing.set_start_method("spawn")

    # Continue with your code

import logging
import pandas as pd
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

# Setting up the logging configuration
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Path to the training data
train_data_path = r"C:\Users\CSE IIT BHILAI\Maneesh\EffiSearch\Download_data\data\msmarco\msmarco-train.tsv"

# Reading the training data from a TSV file or using the provided path
if train_data_path.endswith(".tsv"):
    train_data = pd.read_csv(train_data_path, sep="\t")
else:
    train_data = train_data_path

# Setting up the model arguments
model_args = RetrievalArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_cached_eval_features = False
model_args.include_title = False if "msmarco" in train_data_path else True
model_args.max_seq_length = 256
model_args.num_train_epochs = 5
model_args.train_batch_size = 2
model_args.use_hf_datasets = True
model_args.learning_rate = 1e-6
model_args.warmup_steps = 500
model_args.save_steps = 300000
model_args.evaluate_during_training = False
model_args.save_model_every_epoch = False

# Setting up the project name for Weights & Biases integration. Remove this line if you don't use W&B.
model_args.wandb_project = "Negative Sampling Multilingual - Pretrain"

# Enabling hard negatives for training
model_args.hard_negatives = True

# Setting up the number of GPUs to use and the data format
model_args.n_gpu = 1
model_args.data_format = "beir"

# Disabling ANCE training
model_args.ance_training = False

# Setting up the model type, model name, context name, and question name
model_type = "custom"
model_name = None
context_name = "bert-base-multilingual-cased"
question_name = "bert-base-multilingual-cased"

# Setting up the Weights & Biases run name
model_args.wandb_kwargs = {"name": r"C:\Users\CSE IIT BHILAI\Maneesh\EffiSearch\DPR-BM-msmarco"}

# Setting up the output directory for saving the trained model
model_args.output_dir = r"C:\Users\CSE IIT BHILAI\Maneesh\EffiSearch\DPR_BM_MSmarco_with_-ves"

import wandb


if __name__ == "__main__":
    from multiprocess import get_start_method, set_start_method

    if get_start_method(allow_none=True) != "spawn":
        set_start_method("spawn")

    # Initialize W&B run
    wandb.init(
        project=model_args.wandb_project,
        name=model_args.wandb_kwargs.get("name"),
        config=model_args.__dict__,  # logs all your training args in W&B config
    )
    model_args.use_cuda = False
    model = RetrievalModel(
        model_type,
        model_name,
        context_name,
        question_name,
        args=model_args,
    )




    model.train_model(
        train_data,
        eval_set="dev",
    )
import os
import json
import logging
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
from transformers import AutoTokenizer

import multiprocessing


def setup_ddp():
    if 'LOCAL_RANK' not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), -1, 1
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    return device, local_rank, world_size


@dataclass
class DPRMrTyDiConfig:
    data_path: str = "./data/mrtydi/train.tsv"
    output_dir: str = "./dpr_mrtydi_negative"
    language: str = "all"
    query_max_length: int = 64
    passage_max_length: int = 256
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-6
    warmup_steps: int = 500
    save_steps: int = 1000
    log_steps: int = 100
    max_samples: Optional[int] = None
    num_hard_negatives: int = 1


class MrTyDiDataset:
    def __init__(self, file_path: str, language: str = "all", max_samples: Optional[int] = None, num_hard_negatives: int = 1):
        self.language = language
        self.num_hard_negatives = num_hard_negatives
        self.data = self._load_data(file_path, max_samples)

    def _load_data(self, file_path: str, max_samples: Optional[int]) -> List[Dict]:
        raw_data = []
        df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip")
        first_row = df.iloc[0].tolist()
        is_header = any(isinstance(v, str) and v.lower() in
                        ['query_id', 'query', 'passage', 'positive_passages',
                         'negative_passages', 'language', 'doc_id']
                        for v in first_row)
        start_idx = 1 if is_header else 0
        for idx, row in df.iterrows():
            if idx < start_idx:
                continue
            row = [str(x).strip() for x in row]
            if len(row) < 2:
                continue
            if len(row) >= 5:
                qid, lang, query, doc_id, passage = row[:5]
            elif len(row) >= 4:
                qid, query, passage, _ = row[:4]
                lang = self.language
                doc_id = f"D{idx}"
            elif len(row) == 3:
                qid, query, passage = row
                lang = self.language
                doc_id = f"D{idx}"
            else:
                qid = f"Q{idx}"
                query = row[0]
                passage = row[1] if len(row) > 1 else row[0]
                lang = self.language
                doc_id = f"D{idx}"
            if self.language != "all" and lang != self.language:
                continue
            if len(query.split()) < 1 or len(passage.split()) < 2:
                continue
            raw_data.append({
                "query_id": qid,
                "query": query,
                "passage": passage,
                "doc_id": doc_id
            })
        if max_samples:
            raw_data = raw_data[:max_samples]
        logging.info(f"Loaded {len(raw_data)} samples from {file_path}")

        all_passages = [item["passage"] for item in raw_data]
        processed_data = []
        for item in raw_data:
            positive_passages = [item["passage"]]
            negative_passages = []
            for _ in range(self.num_hard_negatives):
                neg_idx = random.randint(0, len(all_passages) - 1)
                while all_passages[neg_idx] == item["passage"]:
                    neg_idx = random.randint(0, len(all_passages) - 1)
                negative_passages.append(all_passages[neg_idx])
            processed_data.append({
                "query_id": item["query_id"],
                "query": item["query"],
                "positive_passages": positive_passages,
                "negative_passages": negative_passages,
                "doc_id": item["doc_id"]
            })
        return processed_data

    def to_dataframe(self):
        return pd.DataFrame(self.data)


def finetune_dpr_mrtydi_negative(
    data_path: str,
    output_dir: str = "./dpr_mrtydi_negative",
    language: str = "all",
    **kwargs,
):
    device, local_rank, world_size = setup_ddp()

    logging.basicConfig(
        level=logging.INFO if local_rank <= 0 else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    cfg = DPRMrTyDiConfig(
        data_path=data_path,
        output_dir=output_dir,
        language=language,
        **kwargs,
    )

    dataset = MrTyDiDataset(
        file_path=cfg.data_path,
        language=cfg.language,
        max_samples=cfg.max_samples,
        num_hard_negatives=cfg.num_hard_negatives,
    )
    train_df = dataset.to_dataframe()

    model_args = RetrievalArgs()
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.use_cached_eval_features = False
    model_args.include_title = False
    model_args.max_seq_length = cfg.passage_max_length
    model_args.num_train_epochs = cfg.num_epochs
    model_args.train_batch_size = cfg.batch_size
    model_args.use_hf_datasets = True
    model_args.learning_rate = cfg.learning_rate
    model_args.warmup_steps = cfg.warmup_steps
    model_args.save_steps = cfg.save_steps
    model_args.evaluate_during_training = False
    model_args.save_model_every_epoch = False
    model_args.hard_negatives = True
    model_args.n_gpu = 1
    model_args.data_format = "beir"
    model_args.ance_training = False
    model_args.output_dir = cfg.output_dir
    model_args.logging_steps = cfg.log_steps
    model_args.use_cuda = (device.type == "cuda")

    model_type = "custom"
    model_name = None
    context_name = "bert-base-multilingual-cased"
    question_name = "bert-base-multilingual-cased"

    if local_rank <= 0:
        logger.info(f"Training on {len(train_df)} samples with {cfg.num_hard_negatives} hard negatives each")
        logger.info(f"Output dir: {cfg.output_dir}")

    model = RetrievalModel(
        model_type=model_type,
        model_name=model_name,
        context_encoder_name=context_name,
        question_encoder_name=question_name,
        args=model_args,
        use_cuda=(device.type == "cuda"),
    )

    model.train_model(train_df, eval_set=None)

    if local_rank != -1:
        dist.destroy_process_group()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    MY_DATA_PATH = "./data/mrtydi/train.tsv"

    finetune_dpr_mrtydi_negative(
        data_path=MY_DATA_PATH,
        output_dir="./dpr_mrtydi_negative_finetuned",
        language="all",
        batch_size=16,
        num_epochs=10,
        learning_rate=1e-6,
        warmup_steps=500,
        save_steps=1000,
        num_hard_negatives=1,
    )
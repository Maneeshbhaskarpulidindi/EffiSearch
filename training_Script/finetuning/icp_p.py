"""
ICT Fine-tuning on MrTyDi Dataset with both ICT-P and ICT-Q variants.
Supports DDP and AMP for efficient multi-GPU training.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer,
    PreTrainedModel
)
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# ============================================================
# DDP Utilities
# ============================================================
def setup_ddp():
    """Initializes the distributed process group."""
    if 'LOCAL_RANK' not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, -1, 1

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    
    return device, local_rank, world_size

# ============================================================
# Configuration
# ============================================================
@dataclass
class ICTMrTyDiConfig:
    """Configuration for ICT fine-tuning on MrTyDi"""
    # Model variant: "passage" (ICT-P) or "query" (ICT-Q)
    variant: str = "passage"  # "passage" or "query"
    
    # Pre-trained model paths
    encoder_path: str = "./ict_models/final"
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-6
    num_epochs: int = 10
    warmup_steps: int = 5000
    max_grad_norm: float = 1.0
    
    # Tokenization
    query_max_length: int = 64
    passage_max_length: int = 256
    
    # ICT specific
    temperature: float = 0.05
    use_in_batch_negatives: bool = True
    
    # Data
    num_workers: int = 4
    
    # Paths
    output_dir: str = "./ict_mrtydi_model"
    data_path: str = "./data/mrtydi/train.tsv"
    
    # Logging
    log_steps: int = 100
    save_steps: int = 1000
    
    # MrTyDi specific
    language: str = "all"  # all, ar, bn, en, fi, id, ja, ko, ru, sw, te, th, zh

# ============================================================
# Text Processing for ICT variants
# ============================================================
class TextProcessor:
    """Handles text processing for ICT variants"""
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    @staticmethod
    def sample_query_from_passage(passage: str) -> Tuple[str, str]:
        """ICT-Passage: Sample one sentence as query, rest as passage"""
        sentences = TextProcessor.split_sentences(passage)
        
        if len(sentences) < 2:
            mid = len(passage) // 2
            query = passage[:mid].strip()
            positive = passage[mid:].strip()
            
            if len(query) < 10:
                query = passage[:min(100, len(passage))]
                positive = passage
        else:
            query_idx = random.randint(0, len(sentences) - 1)
            query = sentences[query_idx]
            
            positive_sentences = [s for i, s in enumerate(sentences) if i != query_idx]
            positive = ' '.join(positive_sentences)
        
        return query, positive
    
    @staticmethod
    def expand_query_to_passage(query: str, expansion_factor: int = 5) -> Tuple[str, str]:
        """ICT-Query: Expand query into pseudo-passage"""
        words = query.strip().split()
        
        if len(words) < 2:
            expanded = f"{query}. This document is about {query}. Information regarding {query}."
            return query, expanded
        
        expansions = [query + "."]
        
        if len(words) >= 3:
            shuffled = words.copy()
            random.shuffle(shuffled)
            expansions.append(" ".join(shuffled) + ".")
        
        key_terms = random.sample(words, min(2, len(words)))
        expansions.append(f"This document discusses {' and '.join(key_terms)}.")
        expansions.append(f"Information about {words[0]} is presented here.")
        
        if len(words) >= 2:
            expansions.append(f"The topic covers {words[-1]} and related concepts.")
            expansions.append(f"Details regarding {' '.join(words[:2])} are included.")
        
        num_sentences = min(expansion_factor, len(expansions))
        selected = random.sample(expansions, num_sentences)
        
        pseudo_passage = " ".join(selected)
        return query, pseudo_passage

# ============================================================
# MrTyDi Dataset for Both Variants
# ============================================================
class MrTyDiICTDataset(Dataset):
    """MrTyDi dataset for ICT fine-tuning (supports both ICT-P and ICT-Q)"""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        variant: str = "passage",
        query_max_len: int = 64,
        passage_max_len: int = 256,
        language: str = "all",
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
        self.language = language
        self.variant = variant  # "passage" or "query"
        
        self.data = self._load_data(file_path, max_samples)
    
    def _load_data(self, file_path: str, max_samples: Optional[int]) -> List[Dict]:
        """Load data in TSV format - handles multiple formats"""
        data = []
        
        try:
            df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip")
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            raise
        
        logging.info(f"Raw data shape: {df.shape}, Columns: {df.shape[1]}")
        
        # Check if first row is header
        first_row = df.iloc[0].tolist()
        is_header = any(isinstance(val, str) and val.lower() in 
                       ['query_id', 'query', 'passage', 'positive_passages', 
                        'negative_passages', 'language', 'doc_id'] 
                       for val in first_row)
        
        start_idx = 1 if is_header else 0
        
        if is_header:
            logging.info(f"Header detected: {first_row}")
        
        # Handle different TSV formats
        for idx, row in df.iterrows():
            if idx < start_idx:
                continue
            
            try:
                if len(row) < 2:
                    continue
                
                row_list = [str(x).strip() for x in row]
                
                # Format detection based on number of columns and content
                if len(row) >= 4:
                    # Format: query_id, query, positive_passages, negative_passages
                    query_id = row_list[0]
                    query = row_list[1]
                    passage = row_list[2]  # Use positive passages
                    lang = self.language
                    doc_id = f"D{idx}"
                elif len(row) >= 5:
                    # MrTyDi format: query_id, language, query, doc_id, passage
                    query_id = row_list[0]
                    lang = row_list[1] if len(row) > 1 else self.language
                    query = row_list[2] if len(row) > 2 else ""
                    doc_id = row_list[3] if len(row) > 3 else f"D{idx}"
                    passage = row_list[4] if len(row) > 4 else ""
                elif len(row) == 3:
                    # Format: query_id, query, passage
                    query_id = row_list[0]
                    query = row_list[1]
                    passage = row_list[2]
                    lang = self.language
                    doc_id = f"D{idx}"
                else:
                    # Format: query, passage (or passage only)
                    query_id = f"Q{idx}"
                    query = row_list[0]
                    passage = row_list[1] if len(row) > 1 else row_list[0]
                    lang = self.language
                    doc_id = f"D{idx}"
                
                # Filter by language if specified
                if self.language != "all" and lang != self.language:
                    continue
                
                # Validation - be more lenient
                if len(query.split()) < 1 or len(passage.split()) < 2:
                    continue
                
                data.append({
                    "query_id": query_id,
                    "language": lang,
                    "query": query,
                    "doc_id": doc_id,
                    "passage": passage
                })
            
            except Exception as e:
                logging.warning(f"Skipping row {idx}: {e}")
                continue
        
        if max_samples:
            data = data[:max_samples]
        
        logging.info(f"Loaded {len(data)} valid samples from {file_path}")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        if self.variant == "passage":
            # ICT-Passage: Sample query from passage
            passage = item["passage"]
            query, positive = TextProcessor.sample_query_from_passage(passage)
        else:
            # ICT-Query: Expand query to pseudo-passage
            query = item["query"]
            query, positive = TextProcessor.expand_query_to_passage(query)
        
        # Sample random negative
        neg_idx = np.random.randint(0, len(self.data))
        while neg_idx == idx:
            neg_idx = np.random.randint(0, len(self.data))
        
        if self.variant == "passage":
            negative_passage = self.data[neg_idx]["passage"]
        else:
            neg_query = self.data[neg_idx]["query"]
            _, negative_passage = TextProcessor.expand_query_to_passage(neg_query)
        
        # Tokenize
        query_enc = self.tokenizer(
            query,
            max_length=self.query_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        pos_enc = self.tokenizer(
            positive,
            max_length=self.passage_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        neg_enc = self.tokenizer(
            negative_passage,
            max_length=self.passage_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
        }

# ============================================================
# Model Loading Utilities
# ============================================================
def load_encoder_model(encoder_path: str) -> AutoModel:
    """Load encoder model handling both HF format and .pt files"""
    encoder_path = Path(encoder_path)
    
    # Case 1: Hugging Face model format
    if (encoder_path / "config.json").exists() and (encoder_path / "pytorch_model.bin").exists():
        return AutoModel.from_pretrained(str(encoder_path))
    
    # Case 2: .pt file (state dict from pre-training)
    elif (encoder_path / "model.pt").exists():
        # First load a base model, then load the state dict
        config_path = encoder_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                model_name = config_dict.get("model_name", "bert-base-multilingual-cased")
        else:
            model_name = "bert-base-multilingual-cased"
        
        model = AutoModel.from_pretrained(model_name)
        state_dict = torch.load(encoder_path / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        return model
    
    # Case 3: Default - try HF format
    else:
        logging.warning(f"Model format unclear, attempting HF format from {encoder_path}")
        return AutoModel.from_pretrained(str(encoder_path))

# ============================================================
# Model
# ============================================================
class ICTDualEncoder(nn.Module):
    """Dual encoder for ICT fine-tuning"""
    
    def __init__(self, encoder_path: str, share_weights: bool = False):
        super().__init__()
        self.share_weights = share_weights
        
        if share_weights:
            # Single shared encoder
            self.encoder = load_encoder_model(encoder_path)
            self.query_encoder = None
            self.passage_encoder = None
        else:
            # Separate encoders
            self.query_encoder = load_encoder_model(encoder_path)
            self.passage_encoder = load_encoder_model(encoder_path)
            self.encoder = None
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_type: str = "passage"
    ) -> torch.Tensor:
        """Encode text to embeddings"""
        if self.share_weights:
            encoder = self.encoder
        else:
            encoder = self.query_encoder if encoder_type == "query" else self.passage_encoder
        
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        pos_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        neg_input_ids: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass"""
        query_emb = self.encode(query_input_ids, query_attention_mask, "query")
        pos_emb = self.encode(pos_input_ids, pos_attention_mask, "passage")
        
        neg_emb = None
        if neg_input_ids is not None:
            neg_emb = self.encode(neg_input_ids, neg_attention_mask, "passage")
        
        return query_emb, pos_emb, neg_emb

# ============================================================
# Loss Function
# ============================================================
def compute_ict_loss(
    query_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: Optional[torch.Tensor] = None,
    temperature: float = 0.05,
    use_in_batch_negatives: bool = True
) -> torch.Tensor:
    """Compute ICT contrastive loss"""
    batch_size = query_emb.size(0)
    
    logits_matrix = torch.matmul(query_emb, pos_emb.T) / temperature
    
    if neg_emb is not None:
        explicit_neg_sim = torch.sum(query_emb * neg_emb, dim=1, keepdim=True) / temperature
        logits = torch.cat([logits_matrix, explicit_neg_sim], dim=1)
        labels = torch.arange(batch_size, device=query_emb.device)
    elif use_in_batch_negatives:
        logits = logits_matrix
        labels = torch.arange(batch_size, device=query_emb.device)
    else:
        logits = torch.sum(query_emb * pos_emb, dim=1).unsqueeze(1) / temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss

# ============================================================
# Trainer
# ============================================================
class ICTMrTyDiTrainer:
    """Trainer for ICT fine-tuning on MrTyDi"""
    
    def __init__(
        self,
        model: ICTDualEncoder,
        train_dataset: Dataset,
        config: ICTMrTyDiConfig,
        device: torch.device,
        local_rank: int,
        world_size: int,
    ):
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size
        
        self.logger = logging.getLogger(__name__)
        if self.local_rank > 0:
            self.logger.setLevel(logging.WARNING)
        
        # Model setup
        self.model = model.to(self.device)
        if self.local_rank != -1:
            self.model = DDP(
                self.model,
                device_ids=[self.device],
                find_unused_parameters=True
            )
        
        # Optimizer and scaler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.scaler = GradScaler()
        
        # DataLoader
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.local_rank if self.local_rank != -1 else 0,
            shuffle=True
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        self.global_step = 0
        self.epoch = 0
    
    def train(self):
        """Main training loop"""
        if self.local_rank <= 0:
            self.logger.info("=" * 80)
            self.logger.info(f"ICT-{self.config.variant.upper()} Fine-tuning on MrTyDi")
            self.logger.info(f"Language: {self.config.language}")
            self.logger.info(f"Total effective batch size: {self.config.batch_size * self.world_size}")
            self.logger.info("=" * 80)
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.train_epoch()
            
            if self.local_rank <= 0:
                self.save_checkpoint(f"epoch_{epoch}")
        
        if self.local_rank <= 0:
            self.logger.info("Fine-tuning completed!")
            self.save_checkpoint("final")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        if self.local_rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        
        progress_bar = self.train_loader
        if self.local_rank <= 0:
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}"
            )
        
        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast():
                query_emb, pos_emb, neg_emb = self.model(
                    batch["query_input_ids"],
                    batch["query_attention_mask"],
                    batch["pos_input_ids"],
                    batch["pos_attention_mask"],
                    batch["neg_input_ids"],
                    batch["neg_attention_mask"]
                )
                
                loss = compute_ict_loss(
                    query_emb,
                    pos_emb,
                    neg_emb,
                    temperature=self.config.temperature,
                    use_in_batch_negatives=self.config.use_in_batch_negatives
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            if self.local_rank <= 0:
                total_loss += loss.item()
                self.global_step += 1
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                if self.global_step % self.config.log_steps == 0:
                    avg_loss = total_loss / self.config.log_steps
                    self.logger.info(
                        f"Step {self.global_step}: loss={avg_loss:.4f}, "
                        f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                    )
                    total_loss = 0.0
                
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
    
    def save_checkpoint(self, name: str):
        """Save checkpoint"""
        if self.local_rank > 0:
            return
        
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Save model
        if model_to_save.share_weights:
            model_to_save.encoder.save_pretrained(output_dir / "encoder")
        else:
            model_to_save.query_encoder.save_pretrained(output_dir / "query_encoder")
            model_to_save.passage_encoder.save_pretrained(output_dir / "passage_encoder")
        
        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        self.logger.info(f"Checkpoint saved to {output_dir}")

# ============================================================
# Main Training Scripts
# ============================================================
def finetune_ict_mrtydi_passage(
    encoder_path: str,
    data_path: str,
    output_dir: str = "./ict_p_mrtydi",
    language: str = "all",
    **kwargs
):
    """Fine-tune ICT-Passage on MrTyDi"""
    device, local_rank, world_size = setup_ddp()
    
    logging.basicConfig(
        level=logging.INFO if local_rank <= 0 else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    if local_rank <= 0:
        logger.info("Loading ICT-Passage encoder...")
    
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    dataset = MrTyDiICTDataset(
        file_path=data_path,
        tokenizer=tokenizer,
        variant="passage",
        language=language,
        query_max_len=kwargs.get("query_max_length", 64),
        passage_max_len=kwargs.get("passage_max_length", 256),
    )
    
    config = ICTMrTyDiConfig(
        variant="passage",
        encoder_path=encoder_path,
        data_path=data_path,
        output_dir=output_dir,
        language=language,
        **kwargs
    )
    
    model = ICTDualEncoder(encoder_path, share_weights=True)
    trainer = ICTMrTyDiTrainer(model, dataset, config, device, local_rank, world_size)
    trainer.train()
    
    if local_rank != -1:
        dist.destroy_process_group()


def finetune_ict_mrtydi_query(
    encoder_path: str,
    data_path: str,
    output_dir: str = "./ict_q_mrtydi",
    language: str = "all",
    **kwargs
):
    """Fine-tune ICT-Query on MrTyDi"""
    device, local_rank, world_size = setup_ddp()
    
    logging.basicConfig(
        level=logging.INFO if local_rank <= 0 else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    if local_rank <= 0:
        logger.info("Loading ICT-Query encoder...")
    
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    dataset = MrTyDiICTDataset(
        file_path=data_path,
        tokenizer=tokenizer,
        variant="query",
        language=language,
        query_max_len=kwargs.get("query_max_length", 64),
        passage_max_len=kwargs.get("passage_max_length", 256),
    )
    
    config = ICTMrTyDiConfig(
        variant="query",
        encoder_path=encoder_path,
        data_path=data_path,
        output_dir=output_dir,
        language=language,
        **kwargs
    )
    
    model = ICTDualEncoder(encoder_path, share_weights=True)
    trainer = ICTMrTyDiTrainer(model, dataset, config, device, local_rank, world_size)
    trainer.train()
    
    if local_rank != -1:
        dist.destroy_process_group()

# ============================================================
# Example Usage
# ============================================================
if __name__ == "__main__":
    # Launch with: torchrun --nproc_per_node=4 finetune_mrtydi.py
    
    # IMPORTANT: Update these paths to your actual pre-trained model locations
    # From your pre-training script, it saves to: ./ict_passage_model/final/ or ./ict_query_model/final/
    
    # Option 1: Fine-tune ICT-Passage
    finetune_ict_mrtydi_passage(
        encoder_path="/home/maneesh/Desktop/IR/negative-sampling-repro/Repo/Negative_training_Script/finetuning/ict_models/passage/final",  # Path to pre-trained ICT-P checkpoint
        data_path="/home/maneesh/Desktop/IR/negative-sampling-repro/train_full.tsv",
        output_dir="./ict_p_mrtydi_finetuned",
        language="all",
        batch_size=16,
        num_epochs=10,
        learning_rate=1e-6,
        warmup_steps=5000,
    )
    
    # Option 2: Fine-tune ICT-Query (uncomment to use)
    # finetune_ict_mrtydi_query(
    #     encoder_path="./ict_query_model/final",  # Path to pre-trained ICT-Q checkpoint
    #     data_path="./data/mrtydi/train.tsv",
    #     output_dir="./ict_q_mrtydi_finetuned",
    #     language="all",
    #     batch_size=16,
    #     num_epochs=10,
    #     learning_rate=1e-6,
    #     warmup_steps=5000,
    # )
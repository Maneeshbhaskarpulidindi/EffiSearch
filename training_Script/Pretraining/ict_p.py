"""
Completed Inverse Cloze Task (ICT) Implementation with DDP and AMP.
FIXED: Added find_unused_parameters=True to DDP and simplified loss function.
"""

import os
import re
import random
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer,
    PreTrainedModel
)
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import json

# ============================================================
# DDP Utilities 🌐
# ============================================================
def setup_ddp():
    """Initializes the distributed process group."""
    if 'LOCAL_RANK' not in os.environ:
        # Not using torchrun/DDP, run on a single device
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
class ICTConfig:
    """Configuration for ICT training"""
    # Model
    model_name: str = "bert-base-multilingual-cased"
    share_weights: bool = True # Changed to True as default to simplify DDP
    
    # Training
    batch_size: int = 32 # Batch size per GPU
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 1000
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
    output_dir: str = "./ict_models"
    
    # Logging
    log_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 5000

# ============================================================
# Text Processing for ICT
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
# ICT Datasets
# ============================================================
# Datasets remain the same, handling data per sample.
class ICTPassageDataset(Dataset):
    # ... (Dataset implementation remains the same)
    def __init__(
        self, 
        passages: List[str],
        tokenizer: PreTrainedTokenizer,
        query_max_len: int = 64,
        passage_max_len: int = 256
    ):
        self.passages = passages
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
    
    def __len__(self) -> int:
        return len(self.passages)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        passage = self.passages[idx]
        
        query, positive = TextProcessor.sample_query_from_passage(passage)
        
        # Random negative from different passage
        neg_idx = random.randint(0, len(self.passages) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.passages) - 1)
        negative = self.passages[neg_idx]
        
        # Tokenize (removed unnecessary `.squeeze(0)` for batching clarity)
        query_enc = self.tokenizer(
            query, max_length=self.query_max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        pos_enc = self.tokenizer(
            positive, max_length=self.passage_max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        neg_enc = self.tokenizer(
            negative, max_length=self.passage_max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
        }


class ICTQueryDataset(Dataset):
    # ... (Dataset implementation remains the same)
    def __init__(
        self, 
        queries: List[str],
        tokenizer: PreTrainedTokenizer,
        query_max_len: int = 64,
        passage_max_len: int = 256
    ):
        self.queries = queries
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query_text = self.queries[idx]
        
        query, positive = TextProcessor.expand_query_to_passage(query_text)
        
        neg_idx = random.randint(0, len(self.queries) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.queries) - 1)
        _, negative = TextProcessor.expand_query_to_passage(self.queries[neg_idx])
        
        query_enc = self.tokenizer(
            query, max_length=self.query_max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        pos_enc = self.tokenizer(
            positive, max_length=self.passage_max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        neg_enc = self.tokenizer(
            negative, max_length=self.passage_max_len, padding="max_length", truncation=True, return_tensors="pt"
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
# Model
# ============================================================
class ICTDualEncoder(nn.Module):
    """Dual encoder for ICT with optional weight sharing"""
    
    def __init__(self, model_name: str, share_weights: bool = False):
        super().__init__()
        self.share_weights = share_weights
        
        # IMPORTANT: Initialize ALL possible components regardless of share_weights.
        # This prevents the DDP 'unused parameters' error from occurring due to initialization logic.
        # If share_weights is False, we use two separate encoders.
        # If share_weights is True, we only use self.encoder in the forward pass.
        
        if share_weights:
            # Single encoder for both query and passage
            self.encoder = AutoModel.from_pretrained(model_name)
            self.query_encoder = None # Keep as None/Placeholder if not used
            self.passage_encoder = None
        else:
            # Separate encoders
            self.query_encoder = AutoModel.from_pretrained(model_name)
            self.passage_encoder = AutoModel.from_pretrained(model_name)
            self.encoder = None # Keep as None/Placeholder if not used
    
    def encode(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        encoder_type: str = "passage"
    ) -> torch.Tensor:
        """Encode text to embeddings"""
        if self.share_weights:
            # Shared: uses self.encoder for both query and passage
            encoder = self.encoder
        else:
            # Separate: uses the relevant encoder
            encoder = self.query_encoder if encoder_type == "query" else self.passage_encoder
        
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    # Forward pass logic is clean and uses all parameters based on encoder_type logic
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
# Simplified loss when using in-batch negatives (Standard practice)
def compute_ict_loss(
    query_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: Optional[torch.Tensor] = None,
    temperature: float = 0.05,
    use_in_batch_negatives: bool = True
) -> torch.Tensor:
    """Compute ICT contrastive loss using the in-batch negatives method."""
    batch_size = query_emb.size(0)
    
    # 1. Compute similarity matrix (Query_i vs Pos_j)
    logits_matrix = torch.matmul(query_emb, pos_emb.T) / temperature # [B, B]
    
    # 2. Add explicit negatives if provided
    if neg_emb is not None:
        # Similarity between Query_i and Neg_i
        explicit_neg_sim = torch.sum(query_emb * neg_emb, dim=1, keepdim=True) / temperature # [B, 1]
        
        # Combine in-batch negatives (logits_matrix) with explicit negatives
        # The true positive is on the diagonal of logits_matrix.
        # We append the explicit negatives as extra columns.
        logits = torch.cat([logits_matrix, explicit_neg_sim], dim=1) # [B, B+1]
        labels = torch.arange(batch_size, device=query_emb.device) # Target is still the diagonal (0 to B-1)
    
    elif use_in_batch_negatives:
        # Use only in-batch negatives (logits_matrix itself)
        logits = logits_matrix
        labels = torch.arange(batch_size, device=query_emb.device) # Target is the diagonal
    
    else:
        # Fallback for clarity (though this case is usually too weak for good training)
        logits = torch.sum(query_emb * pos_emb, dim=1).unsqueeze(1) / temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss

# ============================================================
# Trainer (Crucial DDP and AMP changes)
# ============================================================
class ICTTrainer:
    """Trainer for ICT models with DDP and AMP support"""
    
    def __init__(
        self,
        model: ICTDualEncoder,
        train_dataset: Dataset,
        config: ICTConfig,
        device: torch.device,
        local_rank: int,
        world_size: int,
        eval_dataset: Optional[Dataset] = None
    ):
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size
        
        self.logger = logging.getLogger(__name__)
        if self.local_rank > 0:
            self.logger.setLevel(logging.WARNING) 
        
        # 1. Model Setup and DDP Wrapping
        self.model = model.to(self.device)
        if self.local_rank != -1:
            # FIX: Added find_unused_parameters=True to prevent DDP RuntimeError
            self.model = DDP(
                self.model, 
                device_ids=[self.device], 
                find_unused_parameters=True 
            )
        
        # 2. Setup Optimizer and AMP Scaler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.scaler = GradScaler()
        
        # 3. Setup Distributed Sampler and DataLoader
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
        
        # 4. Setup Scheduler
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
            self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
            self.logger.info(f"Total effective batch size: {self.config.batch_size * self.world_size}")
            self.logger.info(f"Total steps: {len(self.train_loader) * self.config.num_epochs}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.train_epoch()
            
            if self.local_rank <= 0:
                self.save_checkpoint(f"epoch_{epoch}")
        
        if self.local_rank <= 0:
            self.logger.info("Training completed!")
            self.save_checkpoint("final")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        # Crucial: Set epoch for DistributedSampler
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
            
            # Backward pass and Optimization
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # WARNING FIX: Move scheduler step after optimizer.step()
            self.scheduler.step() 
            
            # Tracking and Logging (Rank 0 only)
            if self.local_rank <= 0:
                total_loss += loss.item()
                self.global_step += 1
            
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                
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
        """Save model checkpoint (only rank 0 saves)"""
        if self.local_rank > 0:
            return
            
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        
        torch.save(
            model_to_save.state_dict(),
            output_dir / "model.pt"
        )
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        self.logger.info(f"Checkpoint saved to {output_dir}")

# ============================================================
# Data Loading Utilities
# ============================================================
def load_passages_from_file(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    # ... (function remains the same)
    df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip")
    
    if df.shape[1] >= 3:
        passages = pd.concat([df.iloc[:, 1], df.iloc[:, 2]]).dropna().unique().tolist()
    elif df.shape[1] == 2:
        passages = df.iloc[:, 1].dropna().tolist()
    else:
        passages = df.iloc[:, 0].dropna().tolist()
    
    passages = [p for p in passages if len(str(p).split()) > 5]
    
    if max_samples:
        passages = passages[:max_samples]
    
    return passages


def load_queries_from_file(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    # ... (function remains the same)
    df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip")
    
    if df.shape[1] >= 2:
        queries = df.iloc[:, 1].dropna().tolist()
    else:
        queries = df.iloc[:, 0].dropna().tolist()
    
    queries = [q for q in queries if len(str(q).split()) >= 2]
    
    if max_samples:
        queries = queries[:max_samples]
    
    return queries

# ============================================================
# Main Training Scripts
# ============================================================
def train_ict_passage(
    data_path: str,
    output_dir: str = "./ict_passage_model",
    **kwargs
):
    """Train ICT-Passage model using DDP"""
    device, local_rank, world_size = setup_ddp()
    
    logging.basicConfig(level=logging.INFO if local_rank <= 0 else logging.WARNING)
    logger = logging.getLogger(__name__)

    if local_rank <= 0:
        logger.info("="*60)
        logger.info(f"Starting ICT-Passage Training on Rank {local_rank} of {world_size}")
        logger.info("="*60)
    
    if local_rank <= 0:
        logger.info(f"Loading passages from {data_path}")
    passages = load_passages_from_file(data_path)
    if local_rank <= 0:
        logger.info(f"Loaded {len(passages)} passages")
    
    config = ICTConfig(output_dir=output_dir, **kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    dataset = ICTPassageDataset(
        passages=passages,
        tokenizer=tokenizer,
        query_max_len=config.query_max_length,
        passage_max_len=config.passage_max_length
    )
    
    model = ICTDualEncoder(config.model_name, config.share_weights)
    
    trainer = ICTTrainer(model, dataset, config, device, local_rank, world_size)
    
    trainer.train()
    
    if local_rank != -1:
        dist.destroy_process_group()


def train_ict_query(
    data_path: str,
    output_dir: str = "./ict_query_model",
    **kwargs
):
    """Train ICT-Query model using DDP"""
    device, local_rank, world_size = setup_ddp()
    
    logging.basicConfig(level=logging.INFO if local_rank <= 0 else logging.WARNING)
    logger = logging.getLogger(__name__)

    if local_rank <= 0:
        logger.info("="*60)
        logger.info(f"Starting ICT-Query Training on Rank {local_rank} of {world_size}")
        logger.info("="*60)
    
    if local_rank <= 0:
        logger.info(f"Loading queries from {data_path}")
    queries = load_queries_from_file(data_path)
    if local_rank <= 0:
        logger.info(f"Loaded {len(queries)} queries")
    
    config = ICTConfig(output_dir=output_dir, **kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    dataset = ICTQueryDataset(
        queries=queries,
        tokenizer=tokenizer,
        query_max_len=config.query_max_length,
        passage_max_len=config.passage_max_length
    )
    
    model = ICTDualEncoder(config.model_name, config.share_weights)
    
    trainer = ICTTrainer(model, dataset, config, device, local_rank, world_size)
    
    trainer.train()
    
    if local_rank != -1:
        dist.destroy_process_group()

# ============================================================
# Example Usage
# ============================================================
if __name__ == "__main__":
    # Your request: focus on ICT-Query only
    
    # Example 1: Train ICT-Passage (commented out as per request)
    train_ict_passage(
        data_path="/home/maneesh/Desktop/data/msmarco/msmarco-train.tsv",
        output_dir="./ict_passage_final_model",
        batch_size=32,
        num_epochs=10,
        learning_rate=2e-5
    )
    
    # Example 2: Train ICT-Query
    # NOTE: You MUST launch this file using torchrun:
    # torchrun --nproc_per_node=2 just.py
    
    # Using the same MS MARCO file, assuming it contains passage data
    # that can be interpreted as queries or passages for simplicity.
    # For a real ICT-Query task, this data_path should ideally point to a query corpus.
    
    #train_ict_query(
    #    data_path="/home/maneesh/Desktop/data/msmarco/msmarco-train.tsv",
    #    output_dir="./ict_query_model",
    #    batch_size=32,
    #    num_epochs=10,
    #    learning_rate=2e-5,
    #    share_weights=True # Explicitly using shared weights
    #)
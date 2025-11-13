#!/usr/bin/env python3
"""
Knowledge Distillation for ICT Models with Hybrid Loss
Implements distillation + retrieval supervision
"""

import os
import json
import torch
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, InputExample, models
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
@dataclass
class DistillationConfig:
    teacher_model_path: str
    student_model_name: str = "distilbert-base-multilingual-cased"
    num_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 2e-5
    output_dir: str = "./distilled_models"
    model_name: str = "ict_distilled"
    data_path: str = None
    num_samples: int = 500
    # Hybrid loss parameters
    lambda_weight: float = 0.5  # Weight for combining losses (λ in paper)
    temperature: float = 1.0     # Temperature for retrieval loss (τ in paper)

# ============================================================
# Loss Functions
# ============================================================
class HybridLoss(nn.Module):
    """
    Hybrid Loss = λ * L_distill + (1 - λ) * L_retrieval
    
    L_distill: MSE between student and teacher embeddings
    L_retrieval: Contrastive loss for query-passage matching
    """
    def __init__(self, lambda_weight=0.5, temperature=1.0):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        logger.info(f"HybridLoss initialized: λ={lambda_weight}, τ={temperature}")
    
    def distillation_loss(self, student_emb, teacher_emb):
        """MSE loss between student and teacher embeddings"""
        return F.mse_loss(student_emb, teacher_emb)
    
    def retrieval_loss(self, query_emb, passage_emb):
        """
        Contrastive retrieval loss (InfoNCE-style)
        
        L_retrieval = -1/N * Σ log[ exp(sim(q_i, p_i*)/τ) / Σ_j exp(sim(q_i, p_j)/τ) ]
        
        Args:
            query_emb: [batch_size, embedding_dim]
            passage_emb: [batch_size, embedding_dim]
        """
        batch_size = query_emb.shape[0]
        
        # Compute similarity matrix: [batch_size, batch_size]
        # sim[i, j] = cosine_similarity(query_i, passage_j)
        sim_matrix = F.cosine_similarity(
            query_emb.unsqueeze(1),      # [B, 1, D]
            passage_emb.unsqueeze(0),    # [1, B, D]
            dim=2
        ) / self.temperature  # Scale by temperature
        
        # Positive pairs are on the diagonal
        # We want to maximize sim[i, i] relative to sim[i, j] for j != i
        labels = torch.arange(batch_size, device=query_emb.device)
        
        # Cross-entropy loss treats each row as a classification problem
        # where the correct class is the diagonal element
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def forward(self, student_query_emb, student_passage_emb, 
                teacher_query_emb, teacher_passage_emb):
        """
        Compute hybrid loss
        
        Args:
            student_query_emb: Student query embeddings [B, D]
            student_passage_emb: Student passage embeddings [B, D]
            teacher_query_emb: Teacher query embeddings [B, D]
            teacher_passage_emb: Teacher passage embeddings [B, D]
        """
        # Distillation loss: match teacher embeddings
        # Average loss for both queries and passages
        distill_loss_q = self.distillation_loss(student_query_emb, teacher_query_emb)
        distill_loss_p = self.distillation_loss(student_passage_emb, teacher_passage_emb)
        distill_loss = (distill_loss_q + distill_loss_p) / 2
        
        # Retrieval loss: query-passage matching
        retrieval_loss = self.retrieval_loss(student_query_emb, student_passage_emb)
        
        # Hybrid loss
        hybrid_loss = (self.lambda_weight * distill_loss + 
                      (1 - self.lambda_weight) * retrieval_loss)
        
        return {
            'total': hybrid_loss,
            'distillation': distill_loss.item(),
            'retrieval': retrieval_loss.item(),
        }

# ============================================================
# Data Loading
# ============================================================
class SentenceTransformerDataset:
    """Simple dataset wrapper for InputExample objects"""
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    """Custom collate function that separates queries and passages"""
    queries = [example.texts[0] for example in batch]
    passages = [example.texts[1] for example in batch]
    return queries, passages

def load_data_from_file(file_path: str, max_samples: int) -> list:
    """Load data from TSV file"""
    data = []
    
    try:
        df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip")
        logger.info(f"Loaded TSV with shape: {df.shape}")
        
        # Skip header if exists
        start_idx = 1 if "query_id" in str(df.iloc[0].values) else 0
        
        for idx in range(start_idx, min(len(df), start_idx + max_samples)):
            row = df.iloc[idx]
            
            # Handle different formats
            if len(row) >= 4:
                query = str(row[1]).strip()
                passage = str(row[2]).strip()
            elif len(row) >= 3:
                query = str(row[1]).strip()
                passage = str(row[2]).strip()
            elif len(row) >= 2:
                query = str(row[0]).strip()
                passage = str(row[1]).strip()
            else:
                continue
            
            if query and passage and len(query.split()) > 1 and len(passage.split()) > 2:
                data.append(InputExample(texts=[query, passage]))
        
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# ============================================================
# Model Loading
# ============================================================
def load_pretrained_encoder(model_path: str) -> SentenceTransformer:
    """Load pre-trained encoder"""
    model_path = Path(model_path)
    
    logger.info(f"Loading teacher model from: {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found at {model_path}")
    
    try:
        logger.info("Loading SentenceTransformer...")
        model = SentenceTransformer(str(model_path))
        logger.info("✓ Teacher model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load teacher model: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(
            f"Cannot load model from {model_path}\n"
            f"Error: {str(e)}\n\n"
            f"SOLUTION: Run rebuild_model.py first to create complete model files"
        )

def create_student_model(student_name: str) -> SentenceTransformer:
    """Create student model"""
    logger.info(f"Creating student model: {student_name}")
    transformer = models.Transformer(student_name, max_seq_length=256)
    pooling = models.Pooling(transformer.get_word_embedding_dimension())
    student = SentenceTransformer(modules=[transformer, pooling])
    logger.info("✓ Student model created")
    return student

# ============================================================
# Training
# ============================================================
class Trainer:
    def __init__(self, teacher, student, config, train_data):
        self.teacher = teacher
        self.student = student
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.teacher.to(self.device)
        self.student.to(self.device)
        
        # Initialize hybrid loss
        self.criterion = HybridLoss(
            lambda_weight=config.lambda_weight,
            temperature=config.temperature
        )
        
        # Use custom collate function
        dataset = SentenceTransformerDataset(train_data)
        self.train_loader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=config.learning_rate)
        logger.info(f"Optimizer created with learning rate: {config.learning_rate}")
    
    def train(self):
        logger.info("Starting training with hybrid loss...")
        logger.info(f"Lambda (λ): {self.config.lambda_weight}")
        logger.info(f"Temperature (τ): {self.config.temperature}\n")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            epoch_distill_loss = 0
            epoch_retrieval_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for queries, passages in pbar:
                # Get teacher embeddings (no gradients)
                with torch.no_grad():
                    teacher_query_emb = self.teacher.encode(
                        queries, convert_to_tensor=True, device=self.device, 
                        show_progress_bar=False
                    ).clone()
                    teacher_passage_emb = self.teacher.encode(
                        passages, convert_to_tensor=True, device=self.device,
                        show_progress_bar=False
                    ).clone()
                
                # Get student embeddings (with gradients)
                # Process queries
                query_features = self.student.tokenize(queries)
                for key in query_features:
                    if isinstance(query_features[key], torch.Tensor):
                        query_features[key] = query_features[key].to(self.device)
                student_query_output = self.student.forward(query_features)
                student_query_emb = student_query_output['sentence_embedding']
                
                # Process passages
                passage_features = self.student.tokenize(passages)
                for key in passage_features:
                    if isinstance(passage_features[key], torch.Tensor):
                        passage_features[key] = passage_features[key].to(self.device)
                student_passage_output = self.student.forward(passage_features)
                student_passage_emb = student_passage_output['sentence_embedding']
                
                # Compute hybrid loss
                loss_dict = self.criterion(
                    student_query_emb, student_passage_emb,
                    teacher_query_emb, teacher_passage_emb
                )
                
                loss = loss_dict['total']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.optimizer.step()
                
                # Track losses
                epoch_loss += loss.item()
                epoch_distill_loss += loss_dict['distillation']
                epoch_retrieval_loss += loss_dict['retrieval']
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "distill": f"{loss_dict['distillation']:.4f}",
                    "retrieval": f"{loss_dict['retrieval']:.4f}"
                })
            
            # Log epoch statistics
            n_batches = len(self.train_loader)
            avg_loss = epoch_loss / n_batches
            avg_distill = epoch_distill_loss / n_batches
            avg_retrieval = epoch_retrieval_loss / n_batches
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"loss={avg_loss:.4f} | "
                f"distill={avg_distill:.4f} | "
                f"retrieval={avg_retrieval:.4f}"
            )
            
            self.save_checkpoint(f"epoch_{epoch+1}")
        
        self.save_checkpoint("final")
    
    def save_checkpoint(self, name: str):
        output_dir = Path(self.config.output_dir) / self.config.model_name / name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.student.save(str(output_dir))
        logger.info(f"✓ Checkpoint saved to {output_dir}")

# ============================================================
# Main
# ============================================================
def distill_ict_model(
    variant: str = "passage",
    teacher_path: str = None,
    data_path: str = None,
    output_dir: str = "./distilled_models",
    num_epochs: int = 3,
    batch_size: int = 64,
    num_samples: int = 500,
    lambda_weight: float = 0.5,
    temperature: float = 1.0,
):
    logger.info("\n" + "="*80)
    logger.info(f"ICT-{variant.upper()} KNOWLEDGE DISTILLATION (HYBRID LOSS)")
    logger.info("="*80 + "\n")
    
    # Validate inputs
    if not teacher_path:
        raise ValueError("teacher_path is required")
    if not data_path:
        raise ValueError("data_path is required")
    
    config = DistillationConfig(
        teacher_model_path=teacher_path,
        output_dir=output_dir,
        model_name=f"ict_{variant}_distilled_hybrid",
        num_epochs=num_epochs,
        batch_size=batch_size,
        data_path=data_path,
        num_samples=num_samples,
        lambda_weight=lambda_weight,
        temperature=temperature,
    )
    
    # Load teacher
    teacher = load_pretrained_encoder(teacher_path)
    
    # Create student
    student = create_student_model(config.student_model_name)
    
    # Log model sizes
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Teacher parameters: {teacher_params:,}")
    logger.info(f"Student parameters: {student_params:,}")
    logger.info(f"Compression ratio: {teacher_params/student_params:.2f}x\n")
    
    # Load data
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    train_data = load_data_from_file(data_path, num_samples)
    
    if not train_data:
        raise ValueError("No training data loaded!")
    
    # Train
    trainer = Trainer(teacher, student, config, train_data)
    trainer.train()
    
    logger.info("\n" + "="*80)
    logger.info("✓ DISTILLATION COMPLETE")
    logger.info("="*80 + "\n")
    
    return student

# ============================================================
# Execute
# ============================================================
if __name__ == "__main__":
    student_model = distill_ict_model(
        variant="passage",
        teacher_path="/home/maneesh/Desktop/IR/negative-sampling-repro/Repo/Distillation_Script/ict_p_mrtydi_finetuned/final/encoder_complete",
        data_path="/home/maneesh/Desktop/IR/negative-sampling-repro/train_full.tsv",
        output_dir="./distilled_models",
        num_epochs=3,
        batch_size=64,
        num_samples=48730,
        lambda_weight=0.5,    # λ: balance between distillation and retrieval
        temperature=1.0,      # τ: temperature for retrieval loss
    )
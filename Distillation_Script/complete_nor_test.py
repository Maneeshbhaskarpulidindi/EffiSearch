#!/usr/bin/env python3
"""
Knowledge Distillation for ICT Models with Hybrid Loss & Dynamic Learning
CORRECTED: Proper loss scaling with cosine similarity distillation
"""

import os
import json
import torch
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from torch.cuda.amp import autocast, GradScaler

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
    lambda_weight: float = 0.5
    temperature: float = 1.0
    # Dynamic learning parameters
    use_dynamic_lambda: bool = True
    use_lr_scheduling: bool = True
    warmup_steps: int = 500
    use_mixed_precision: bool = True
    grad_accumulation_steps: int = 2
    use_dynamic_temp: bool = True

# ============================================================
# Learning Rate Schedulers
# ============================================================
class WarmupLinearScheduler:
    """Warmup followed by linear decay"""
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr_scale = self.current_step / self.warmup_steps
        else:
            lr_scale = max(0, (self.total_steps - self.current_step) / (self.total_steps - self.warmup_steps))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['base_lr'] * lr_scale
        
        return param_group['lr']

class CosineAnnealingWarmupRestarts:
    """Cosine annealing with warmup"""
    def __init__(self, optimizer, warmup_steps, first_cycle_steps, max_epochs=3):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.first_cycle_steps = first_cycle_steps
        self.max_epochs = max_epochs
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            lr_scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.first_cycle_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['base_lr'] * lr_scale
        
        return param_group['lr']

# ============================================================
# Dynamic Loss Weighting
# ============================================================
class DynamicLambda:
    """Adaptively adjust λ based on loss ratio"""
    def __init__(self, initial_lambda=0.5, min_lambda=0.2, max_lambda=0.8):
        self.lambda_val = initial_lambda
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.distill_history = []
        self.retrieval_history = []
        self.window_size = 10
    
    def update(self, distill_loss, retrieval_loss):
        self.distill_history.append(distill_loss)
        self.retrieval_history.append(retrieval_loss)
        
        if len(self.distill_history) > self.window_size:
            self.distill_history.pop(0)
            self.retrieval_history.pop(0)
        
        if len(self.distill_history) >= 5:
            avg_distill = np.mean(self.distill_history[-5:])
            avg_retrieval = np.mean(self.retrieval_history[-5:])
            
            # If distillation loss is too high, increase its weight
            if avg_distill > avg_retrieval * 1.5:
                self.lambda_val = min(self.max_lambda, self.lambda_val + 0.02)
            # If retrieval loss is too high, increase its weight
            elif avg_retrieval > avg_distill * 1.5:
                self.lambda_val = max(self.min_lambda, self.lambda_val - 0.02)
        
        return self.lambda_val

# ============================================================
# Loss Functions - CORRECTED VERSION
# ============================================================
class HybridLoss(nn.Module):
    """Hybrid Loss with proper scaling"""
    def __init__(self, lambda_weight=0.5, temperature=1.0, use_dynamic_temp=True):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.base_temperature = temperature
        self.use_dynamic_temp = use_dynamic_temp
        self.temp_history = []
        logger.info(f"HybridLoss initialized: λ={lambda_weight}, τ={temperature}")
    
    def normalize_embeddings(self, embeddings):
        """L2 normalize embeddings to unit vectors"""
        return F.normalize(embeddings, p=2, dim=1)
    
    def distillation_loss(self, student_emb, teacher_emb):
        """
        Cosine similarity loss for distillation.
        Measures alignment between student and teacher embeddings.
        Range: [0, 2] where 0 = perfect alignment, 2 = opposite
        """
        # Normalize both embeddings
        student_norm = self.normalize_embeddings(student_emb)
        teacher_norm = self.normalize_embeddings(teacher_emb)
        
        # Cosine similarity for each pair (element-wise)
        cos_sim = F.cosine_similarity(student_norm, teacher_norm, dim=1)  # [B]
        
        # Loss = 1 - cos_sim (maximize similarity = minimize loss)
        # For random embeddings: cos_sim ≈ 0, loss ≈ 1.0
        # For aligned embeddings: cos_sim ≈ 1, loss ≈ 0.0
        loss = (1 - cos_sim).mean()
        
        return loss
    
    def retrieval_loss(self, query_emb, passage_emb):
        """
        Contrastive retrieval loss (InfoNCE-style).
        Measures how well query embeddings match positive passages.
        """
        batch_size = query_emb.shape[0]
        embedding_dim = query_emb.shape[1]
        
        # Normalize embeddings to unit length
        query_norm = self.normalize_embeddings(query_emb)
        passage_norm = self.normalize_embeddings(passage_emb)
        
        # Compute similarity matrix [B, B]
        sim_matrix = torch.mm(query_norm, passage_norm.t())  # Cosine similarity
        
        # Scale by sqrt(embedding_dim) for stable cross-entropy
        # Standard in contrastive learning (SimCLR, MoCo)
        scale = np.sqrt(embedding_dim)
        sim_matrix = sim_matrix * scale / self.temperature
        
        # Cross-entropy loss
        labels = torch.arange(batch_size, device=query_emb.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def update_temperature(self, retrieval_loss):
        """Dynamically adjust temperature based on retrieval loss"""
        if self.use_dynamic_temp:
            self.temp_history.append(retrieval_loss)
            
            if len(self.temp_history) >= 3:
                recent_loss = np.mean(self.temp_history[-3:])
                
                # If loss is high, increase temperature to soften targets
                if recent_loss > 5.0:
                    self.temperature = min(2.0, self.temperature + 0.05)
                # If loss is very low, decrease temperature to sharpen
                elif recent_loss < 3.5:
                    self.temperature = max(0.5, self.temperature - 0.05)
    
    def forward(self, student_query_emb, student_passage_emb, 
                teacher_query_emb, teacher_passage_emb, lambda_weight=None):
        """Compute hybrid loss"""
        if lambda_weight is not None:
            self.lambda_weight = lambda_weight
        
        # Distillation: align student with teacher
        distill_loss_q = self.distillation_loss(student_query_emb, teacher_query_emb)
        distill_loss_p = self.distillation_loss(student_passage_emb, teacher_passage_emb)
        distill_loss = (distill_loss_q + distill_loss_p) / 2
        
        # Retrieval: match queries with passages
        retrieval_loss = self.retrieval_loss(student_query_emb, student_passage_emb)
        
        # Update dynamic temperature
        self.update_temperature(retrieval_loss.item())
        
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
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    queries = [example.texts[0] for example in batch]
    passages = [example.texts[1] for example in batch]
    return queries, passages

def load_data_from_file(file_path: str, max_samples: int) -> list:
    """Load data from TSV file"""
    data = []
    
    try:
        df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip")
        logger.info(f"Loaded TSV with shape: {df.shape}")
        
        start_idx = 1 if "query_id" in str(df.iloc[0].values) else 0
        
        for idx in range(start_idx, min(len(df), start_idx + max_samples)):
            row = df.iloc[idx]
            
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
    """Create student model with L2 normalization layer"""
    logger.info(f"Creating student model: {student_name}")
    transformer = models.Transformer(student_name, max_seq_length=256)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode='mean'
    )
    normalize = models.Normalize()
    student = SentenceTransformer(modules=[transformer, pooling, normalize])
    logger.info("✓ Student model created with L2 normalization")
    return student

# ============================================================
# Training with Dynamic Learning
# ============================================================
class DynamicTrainer:
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
            temperature=config.temperature,
            use_dynamic_temp=config.use_dynamic_temp
        )
        
        dataset = SentenceTransformerDataset(train_data)
        self.train_loader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=config.learning_rate)
        
        # Add base_lr for schedulers
        for param_group in self.optimizer.param_groups:
            param_group['base_lr'] = config.learning_rate
        
        # Learning rate scheduler
        if config.use_lr_scheduling:
            total_steps = len(self.train_loader) * config.num_epochs
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                warmup_steps=config.warmup_steps,
                first_cycle_steps=total_steps
            )
            logger.info(f"LR Scheduler: Cosine Annealing with {config.warmup_steps} warmup steps")
        
        # Dynamic lambda
        if config.use_dynamic_lambda:
            self.dynamic_lambda = DynamicLambda(initial_lambda=config.lambda_weight)
            logger.info("Dynamic λ adjustment enabled")
        
        # Mixed precision
        if config.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled (FP16)")
        
        logger.info(f"Gradient accumulation: {config.grad_accumulation_steps} steps")
    
    def train(self):
        logger.info("Starting training with dynamic learning...")
        logger.info(f"Initial λ: {self.config.lambda_weight}")
        logger.info(f"Initial τ: {self.config.temperature}\n")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            epoch_distill_loss = 0
            epoch_retrieval_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            accumulated_loss = 0
            accumulation_step = 0
            
            for batch_idx, (queries, passages) in enumerate(pbar):
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
                    # Normalize teacher embeddings
                    teacher_query_emb = F.normalize(teacher_query_emb, p=2, dim=1)
                    teacher_passage_emb = F.normalize(teacher_passage_emb, p=2, dim=1)
                
                # Get student embeddings with mixed precision
                if self.config.use_mixed_precision:
                    with autocast():
                        query_features = self.student.tokenize(queries)
                        for key in query_features:
                            if isinstance(query_features[key], torch.Tensor):
                                query_features[key] = query_features[key].to(self.device)
                        student_query_output = self.student.forward(query_features)
                        student_query_emb = student_query_output['sentence_embedding']
                        
                        passage_features = self.student.tokenize(passages)
                        for key in passage_features:
                            if isinstance(passage_features[key], torch.Tensor):
                                passage_features[key] = passage_features[key].to(self.device)
                        student_passage_output = self.student.forward(passage_features)
                        student_passage_emb = student_passage_output['sentence_embedding']
                        
                        # Update dynamic lambda
                        current_lambda = self.config.lambda_weight
                        if self.config.use_dynamic_lambda:
                            current_lambda = self.dynamic_lambda.lambda_val
                        
                        # Compute loss
                        loss_dict = self.criterion(
                            student_query_emb, student_passage_emb,
                            teacher_query_emb, teacher_passage_emb,
                            lambda_weight=current_lambda
                        )
                        loss = loss_dict['total'] / self.config.grad_accumulation_steps
                else:
                    query_features = self.student.tokenize(queries)
                    for key in query_features:
                        if isinstance(query_features[key], torch.Tensor):
                            query_features[key] = query_features[key].to(self.device)
                    student_query_output = self.student.forward(query_features)
                    student_query_emb = student_query_output['sentence_embedding']
                    
                    passage_features = self.student.tokenize(passages)
                    for key in passage_features:
                        if isinstance(passage_features[key], torch.Tensor):
                            passage_features[key] = passage_features[key].to(self.device)
                    student_passage_output = self.student.forward(passage_features)
                    student_passage_emb = student_passage_output['sentence_embedding']
                    
                    current_lambda = self.config.lambda_weight
                    if self.config.use_dynamic_lambda:
                        current_lambda = self.dynamic_lambda.lambda_val
                    
                    loss_dict = self.criterion(
                        student_query_emb, student_passage_emb,
                        teacher_query_emb, teacher_passage_emb,
                        lambda_weight=current_lambda
                    )
                    loss = loss_dict['total'] / self.config.grad_accumulation_steps
                
                # Backward pass
                if self.config.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
                accumulation_step += 1
                
                # Gradient accumulation step
                if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                    if self.config.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    
                    if self.config.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # Update learning rate
                    if self.config.use_lr_scheduling:
                        current_lr = self.scheduler.step()
                    
                    # Update dynamic lambda
                    if self.config.use_dynamic_lambda:
                        self.dynamic_lambda.update(loss_dict['distillation'], loss_dict['retrieval'])
                    
                    epoch_loss += accumulated_loss
                    epoch_distill_loss += loss_dict['distillation']
                    epoch_retrieval_loss += loss_dict['retrieval']
                    
                    pbar.set_postfix({
                        "loss": f"{accumulated_loss:.4f}",
                        "distill": f"{loss_dict['distillation']:.4f}",
                        "retrieval": f"{loss_dict['retrieval']:.4f}",
                        "λ": f"{self.dynamic_lambda.lambda_val:.3f}" if self.config.use_dynamic_lambda else f"{self.config.lambda_weight:.3f}",
                        "τ": f"{self.criterion.temperature:.2f}"
                    })
                    
                    accumulated_loss = 0
                    accumulation_step = 0
            
            # Log epoch statistics
            n_batches = len(self.train_loader) // self.config.grad_accumulation_steps
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_distill = epoch_distill_loss / max(n_batches, 1)
            avg_retrieval = epoch_retrieval_loss / max(n_batches, 1)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"loss={avg_loss:.4f} | distill={avg_distill:.4f} | retrieval={avg_retrieval:.4f}"
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
    use_dynamic_lambda: bool = True,
    use_lr_scheduling: bool = True,
    use_mixed_precision: bool = True,
    grad_accumulation_steps: int = 2,
):
    logger.info("\n" + "="*80)
    logger.info(f"ICT-{variant.upper()} KNOWLEDGE DISTILLATION (CORRECTED)")
    logger.info("="*80 + "\n")
    
    if not teacher_path:
        raise ValueError("teacher_path is required")
    if not data_path:
        raise ValueError("data_path is required")
    
    config = DistillationConfig(
        teacher_model_path=teacher_path,
        output_dir=output_dir,
        model_name=f"ict_{variant}_distilled_corrected",
        num_epochs=num_epochs,
        batch_size=batch_size,
        data_path=data_path,
        num_samples=num_samples,
        lambda_weight=lambda_weight,
        temperature=temperature,
        use_dynamic_lambda=use_dynamic_lambda,
        use_lr_scheduling=use_lr_scheduling,
        use_mixed_precision=use_mixed_precision,
    )
    config.grad_accumulation_steps = grad_accumulation_steps
    
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
    
    # Train with dynamic learning
    trainer = DynamicTrainer(teacher, student, config, train_data)
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
        lambda_weight=0.5,
        temperature=1.0,
        use_dynamic_lambda=True,
        use_lr_scheduling=True,
        use_mixed_precision=True,
        grad_accumulation_steps=2,
    )
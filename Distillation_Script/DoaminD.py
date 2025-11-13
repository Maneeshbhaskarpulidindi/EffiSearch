#!/usr/bin/env python3
"""
Knowledge Distillation for ICT Models with Hybrid Loss & Dynamic Learning
FIXED: All helper functions and classes (including model loaders) are now fully defined 
       in the script to prevent "NameError: name 'X' is not defined".
"""

import os
import json
import torch
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

# Required for data loading
from datasets import load_dataset 
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, InputExample, models
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration & Domain Definition
# ============================================================
DOMAIN_LANGUAGES = [
    "Arabic", "Bengali", "English", "Finnish", "Indonesian", 
    "Japanese", "Korean", "Russian", "Swahili", "Telugu", "Thai"
]
DOMAIN_ABBREVIATIONS = ['ar', 'bn', 'en', 'fi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th']


@dataclass
class DistillationConfig:
    teacher_model_path: str
    student_model_name: str = "distilbert-base-multilingual-cased"
    num_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 2e-5
    output_dir: str = "./distilled_models"
    model_name: str = "ict_distilled_ddk"
    num_samples: int = 50000
    
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
    
    # DDK Integration Parameters (NEW)
    use_ddk_weighting: bool = True
    ddk_update_interval: int = 500
    ddk_weight_max: float = 1.5
    ddk_weight_min: float = 0.7


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
            
            if avg_distill > avg_retrieval * 1.5:
                self.lambda_val = min(self.max_lambda, self.lambda_val + 0.02)
            elif avg_retrieval > avg_distill * 1.5:
                self.lambda_val = max(self.min_lambda, self.lambda_val - 0.02)
        
        return self.lambda_val

# ============================================================
# Hybrid Loss
# ============================================================
class HybridLoss(nn.Module):
    """Hybrid Loss with proper scaling and DDK weighting."""
    def __init__(self, lambda_weight=0.5, temperature=1.0, use_dynamic_temp=True):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.base_temperature = temperature
        self.use_dynamic_temp = use_dynamic_temp
        self.temp_history = []
        self.ddk_weight = 1.0
        logger.info(f"HybridLoss initialized: λ={lambda_weight}, τ={temperature}")
    
    def normalize_embeddings(self, embeddings):
        return F.normalize(embeddings, p=2, dim=1)
    
    def distillation_loss(self, student_emb, teacher_emb):
        student_norm = self.normalize_embeddings(student_emb)
        teacher_norm = self.normalize_embeddings(teacher_emb)
        cos_sim = F.cosine_similarity(student_norm, teacher_norm, dim=1)
        loss = (1 - cos_sim).mean()
        return loss
    
    def retrieval_loss(self, query_emb, passage_emb):
        batch_size = query_emb.shape[0]
        embedding_dim = query_emb.shape[1]
        
        query_norm = self.normalize_embeddings(query_emb)
        passage_norm = self.normalize_embeddings(passage_emb)
        
        sim_matrix = torch.mm(query_norm, passage_norm.t())
        
        scale = np.sqrt(embedding_dim)
        sim_matrix = sim_matrix * scale / self.temperature
        
        labels = torch.arange(batch_size, device=query_emb.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def update_temperature(self, retrieval_loss):
        if self.use_dynamic_temp:
            self.temp_history.append(retrieval_loss)
            
            if len(self.temp_history) >= 3:
                recent_loss = np.mean(self.temp_history[-3:])
                if recent_loss > 5.0:
                    self.temperature = min(2.0, self.temperature + 0.05)
                elif recent_loss < 3.5:
                    self.temperature = max(0.5, self.temperature - 0.05)
    
    def set_ddk_weight(self, weight: float):
        self.ddk_weight = weight
    
    def forward(self, student_query_emb, student_passage_emb, 
                teacher_query_emb, teacher_passage_emb, lambda_weight=None):
        if lambda_weight is not None:
            self.lambda_weight = lambda_weight
        
        distill_loss_q = self.distillation_loss(student_query_emb, teacher_query_emb)
        distill_loss_p = self.distillation_loss(student_passage_emb, teacher_passage_emb)
        distill_loss = (distill_loss_q + distill_loss_p) / 2
        
        retrieval_loss = self.retrieval_loss(student_query_emb, student_passage_emb)
        self.update_temperature(retrieval_loss.item())
        
        hybrid_loss = (self.lambda_weight * distill_loss + 
                      (1 - self.lambda_weight) * retrieval_loss)
        
        hybrid_loss = hybrid_loss * self.ddk_weight

        return {
            'total': hybrid_loss,
            'distillation': distill_loss.item(),
            'retrieval': retrieval_loss.item(),
            'ddk_weight': self.ddk_weight
        }

# ============================================================
# DDK Domain Evaluation
# ============================================================
class DDKEvaluator:
    """Evaluates the Student-Teacher discrepancy (r) for each domain (language)"""
    def __init__(self, languages: List[str], max_weight: float, min_weight: float):
        self.languages = languages
        self.teacher_loss_history = {lang: [] for lang in languages}
        self.student_loss_history = {lang: [] for lang in languages}
        self.current_ddk_weights = {lang: 1.0 for lang in languages}
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.window_size = 5

    def update_history(self, language: str, teacher_loss: float, student_loss: float):
        if language in self.languages:
            self.teacher_loss_history[language].append(teacher_loss)
            self.student_loss_history[language].append(student_loss)
            
            if len(self.teacher_loss_history[language]) > self.window_size:
                self.teacher_loss_history[language].pop(0)
                self.student_loss_history[language].pop(0)

    def calculate_and_update_weights(self):
        ddk_weights = {}
        for lang in self.languages:
            if len(self.student_loss_history[lang]) > 0:
                avg_l_s = np.mean(self.student_loss_history[lang])
                avg_l_t = np.mean(self.teacher_loss_history[lang])
                
                if avg_l_t < 0.1: avg_l_t = 0.1
                
                ratio = avg_l_s / avg_l_t
                weight = 1.0 + (ratio - 1.0) * 0.2
                weight = max(self.min_weight, min(self.max_weight, weight))
                ddk_weights[lang] = weight
            else:
                ddk_weights[lang] = 1.0
        
        self.current_ddk_weights = ddk_weights
        return ddk_weights

# ============================================================
# Data Loading and Collation
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
    languages = [example.label for example in batch] 
    return queries, passages, languages

def load_data_from_huggingface(
    num_samples: int,
    languages: List[str] = DOMAIN_ABBREVIATIONS
) -> List[InputExample]:
    """Downloads Mr. TyDi training data from Hugging Face and assigns language labels."""
    logger.info("Starting Mr. TyDi data download from Hugging Face...")
    all_examples = []
    
    max_per_lang = num_samples // len(DOMAIN_ABBREVIATIONS) + 1

    for abbr, full_name in zip(DOMAIN_ABBREVIATIONS, DOMAIN_LANGUAGES):
        if abbr in languages:
            try:
                # FIX: Added legacy=True to load older dataset scripts
                dataset = load_dataset('castorini/mr-tydi', abbr, split='train', legacy=True)
                
                logger.info(f"Loaded {abbr} ({full_name}) with {len(dataset)} training examples.")
                
                count = 0
                for item in tqdm(dataset, desc=f"Processing {full_name} data"):
                    query = item['query']
                    
                    if item['positive_passages']:
                        positive_passage = item['positive_passages'][0]['text'] 
                        
                        example = InputExample(texts=[query, positive_passage])
                        example.label = full_name
                        all_examples.append(example)
                        count += 1
                        
                    if count >= max_per_lang:
                        break

            except Exception as e:
                logger.error(f"Failed to load or process data for {full_name} ({abbr}): {e}")

    logger.info(f"Finished loading data. Total positive examples loaded: {len(all_examples)}")
    np.random.shuffle(all_examples) 
    return all_examples

# ============================================================
# Model Loading (RESTORED)
# ============================================================
def load_pretrained_encoder(model_path: str) -> SentenceTransformer:
    """Loads the teacher SentenceTransformer model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    try:
        model = SentenceTransformer(str(model_path))
        logger.info("✓ Teacher model loaded successfully")
        return model
    except Exception as e:
        raise ValueError(f"Failed to load teacher model: {e}")

def create_student_model(student_name: str) -> SentenceTransformer:
    """Creates the student SentenceTransformer model with required pooling and normalization."""
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
# Dynamic Trainer
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
        for param_group in self.optimizer.param_groups:
            param_group['base_lr'] = config.learning_rate
        
        if config.use_lr_scheduling:
            total_steps = len(self.train_loader) * config.num_epochs
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                warmup_steps=config.warmup_steps,
                first_cycle_steps=total_steps
            )
        
        if config.use_dynamic_lambda:
            self.dynamic_lambda = DynamicLambda(initial_lambda=config.lambda_weight)
        
        if config.use_mixed_precision:
            self.scaler = GradScaler()
        
        self.ddk_evaluator = None
        if config.use_ddk_weighting:
            self.ddk_evaluator = DDKEvaluator(
                languages=DOMAIN_LANGUAGES,
                max_weight=config.ddk_weight_max,
                min_weight=config.ddk_weight_min
            )
            logger.info("DDK Weighting enabled.")

        logger.info(f"Gradient accumulation: {config.grad_accumulation_steps} steps")
    
    def train(self):
        logger.info("Starting training with dynamic learning...")
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss, epoch_distill_loss, epoch_retrieval_loss = 0, 0, 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            accumulated_loss = 0
            
            for batch_idx, (queries, passages, languages) in enumerate(pbar):
                
                # 1. Teacher Embeddings
                with torch.no_grad():
                    teacher_query_emb = self.teacher.encode(queries, convert_to_tensor=True, device=self.device, show_progress_bar=False).clone()
                    teacher_passage_emb = self.teacher.encode(passages, convert_to_tensor=True, device=self.device, show_progress_bar=False).clone()
                    teacher_query_emb = F.normalize(teacher_query_emb, p=2, dim=1)
                    teacher_passage_emb = F.normalize(teacher_passage_emb, p=2, dim=1)
                    teacher_retrieval_loss = self.criterion.retrieval_loss(teacher_query_emb, teacher_passage_emb).item()

                # 2. Student Embeddings and Loss Calculation
                current_lambda, ddk_weight = self.config.lambda_weight, 1.0
                
                with autocast() if self.config.use_mixed_precision else torch.enable_grad():
                    student_query_emb = self.student.encode(queries, convert_to_tensor=True, device=self.device, show_progress_bar=False)
                    student_passage_emb = self.student.encode(passages, convert_to_tensor=True, device=self.device, show_progress_bar=False)
                    
                    if self.config.use_dynamic_lambda: current_lambda = self.dynamic_lambda.lambda_val
                    
                    if self.config.use_ddk_weighting and self.ddk_evaluator:
                        most_frequent_lang = max(set(languages), key=languages.count)
                        ddk_weight = self.ddk_evaluator.current_ddk_weights.get(most_frequent_lang, 1.0)
                        self.criterion.set_ddk_weight(ddk_weight)

                    loss_dict = self.criterion(
                        student_query_emb, student_passage_emb,
                        teacher_query_emb, teacher_passage_emb,
                        lambda_weight=current_lambda
                    )
                    loss = loss_dict['total'] / self.config.grad_accumulation_steps
                
                # 3. DDK History Update
                if self.config.use_ddk_weighting and self.ddk_evaluator:
                    self.ddk_evaluator.update_history(most_frequent_lang, teacher_retrieval_loss, loss_dict['retrieval'])

                # 4. Backward pass
                if self.config.use_mixed_precision: self.scaler.scale(loss).backward()
                else: loss.backward()
                accumulated_loss += loss.item()
                
                # 5. Gradient accumulation step
                if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                    global_step += 1
                    
                    if self.config.use_mixed_precision: self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    
                    if self.config.use_mixed_precision: self.scaler.step(self.optimizer); self.scaler.update()
                    else: self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    if self.config.use_lr_scheduling: self.scheduler.step()
                    if self.config.use_dynamic_lambda: self.dynamic_lambda.update(loss_dict['distillation'], loss_dict['retrieval'])
                    
                    if self.config.use_ddk_weighting and self.ddk_evaluator and global_step % self.config.ddk_update_interval == 0: self.ddk_evaluator.calculate_and_update_weights()

                    epoch_loss += accumulated_loss; epoch_distill_loss += loss_dict['distillation']; epoch_retrieval_loss += loss_dict['retrieval']
                    pbar.set_postfix({"loss": f"{accumulated_loss:.4f}", "distill": f"{loss_dict['distillation']:.4f}", "retrieval": f"{loss_dict['retrieval']:.4f}", "λ": f"{current_lambda:.3f}", "τ": f"{self.criterion.temperature:.2f}", "DDK": f"{ddk_weight:.2f}"})
                    accumulated_loss = 0
            
            n_batches = len(self.train_loader) // self.config.grad_accumulation_steps
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_distill = epoch_distill_loss / max(n_batches, 1)
            avg_retrieval = epoch_retrieval_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}: loss={avg_loss:.4f} | distill={avg_distill:.4f} | retrieval={avg_retrieval:.4f}")
            self.save_checkpoint(f"epoch_{epoch+1}")
        self.save_checkpoint("final")
    
    def save_checkpoint(self, name: str):
        output_dir = Path(self.config.output_dir) / self.config.model_name / name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.student.save(str(output_dir))
        logger.info(f"✓ Checkpoint saved to {output_dir}")
        
# ============================================================
# Main Execution
# ============================================================
def distill_ict_model(
    teacher_path: str,
    num_epochs: int = 3,
    batch_size: int = 64,
    num_samples: int = 50000,
    grad_accumulation_steps: int = 2,
    use_ddk_weighting: bool = True,
):
    logger.info("\n" + "="*80)
    logger.info("ICT KNOWLEDGE DISTILLATION with DDK Proxy (Mr. TyDi Languages)")
    logger.info("="*80 + "\n")
    
    if not teacher_path:
        raise ValueError("teacher_path is required")
    
    config = DistillationConfig(
        teacher_model_path=teacher_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_samples=num_samples,
        grad_accumulation_steps=grad_accumulation_steps,
        use_ddk_weighting=use_ddk_weighting,
    )
    
    # 1. Load Teacher (Fix for NameError)
    teacher = load_pretrained_encoder(teacher_path)
    # 2. Create Student (Fix for NameError)
    student = create_student_model(config.student_model_name)
    
    # 3. Load Data
    train_data = load_data_from_huggingface(num_samples=config.num_samples)
    
    if not train_data:
        raise ValueError("No training data loaded!")
    
    # 4. Train
    trainer = DynamicTrainer(teacher, student, config, train_data)
    trainer.train()
    
    logger.info("\n" + "="*80)
    logger.info("✓ DISTILLATION COMPLETE")
    logger.info("="*80 + "\n")
    
    return student

if __name__ == "__main__":
    # >>>>>>>>>>>>>>>>>>>>>>>>>>
    # !!! IMPORTANT: VERIFY THIS PATH !!!
    # >>>>>>>>>>>>>>>>>>>>>>>>>>
    TEACHER_MODEL_PATH = "/home/maneesh/Desktop/IR/negative-sampling-repro/Repo/Distillation_Script/ict_p_mrtydi_finetuned/final/encoder_complete"

    try:
        student_model = distill_ict_model(
            teacher_path=TEACHER_MODEL_PATH,
            num_epochs=3,
            batch_size=64,
            num_samples=50000,
            grad_accumulation_steps=2,
            use_ddk_weighting=True,
        )
    except FileNotFoundError as e:
        logger.error(f"FATAL ERROR: Teacher model not found. Please verify the path: {TEACHER_MODEL_PATH}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}")
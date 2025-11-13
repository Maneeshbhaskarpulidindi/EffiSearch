import os
import math
import json
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import logging
import re

# ============================================================
# ✅ CONFIGURATION
# ============================================================
DATA_PATH = "/home/maneesh/Desktop/data/msmarco/msmarco-train.tsv"  # Changed: need passages only
MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_DIR = "/home/maneesh/Desktop/IR/negative-sampling-repro/trained_models_Negative/pretrained_ict_p/"

BATCH_SIZE = 1028
MAX_LEN = 256
QUERY_MAX_LEN = 64  # Shorter for query sentences
LR = 2e-5
EPOCHS = 10
FP16 = True
NUM_WORKERS = 8
GRAD_ACCUM_STEPS = 1
TEMPERATURE = 0.05  # For contrastive loss

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ✅ DDP SETUP
# ============================================================
def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1
    return local_rank, world_size

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# ============================================================
# ✅ LOGGING
# ============================================================
def get_logger(rank):
    logger = logging.getLogger(f"rank_{rank}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][GPU%(rank)d] %(message)s".replace("%(rank)d", str(rank)))
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    return logger

# ============================================================
# ✅ TEXT PROCESSING FOR ICT
# ============================================================
def split_into_sentences(text):
    """Split text into sentences using basic regex"""
    # Simple sentence splitting (can be improved with NLTK/spaCy)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]  # Filter very short

def sample_query_from_passage(passage):
    """
    ICT Core Logic: Sample a sentence as query, rest as positive passage
    """
    sentences = split_into_sentences(passage)
    
    if len(sentences) < 2:
        # If passage is too short, use first half as query, second half as passage
        mid = len(passage) // 2
        query = passage[:mid].strip()
        positive = passage[mid:].strip()
        if len(query) < 10:
            query = passage[:100]  # Fallback
            positive = passage
    else:
        # Randomly sample one sentence as query
        query_idx = random.randint(0, len(sentences) - 1)
        query = sentences[query_idx]
        
        # Rest becomes positive passage
        positive_sentences = [s for i, s in enumerate(sentences) if i != query_idx]
        positive = ' '.join(positive_sentences)
    
    return query, positive

# ============================================================
# ✅ ICT DATASET (Self-Supervised)
# ============================================================
class ICTDataset(Dataset):
    """
    True ICT Dataset: Self-supervised passage-to-query generation
    """
    def __init__(self, passages, tokenizer, query_max_len=64, passage_max_len=256):
        self.passages = passages
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
        
    def __len__(self):
        return len(self.passages)
    
    def __getitem__(self, idx):
        passage = self.passages[idx]
        
        # ICT Core: Sample query from passage
        query, positive = sample_query_from_passage(passage)
        
        # Random negative from different passage
        neg_idx = random.randint(0, len(self.passages) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.passages) - 1)
        negative = self.passages[neg_idx]
        
        # Tokenize
        encoded_q = self.tokenizer(
            query, 
            truncation=True, 
            padding="max_length", 
            max_length=self.query_max_len,
            return_tensors=None
        )
        encoded_pos = self.tokenizer(
            positive, 
            truncation=True, 
            padding="max_length", 
            max_length=self.passage_max_len,
            return_tensors=None
        )
        encoded_neg = self.tokenizer(
            negative, 
            truncation=True, 
            padding="max_length", 
            max_length=self.passage_max_len,
            return_tensors=None
        )
        
        return {
            "q_input_ids": torch.tensor(encoded_q["input_ids"], dtype=torch.long),
            "q_attention_mask": torch.tensor(encoded_q["attention_mask"], dtype=torch.long),
            "pos_input_ids": torch.tensor(encoded_pos["input_ids"], dtype=torch.long),
            "pos_attention_mask": torch.tensor(encoded_pos["attention_mask"], dtype=torch.long),
            "neg_input_ids": torch.tensor(encoded_neg["input_ids"], dtype=torch.long),
            "neg_attention_mask": torch.tensor(encoded_neg["attention_mask"], dtype=torch.long),
        }

# ============================================================
# ✅ MODEL (Shared Weight Initialization)
# ============================================================
class ICTDualEncoder(nn.Module):
    """
    ICT Dual Encoder with option for shared or separate encoders
    """
    def __init__(self, model_name, share_weights=False):
        super().__init__()
        self.share_weights = share_weights
        
        if share_weights:
            # Both encoders share the same weights (true ICT style)
            self.encoder = AutoModel.from_pretrained(model_name)
            self.passage_encoder = self.encoder
            self.question_encoder = self.encoder
        else:
            # Separate encoders (initialized from same checkpoint)
            self.passage_encoder = AutoModel.from_pretrained(model_name)
            self.question_encoder = AutoModel.from_pretrained(model_name)
    
    def forward_encoder(self, input_ids, attention_mask, encoder_type="passage"):
        if self.share_weights:
            encoder = self.encoder
        else:
            encoder = self.passage_encoder if encoder_type == "passage" else self.question_encoder
        
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(cls_output, p=2, dim=1)

# ============================================================
# ✅ CONTRASTIVE LOSS (NLL-based)
# ============================================================
def compute_contrastive_loss(q_emb, pos_emb, neg_emb, temperature=0.05):
    """
    Compute contrastive loss with in-batch negatives
    """
    batch_size = q_emb.size(0)
    
    # Compute similarities
    pos_sim = torch.sum(q_emb * pos_emb, dim=1) / temperature  # [batch_size]
    neg_sim = torch.sum(q_emb * neg_emb, dim=1) / temperature  # [batch_size]
    
    # Also use all other positives in batch as negatives (in-batch negatives)
    all_pos_sim = torch.matmul(q_emb, pos_emb.T) / temperature  # [batch_size, batch_size]
    
    # Concatenate positive and negative similarities
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1), all_pos_sim], dim=1)
    
    # Target is always the first one (the actual positive)
    labels = torch.zeros(batch_size, dtype=torch.long, device=q_emb.device)
    
    # Cross-entropy loss
    loss = nn.functional.cross_entropy(logits, labels)
    return loss

# ============================================================
# ✅ EVALUATION
# ============================================================
def evaluate(model, dataset, tokenizer, device, top_k=100, num_samples=1000):
    """
    Evaluate on a subset of data with detailed progress
    """
    model.eval()
    
    # Sample subset for faster evaluation
    eval_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
    
    loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
    
    q_embs, p_embs = [], []
    
    print(f"   Encoding {len(eval_dataset)} query-passage pairs...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="   Evaluation", leave=False, ncols=80):
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            p_ids = batch["pos_input_ids"].to(device)
            p_mask = batch["pos_attention_mask"].to(device)
            
            if hasattr(model, "module"):
                q_emb = model.module.forward_encoder(q_ids, q_mask, "query")
                p_emb = model.module.forward_encoder(p_ids, p_mask, "passage")
            else:
                q_emb = model.forward_encoder(q_ids, q_mask, "query")
                p_emb = model.forward_encoder(p_ids, p_mask, "passage")
            
            q_embs.append(q_emb.cpu())
            p_embs.append(p_emb.cpu())
    
    print(f"   Computing similarity matrix...")
    q_embs = torch.cat(q_embs)
    p_embs = torch.cat(p_embs)
    sims = torch.matmul(q_embs, p_embs.T).numpy()
    
    print(f"   Calculating metrics...")
    ranks = np.argsort(-sims, axis=1)
    rr, recall = [], []
    
    for i in range(len(ranks)):
        rank = np.where(ranks[i] == i)[0]
        if len(rank) > 0:
            rr.append(1 / (rank[0] + 1))
            recall.append(1 if rank[0] < top_k else 0)
        else:
            rr.append(0)
            recall.append(0)
    
    metrics = {
        "mrr": np.mean(rr),
        "recall_100": np.mean(recall),
    }
    return metrics

# ============================================================
# ✅ MAIN TRAIN LOOP
# ============================================================
def main():
    local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    logger = get_logger(local_rank)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # ========================================================
    # ✅ Load passages (not triplets!)
    # ========================================================
    if local_rank == 0:
        print("\n" + "="*60)
        print(f"📂 Loading data from {DATA_PATH}")
        print("="*60)
        import sys
        sys.stdout.flush()
    
    # Load passages - adjust based on your data format
    try:
        # Try loading as passage file first (pid \t passage)
        df = pd.read_csv(DATA_PATH, sep="\t", header=None, on_bad_lines="skip", nrows=2)
        
        # Check if it's triplet format (query, pos, neg) or passage format
        if df.shape[1] >= 3:
            # It's triplet format - extract unique passages
            if local_rank == 0:
                print("⚠️  Detected triplet format (query, pos, neg)")
                print("📝 Extracting unique passages for ICT pretraining...")
                sys.stdout.flush()
            
            df_full = pd.read_csv(DATA_PATH, sep="\t", header=None, on_bad_lines="skip", usecols=[1, 2])
            # Combine positive and negative passages
            pos_passages = df_full.iloc[:, 0].dropna().astype(str).tolist()
            neg_passages = df_full.iloc[:, 1].dropna().astype(str).tolist()
            all_passages = pos_passages + neg_passages
            
            # Remove duplicates while preserving order
            seen = set()
            passages = []
            for p in all_passages:
                if p not in seen and len(p) > 20:  # Filter very short passages
                    seen.add(p)
                    passages.append(p)
        else:
            # It's passage format - use directly
            if local_rank == 0:
                print("✅ Detected passage format")
                sys.stdout.flush()
            df_full = pd.read_csv(DATA_PATH, sep="\t", header=None, on_bad_lines="skip")
            passages = df_full.iloc[:, 1 if df_full.shape[1] >= 2 else 0].dropna().astype(str).tolist()
            passages = [p for p in passages if len(p) > 20]
    except Exception as e:
        if local_rank == 0:
            print(f"❌ Error loading data: {e}")
            sys.stdout.flush()
        raise
    
    if local_rank == 0:
        print(f"📊 Loaded {len(passages)} unique passages")
        print(f"📝 Sample passage (first 150 chars):")
        print(f"    '{passages[0][:150]}...'")
        print("="*60)
        print("")
        sys.stdout.flush()
    
    # ========================================================
    # ✅ Create ICT Dataset
    # ========================================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = ICTDataset(passages, tokenizer, query_max_len=QUERY_MAX_LEN, passage_max_len=MAX_LEN)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE // world_size, 
        sampler=sampler,
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        drop_last=True
    )
    
    # ========================================================
    # ✅ Initialize Model
    # ========================================================
    model = ICTDualEncoder(MODEL_NAME, share_weights=False).to(device)
    model.passage_encoder.gradient_checkpointing_enable()
    model.question_encoder.gradient_checkpointing_enable()
    
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank
        )
    
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = (len(loader) * EPOCHS) // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, 5000, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=FP16)
    
    train_losses, eval_scores = [], []
    
    # ========================================================
    # ✅ Initial Evaluation (Before Training)
    # ========================================================
    if local_rank == 0:
        print("\n" + "="*60)
        print("🔍 INITIAL EVALUATION (EPOCH 0 - BEFORE TRAINING)")
        print("="*60)
        initial_metrics = evaluate(model, dataset, tokenizer, device)
        eval_scores.append(initial_metrics)
        print(f"📊 MRR: {initial_metrics['mrr']:.4f}")
        print(f"📊 Recall@100: {initial_metrics['recall_100']:.4f}")
        print("="*60)
        print("")
        import sys
        sys.stdout.flush()
    
    # ========================================================
    # ✅ TRAINING LOOP
    # ========================================================
    if local_rank == 0:
        print("\n" + "="*60)
        print("🔥 STARTING ICT PRETRAINING")
        print("="*60)
        print(f"📊 Number of GPUs: {world_size}")
        print(f"📊 Total training steps: {total_steps}")
        print(f"📊 Batch size per GPU: {BATCH_SIZE // world_size}")
        print(f"📊 Global batch size: {BATCH_SIZE}")
        print(f"📊 Learning rate: {LR}")
        print(f"📊 Epochs: {EPOCHS}")
        print("="*60)
        print("")
        import sys
        sys.stdout.flush()
    
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}") if local_rank == 0 else loader
        
        for step, batch in enumerate(progress):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=FP16):
                # Forward pass
                if hasattr(model, "module"):
                    q_emb = model.module.forward_encoder(batch["q_input_ids"], batch["q_attention_mask"], "query")
                    pos_emb = model.module.forward_encoder(batch["pos_input_ids"], batch["pos_attention_mask"], "passage")
                    neg_emb = model.module.forward_encoder(batch["neg_input_ids"], batch["neg_attention_mask"], "passage")
                else:
                    q_emb = model.forward_encoder(batch["q_input_ids"], batch["q_attention_mask"], "query")
                    pos_emb = model.forward_encoder(batch["pos_input_ids"], batch["pos_attention_mask"], "passage")
                    neg_emb = model.forward_encoder(batch["neg_input_ids"], batch["neg_attention_mask"], "passage")
                
                # Compute contrastive loss
                loss = compute_contrastive_loss(q_emb, pos_emb, neg_emb, TEMPERATURE) / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            if local_rank == 0:
                progress.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"})
        
        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        
        if local_rank == 0:
            import sys
            print("\n" + "="*60)
            print(f"✅ EPOCH {epoch+1}/{EPOCHS} COMPLETED")
            print("="*60)
            print(f"📉 Average Training Loss: {avg_loss:.4f}")
            print("")
            print("🔍 Running Evaluation...")
            sys.stdout.flush()
            
            # Evaluate
            metrics = evaluate(model, dataset, tokenizer, device)
            eval_scores.append(metrics)
            
            print("─"*60)
            print("📊 EVALUATION RESULTS:")
            print(f"   • MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
            print(f"   • Recall@100: {metrics['recall_100']:.4f}")
            print("─"*60)
            
            # Show improvement from initial
            if len(eval_scores) > 1:
                mrr_improvement = (metrics['mrr'] - eval_scores[0]['mrr']) * 100
                recall_improvement = (metrics['recall_100'] - eval_scores[0]['recall_100']) * 100
                print("📈 Improvement from Initial:")
                print(f"   • MRR: {mrr_improvement:+.2f}%")
                print(f"   • Recall@100: {recall_improvement:+.2f}%")
                print("─"*60)
            
            # Save checkpoint
            save_path = os.path.join(OUTPUT_DIR, f"ict_epoch{epoch+1}.pt")
            torch.save(
                model.module.state_dict() if hasattr(model, "module") else model.state_dict(), 
                save_path
            )
            print(f"💾 Model checkpoint saved: {save_path}")
            print("="*60)
            print("")
            sys.stdout.flush()
    
    # ========================================================
    # ✅ Final saving & plotting
    # ========================================================
    if local_rank == 0:
        import sys
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED!")
        print("="*60)
        
        final_path = os.path.join(OUTPUT_DIR, "ict_final.pt")
        torch.save(
            model.module.state_dict() if hasattr(model, "module") else model.state_dict(), 
            final_path
        )
        print(f"💾 Final model saved: {final_path}")
        
        # Print final summary
        print("")
        print("📊 TRAINING SUMMARY:")
        print(f"   • Initial Loss: {train_losses[0]:.4f}")
        print(f"   • Final Loss: {train_losses[-1]:.4f}")
        print(f"   • Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
        print("")
        print(f"   • Initial MRR: {eval_scores[0]['mrr']:.4f}")
        print(f"   • Final MRR: {eval_scores[-1]['mrr']:.4f}")
        print(f"   • MRR Improvement: {((eval_scores[-1]['mrr'] - eval_scores[0]['mrr']) * 100):+.2f}%")
        print("")
        print(f"   • Initial Recall@100: {eval_scores[0]['recall_100']:.4f}")
        print(f"   • Final Recall@100: {eval_scores[-1]['recall_100']:.4f}")
        print(f"   • Recall@100 Improvement: {((eval_scores[-1]['recall_100'] - eval_scores[0]['recall_100']) * 100):+.2f}%")
        print("="*60)
        sys.stdout.flush()
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("ICT Training Loss")
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))
        plt.close()
        
        # Plot evaluation metrics
        if eval_scores:
            epochs = list(range(0, len(eval_scores)))
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, [m["mrr"] for m in eval_scores], marker='o', label="MRR")
            plt.plot(epochs, [m["recall_100"] for m in eval_scores], marker='s', label="Recall@100")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("ICT Evaluation Metrics")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_metrics.png"))
            plt.close()
        
        logger.info("✅ Training complete!")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()
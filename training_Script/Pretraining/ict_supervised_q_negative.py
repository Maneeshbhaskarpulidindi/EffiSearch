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
import matplotlib.pyplot as plt
import logging

# ============================================================
# ✅ CONFIGURATION
# ============================================================
DATA_PATH = "/home/maneesh/Desktop/data/msmarco/msmarco-train.tsv"  # QUERIES ONLY
MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_DIR = "/home/maneesh/Desktop/IR/negative-sampling-repro/trained_models_Negative/pretrained_ict_q/"

BATCH_SIZE = 1028
QUERY_MAX_LEN = 64
PASSAGE_MAX_LEN = 256  # Generated pseudo-passages will be longer
LR = 2e-5
EPOCHS = 10
FP16 = True
NUM_WORKERS = 8
GRAD_ACCUM_STEPS = 1
TEMPERATURE = 0.05

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
# ✅ QUERY EXPANSION FOR ICT (REVERSE LOGIC)
# ============================================================
def expand_query_to_passage(query, expansion_factor=4):
    """
    ICT-Query Core Logic: Expand a query into a pseudo-passage
    
    Strategy: Since queries are short, we need to CREATE passage-like text
    We'll use multiple expansion techniques:
    1. Repeat key terms with variations
    2. Add context words
    3. Create question-answer style text
    """
    
    # Split query into words
    words = query.strip().split()
    
    if len(words) < 2:
        # Very short query - just repeat and add filler
        expanded = f"{query}. This is about {query}. Information regarding {query}."
        return query, expanded
    
    # Strategy 1: Keep original query, expand the rest
    original_query = query
    
    # Strategy 2: Create pseudo-passage by expansion
    expansions = []
    
    # Add the original query as a sentence
    expansions.append(query + ".")
    
    # Add rephrased versions
    if len(words) >= 3:
        # Rearrange words to create variation
        shuffled = words.copy()
        random.shuffle(shuffled)
        expansions.append(" ".join(shuffled) + ".")
    
    # Add contextual sentences (simulating passage structure)
    key_terms = random.sample(words, min(2, len(words)))
    expansions.append(f"This document discusses {' and '.join(key_terms)}.")
    expansions.append(f"Information about {words[0]} is presented here.")
    
    # Add more filler to make it passage-like
    if len(words) >= 2:
        expansions.append(f"The topic covers {words[-1]} and related concepts.")
    
    # Randomly sample expansion_factor sentences
    num_sentences = min(expansion_factor, len(expansions))
    selected = random.sample(expansions, num_sentences)
    
    pseudo_passage = " ".join(selected)
    
    return original_query, pseudo_passage

# ============================================================
# ✅ ICT-QUERY DATASET (Query → Pseudo-Passage)
# ============================================================
class ICTQueryDataset(Dataset):
    """
    Reverse ICT: Start from queries, generate pseudo-passages
    Task: Given a query, can the model retrieve the expanded pseudo-passage?
    """
    def __init__(self, queries, tokenizer, query_max_len=64, passage_max_len=256):
        self.queries = queries
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_text = self.queries[idx]
        
        # ICT-Query Core: Expand query to pseudo-passage
        query, positive_passage = expand_query_to_passage(query_text)
        
        # Random negative: expand a different query
        neg_idx = random.randint(0, len(self.queries) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.queries) - 1)
        _, negative_passage = expand_query_to_passage(self.queries[neg_idx])
        
        # Tokenize
        encoded_q = self.tokenizer(
            query, 
            truncation=True, 
            padding="max_length", 
            max_length=self.query_max_len,
            return_tensors=None
        )
        encoded_pos = self.tokenizer(
            positive_passage, 
            truncation=True, 
            padding="max_length", 
            max_length=self.passage_max_len,
            return_tensors=None
        )
        encoded_neg = self.tokenizer(
            negative_passage, 
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
# ✅ MODEL (Same dual encoder architecture)
# ============================================================
class ICTDualEncoder(nn.Module):
    def __init__(self, model_name, share_weights=False):
        super().__init__()
        self.share_weights = share_weights
        
        if share_weights:
            self.encoder = AutoModel.from_pretrained(model_name)
            self.passage_encoder = self.encoder
            self.question_encoder = self.encoder
        else:
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
# ✅ CONTRASTIVE LOSS (Same as ICT-Passage)
# ============================================================
def compute_contrastive_loss(q_emb, pos_emb, neg_emb, temperature=0.05):
    batch_size = q_emb.size(0)
    
    pos_sim = torch.sum(q_emb * pos_emb, dim=1) / temperature
    neg_sim = torch.sum(q_emb * neg_emb, dim=1) / temperature
    
    all_pos_sim = torch.matmul(q_emb, pos_emb.T) / temperature
    
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1), all_pos_sim], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=q_emb.device)
    
    loss = nn.functional.cross_entropy(logits, labels)
    return loss

# ============================================================
# ✅ EVALUATION
# ============================================================
def evaluate(model, dataset, tokenizer, device, top_k=100, num_samples=1000):
    model.eval()
    
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
    # ✅ Load QUERIES (not passages!)
    # ========================================================
    if local_rank == 0:
        print("\n" + "="*60)
        print(f"📂 Loading QUERIES from {DATA_PATH}")
        print("="*60)
        import sys
        sys.stdout.flush()
    
    try:
        # Load queries
        df = pd.read_csv(DATA_PATH, sep="\t", header=None, on_bad_lines="skip")
        
        # Queries are typically in format: qid \t query
        if df.shape[1] >= 2:
            queries = df.iloc[:, 1].dropna().astype(str).tolist()
        else:
            queries = df.iloc[:, 0].dropna().astype(str).tolist()
        
        # Filter very short queries
        queries = [q for q in queries if len(q.split()) >= 2]
        
    except Exception as e:
        if local_rank == 0:
            print(f"❌ Error loading data: {e}")
            print("⚠️  Note: Expected format is TSV with queries")
            print("⚠️  If you have triplet data, extract queries first!")
            sys.stdout.flush()
        raise
    
    if local_rank == 0:
        print(f"📊 Loaded {len(queries)} queries")
        print(f"📝 Sample query:")
        print(f"    '{queries[0]}'")
        print(f"📝 Sample expansion:")
        q, p = expand_query_to_passage(queries[0])
        print(f"    Query: '{q}'")
        print(f"    Pseudo-passage: '{p}'")
        print("="*60)
        print("")
        sys.stdout.flush()
    
    # ========================================================
    # ✅ Create ICT-Query Dataset
    # ========================================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = ICTQueryDataset(queries, tokenizer, query_max_len=QUERY_MAX_LEN, passage_max_len=PASSAGE_MAX_LEN)
    
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
    # ✅ Initial Evaluation
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
        print("🔥 STARTING ICT-QUERY PRETRAINING")
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
                if hasattr(model, "module"):
                    q_emb = model.module.forward_encoder(batch["q_input_ids"], batch["q_attention_mask"], "query")
                    pos_emb = model.module.forward_encoder(batch["pos_input_ids"], batch["pos_attention_mask"], "passage")
                    neg_emb = model.module.forward_encoder(batch["neg_input_ids"], batch["neg_attention_mask"], "passage")
                else:
                    q_emb = model.forward_encoder(batch["q_input_ids"], batch["q_attention_mask"], "query")
                    pos_emb = model.forward_encoder(batch["pos_input_ids"], batch["pos_attention_mask"], "passage")
                    neg_emb = model.forward_encoder(batch["neg_input_ids"], batch["neg_attention_mask"], "passage")
                
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
            
            metrics = evaluate(model, dataset, tokenizer, device)
            eval_scores.append(metrics)
            
            print("─"*60)
            print("📊 EVALUATION RESULTS:")
            print(f"   • MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
            print(f"   • Recall@100: {metrics['recall_100']:.4f}")
            print("─"*60)
            
            if len(eval_scores) > 1:
                mrr_improvement = (metrics['mrr'] - eval_scores[0]['mrr']) * 100
                recall_improvement = (metrics['recall_100'] - eval_scores[0]['recall_100']) * 100
                print("📈 Improvement from Initial:")
                print(f"   • MRR: {mrr_improvement:+.2f}%")
                print(f"   • Recall@100: {recall_improvement:+.2f}%")
                print("─"*60)
            
            save_path = os.path.join(OUTPUT_DIR, f"ict_query_epoch{epoch+1}.pt")
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
        
        final_path = os.path.join(OUTPUT_DIR, "ict_query_final.pt")
        torch.save(
            model.module.state_dict() if hasattr(model, "module") else model.state_dict(), 
            final_path
        )
        print(f"💾 Final model saved: {final_path}")
        
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
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("ICT-Query Training Loss")
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))
        plt.close()
        
        if eval_scores:
            epochs = list(range(0, len(eval_scores)))
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, [m["mrr"] for m in eval_scores], marker='o', label="MRR")
            plt.plot(epochs, [m["recall_100"] for m in eval_scores], marker='s', label="Recall@100")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("ICT-Query Evaluation Metrics")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_metrics.png"))
            plt.close()
        
        logger.info("✅ Training complete!")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()
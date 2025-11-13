#!/usr/bin/env python3
"""
DPR Fine-tuning - FAST & WORKING VERSION
Direct training without SimpleTransformers complications
"""

import logging
import torch
import pandas as pd
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np

# ==============================
# SETUP
# ==============================
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_PATH = Path("/home/maneesh/Desktop/IR/negative-sampling-repro/train_full.tsv")
OUTPUT_DIR = SCRIPT_DIR / "output" / "models" / "msmarco"
LOGS_DIR = SCRIPT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOGGING
# ==============================
log_file = LOGS_DIR / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logs saved to: {log_file}")

# ==============================
# CONFIG
# ==============================
CONFIG = {
    'model_name': 'bert-base-multilingual-cased',
    'max_seq_length': 256,
    'train_batch_size': 4,
    'num_train_epochs': 3,
    'learning_rate': 2e-5,
    'warmup_steps': 1000,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ==============================
# DATASET
# ==============================
class DPRDataset(Dataset):
    """DPR Dataset with query, positive, negative"""
    
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        query = str(row['query_text']).strip()
        pos = str(row['gold_passage']).strip()
        neg = str(row['hard_negative']).strip()
        
        # Tokenize
        q_tok = self.tokenizer(query, max_length=self.max_length, padding='max_length', 
                               truncation=True, return_tensors='pt')
        p_tok = self.tokenizer(pos, max_length=self.max_length, padding='max_length', 
                               truncation=True, return_tensors='pt')
        n_tok = self.tokenizer(neg, max_length=self.max_length, padding='max_length', 
                               truncation=True, return_tensors='pt')
        
        return {
            'q_ids': q_tok['input_ids'].squeeze(0),
            'q_mask': q_tok['attention_mask'].squeeze(0),
            'p_ids': p_tok['input_ids'].squeeze(0),
            'p_mask': p_tok['attention_mask'].squeeze(0),
            'n_ids': n_tok['input_ids'].squeeze(0),
            'n_mask': n_tok['attention_mask'].squeeze(0),
        }

# ==============================
# LOAD & PREP DATA
# ==============================
def prep_data(data_path):
    """Load and prepare data"""
    logger.info(f"\n{'='*60}")
    logger.info("LOADING & PREPARING DATA")
    logger.info(f"{'='*60}")
    
    if not Path(data_path).exists():
        logger.error(f"❌ File not found: {data_path}")
        sys.exit(1)
    
    logger.info(f"Loading: {data_path}")
    df = pd.read_csv(data_path, sep='\t')
    logger.info(f"✓ Loaded: {len(df)} rows")
    
    # Transform
    result_df = pd.DataFrame()
    result_df['query_text'] = df['query']
    result_df['gold_passage'] = df['positive_passages']
    result_df['hard_negative'] = df['negative_passages']
    
    # Clean
    result_df = result_df.fillna('')
    result_df = result_df[(result_df['query_text'].str.len() > 0) & 
                          (result_df['gold_passage'].str.len() > 0) & 
                          (result_df['hard_negative'].str.len() > 0)]
    result_df = result_df.drop_duplicates()
    
    logger.info(f"✓ After cleaning: {len(result_df)} rows")
    logger.info(f"✓ Sample query: {result_df['query_text'].iloc[0][:80]}...")
    
    return result_df

# ==============================
# TRAINING
# ==============================
def train(df):
    """Main training function"""
    logger.info(f"\n{'='*60}")
    logger.info("DPR FINE-TUNING")
    logger.info(f"{'='*60}")
    
    device = torch.device(CONFIG['device'])
    logger.info(f"✓ Device: {device}")
    
    # Load model
    logger.info(f"\nLoading model: {CONFIG['model_name']}")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    model = BertModel.from_pretrained(CONFIG['model_name'])
    model.to(device)
    
    # Dataset
    logger.info("Creating dataset...")
    dataset = DPRDataset(df, tokenizer, CONFIG['max_seq_length'])
    loader = DataLoader(dataset, batch_size=CONFIG['train_batch_size'], shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    total_steps = len(loader) * CONFIG['num_train_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CONFIG['warmup_steps'],
                                                num_training_steps=total_steps)
    
    logger.info(f"\nTraining Config:")
    logger.info(f"  Batch Size: {CONFIG['train_batch_size']}")
    logger.info(f"  Epochs: {CONFIG['num_train_epochs']}")
    logger.info(f"  Samples: {len(dataset)}")
    logger.info(f"  Total Steps: {total_steps}")
    logger.info(f"  Learning Rate: {CONFIG['learning_rate']}")
    
    logger.info(f"\n{'='*60}")
    logger.info("STARTING TRAINING")
    logger.info(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(CONFIG['num_train_epochs']):
        model.train()
        epoch_loss = 0
        
        for step, batch in enumerate(loader):
            q_ids = batch['q_ids'].to(device)
            q_mask = batch['q_mask'].to(device)
            p_ids = batch['p_ids'].to(device)
            p_mask = batch['p_mask'].to(device)
            n_ids = batch['n_ids'].to(device)
            n_mask = batch['n_mask'].to(device)
            
            # Forward pass
            q_out = model(q_ids, attention_mask=q_mask, output_hidden_states=True)
            q_embed = q_out.pooler_output
            
            p_out = model(p_ids, attention_mask=p_mask, output_hidden_states=True)
            p_embed = p_out.pooler_output
            
            n_out = model(n_ids, attention_mask=n_mask, output_hidden_states=True)
            n_embed = n_out.pooler_output
            
            # Cosine similarity
            pos_sim = torch.nn.functional.cosine_similarity(q_embed, p_embed, dim=1)
            neg_sim = torch.nn.functional.cosine_similarity(q_embed, n_embed, dim=1)
            
            # Loss: contrastive
            loss = -torch.log(torch.sigmoid(pos_sim - neg_sim) + 1e-8).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            if (step + 1) % 100 == 0:
                avg_loss = epoch_loss / (step + 1)
                logger.info(f"Epoch {epoch+1}/{CONFIG['num_train_epochs']} | Step {step+1}/{len(loader)} | Loss: {avg_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / len(loader)
        logger.info(f"✓ Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f}\n")
    
    # Save
    logger.info(f"Saving model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("✓ Model saved!")
    
    return model

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    try:
        torch.manual_seed(CONFIG['seed'])
        np.random.seed(CONFIG['seed'])
        
        logger.info("\n" + "="*60)
        logger.info("MS-MARCO DPR FINE-TUNING")
        logger.info("="*60)
        
        # Data
        df = prep_data(DATA_PATH)
        
        # Train
        model = train(df)
        
        logger.info("\n" + "="*60)
        logger.info("✓ TRAINING COMPLETE")
        logger.info(f"Model: {OUTPUT_DIR}")
        logger.info(f"Logs: {LOGS_DIR}")
        logger.info("="*60 + "\n")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
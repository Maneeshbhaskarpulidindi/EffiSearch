import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import argparse
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd

class Config:

    SLM_MODEL_PATH = "path/to/distilled-bert"  # Your distilled BERT model
    LLM_MODEL_PATH = "bert-base-multilingual-cased"  # BERT multilingual
    

    BATCH_SIZE = 64
    MAX_QUERY_LEN = 64
    MAX_DOC_LEN = 256
    NUM_WORKERS = 4
    TOP_K_VALUES = [10, 100, 1000]
    
    # Output
    OUTPUT_DIR = "./evaluation_results"
    
    # Mr.TyDi languages
    MRTYDI_LANGUAGES = ['arabic', 'bengali', 'english', 'finnish', 'indonesian', 
                        'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai']
    
    # mMARCO languages
    MMARCO_LANGUAGES = ['english', 'spanish', 'portuguese', 'french', 'italian', 
                        'indonesian', 'chinese', 'vietnamese', 'german']

def setup_logging(output_dir: str):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class DualEncoderModel(nn.Module):
    """Wrapper for dual encoder models (both SLM and LLM)"""
    def __init__(self, model_path: str, share_weights: bool = False):
        super().__init__()
        self.share_weights = share_weights
        
        if share_weights:
            self.encoder = AutoModel.from_pretrained(model_path)
            self.passage_encoder = self.encoder
            self.question_encoder = self.encoder
        else:
            self.passage_encoder = AutoModel.from_pretrained(model_path)
            self.question_encoder = AutoModel.from_pretrained(model_path)
    
    def encode(self, input_ids, attention_mask, encoder_type="passage"):
        """Encode text to dense vectors"""
        if self.share_weights:
            encoder = self.encoder
        else:
            encoder = self.passage_encoder if encoder_type == "passage" else self.question_encoder
        
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        # L2 normalize
        return torch.nn.functional.normalize(cls_output, p=2, dim=1)

class MrTyDiEvalDataset(Dataset):
    """Dataset for Mr.TyDi evaluation"""
    def __init__(self, queries_data, corpus_data, tokenizer, 
                 query_max_len=64, doc_max_len=256):
        self.queries = queries_data
        self.corpus = corpus_data
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len
        

        self.corpus_lookup = {doc['docid']: doc for doc in corpus_data}
        
        self.eval_pairs = []
        for query in queries_data:
            query_id = query['query_id']
            query_text = query['query']
            pos_docids = [p['docid'] for p in query['positive_passages']]
            self.eval_pairs.append({
                'query_id': query_id,
                'query_text': query_text,
                'positive_docids': pos_docids
            })
    
    def __len__(self):
        return len(self.eval_pairs)
    
    def __getitem__(self, idx):
        return self.eval_pairs[idx]
    
    def get_corpus_text(self, docid: str) -> str:
        """Get corpus text by docid"""
        doc = self.corpus_lookup.get(docid, {})
        title = doc.get('title', '')
        text = doc.get('text', '')
        return f"{title} {text}".strip() if title else text


class MMarcoEvalDataset(Dataset):
    def __init__(self, queries_data, collection_data, qrels_data, 
                 tokenizer, query_max_len=64, doc_max_len=256):
        self.queries = queries_data
        self.collection = collection_data
        self.qrels = qrels_data
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len
        
        self.collection_lookup = {str(doc['id']): doc['text'] for doc in collection_data}
        
        self.qrels_map = defaultdict(list)
        for qrel in qrels_data:
            self.qrels_map[str(qrel['query_id'])].append(str(qrel['doc_id']))
        
        self.eval_pairs = []
        for query in queries_data:
            query_id = str(query['id'])
            if query_id in self.qrels_map:
                self.eval_pairs.append({
                    'query_id': query_id,
                    'query_text': query['text'],
                    'positive_docids': self.qrels_map[query_id]
                })
    
    def __len__(self):
        return len(self.eval_pairs)
    
    def __getitem__(self, idx):
        return self.eval_pairs[idx]
    
    def get_corpus_text(self, docid: str) -> str:
        """Get collection text by docid"""
        return self.collection_lookup.get(docid, "")

def compute_mrr(ranks: List[int], k: int = 100) -> float:
    rr_sum = 0.0
    count = 0
    for rank_list in ranks:
        # Get first relevant rank
        relevant_ranks = [r for r in rank_list if r <= k]
        if relevant_ranks:
            rr_sum += 1.0 / min(relevant_ranks)
            count += 1
    return rr_sum / count if count > 0 else 0.0

def compute_recall(ranks: List[int], k: int = 100) -> float:
    """Compute Recall @ k"""
    recall_sum = 0.0
    for rank_list in ranks:
        if any(r <= k for r in rank_list):
            recall_sum += 1.0
    return recall_sum / len(ranks) if ranks else 0.0

def compute_ndcg(relevance_scores: List[List[float]], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain @ k"""
    ndcg_sum = 0.0
    for scores in relevance_scores:
        if not scores:
            continue
        # DCG
        dcg = sum((2**scores[i] - 1) / np.log2(i + 2) for i in range(min(k, len(scores))))
        # IDCG
        ideal_scores = sorted(scores, reverse=True)
        idcg = sum((2**ideal_scores[i] - 1) / np.log2(i + 2) for i in range(min(k, len(ideal_scores))))
        ndcg_sum += dcg / idcg if idcg > 0 else 0.0
    return ndcg_sum / len(relevance_scores) if relevance_scores else 0.0

def compute_map(ranks: List[List[int]], k: int = 100) -> float:
    """Compute Mean Average Precision @ k"""
    ap_sum = 0.0
    for rank_list in ranks:
        relevant_ranks = sorted([r for r in rank_list if r <= k])
        if not relevant_ranks:
            continue
        
        precision_sum = 0.0
        for i, rank in enumerate(relevant_ranks, 1):
            precision_at_rank = i / rank
            precision_sum += precision_at_rank
        
        ap = precision_sum / len(relevant_ranks)
        ap_sum += ap
    
    return ap_sum / len(ranks) if ranks else 0.0

class DenseRetriever:
    """Dense retrieval evaluator"""
    def __init__(self, model: DualEncoderModel, tokenizer, 
                 device: torch.device, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.model.eval()
    
    @torch.no_grad()
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode queries to dense vectors"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(queries), self.config.BATCH_SIZE), 
                     desc="Encoding queries", leave=False):
            batch_queries = queries[i:i + self.config.BATCH_SIZE]
            
            encoded = self.tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=self.config.MAX_QUERY_LEN,
                return_tensors='pt'
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            embeddings = self.model.encode(
                encoded['input_ids'], 
                encoded['attention_mask'], 
                encoder_type='query'
            )
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    @torch.no_grad()
    def encode_corpus(self, corpus_texts: List[str]) -> torch.Tensor:
        """Encode corpus to dense vectors"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(corpus_texts), self.config.BATCH_SIZE), 
                     desc="Encoding corpus", leave=False):
            batch_docs = corpus_texts[i:i + self.config.BATCH_SIZE]
            
            encoded = self.tokenizer(
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.config.MAX_DOC_LEN,
                return_tensors='pt'
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            embeddings = self.model.encode(
                encoded['input_ids'], 
                encoded['attention_mask'], 
                encoder_type='passage'
            )
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def retrieve_and_evaluate(self, eval_dataset, corpus_size: int = None) -> Dict:

        logger = logging.getLogger(__name__)
        
        # Sample corpus if specified
        if corpus_size and hasattr(eval_dataset, 'corpus_lookup'):
            all_docids = list(eval_dataset.corpus_lookup.keys())
            if len(all_docids) > corpus_size:
                # Keep all relevant docs + sample from rest
                relevant_docids = set()
                for pair in eval_dataset.eval_pairs:
                    relevant_docids.update(pair['positive_docids'])
                
                non_relevant = [d for d in all_docids if d not in relevant_docids]
                sampled = list(relevant_docids) + non_relevant[:corpus_size - len(relevant_docids)]
                corpus_docids = sampled[:corpus_size]
            else:
                corpus_docids = all_docids
        else:
            corpus_docids = list(eval_dataset.corpus_lookup.keys())
        
        logger.info(f"  Encoding {len(corpus_docids)} corpus documents...")
        corpus_texts = [eval_dataset.get_corpus_text(docid) for docid in corpus_docids]
        corpus_embeddings = self.encode_corpus(corpus_texts)
        
        logger.info(f"  Encoding {len(eval_dataset)} queries...")
        query_texts = [pair['query_text'] for pair in eval_dataset.eval_pairs]
        query_embeddings = self.encode_queries(query_texts)
        
        logger.info(f"  Computing similarities...")
        # Compute similarity scores (batch-wise to save memory)
        similarity_matrix = torch.matmul(query_embeddings, corpus_embeddings.T)
        
        logger.info(f"  Ranking documents...")
        # Get rankings for each query
        all_ranks = []
        all_relevance_scores = []
        
        for idx, pair in enumerate(tqdm(eval_dataset.eval_pairs, desc="Computing metrics", leave=False)):
            similarities = similarity_matrix[idx]
            ranked_indices = torch.argsort(similarities, descending=True).tolist()
            
            # Map indices back to docids
            ranked_docids = [corpus_docids[i] for i in ranked_indices]
            
            # Find ranks of relevant documents
            relevant_docids = set(pair['positive_docids'])
            ranks_for_query = []
            relevance_for_query = [0] * len(ranked_docids)
            
            for rank, docid in enumerate(ranked_docids, 1):
                if docid in relevant_docids:
                    ranks_for_query.append(rank)
                    relevance_for_query[rank - 1] = 1
            
            all_ranks.append(ranks_for_query)
            all_relevance_scores.append(relevance_for_query[:max(self.config.TOP_K_VALUES)])
        
        # Compute metrics for different k values
        logger.info(f"  Computing final metrics...")
        metrics = {}
        for k in self.config.TOP_K_VALUES:
            metrics[f'MRR@{k}'] = compute_mrr(all_ranks, k)
            metrics[f'Recall@{k}'] = compute_recall(all_ranks, k)
            metrics[f'MAP@{k}'] = compute_map(all_ranks, k)
        
        # NDCG only for k=10
        metrics['NDCG@10'] = compute_ndcg(all_relevance_scores, 10)
        
        return metrics

def evaluate_mrtydi(model: DualEncoderModel, tokenizer, device: torch.device,
                   config: Config, languages: List[str] = None) -> Dict:
    """Evaluate on Mr.TyDi dataset"""
    logger = logging.getLogger(__name__)
    
    if languages is None:
        languages = config.MRTYDI_LANGUAGES
    
    retriever = DenseRetriever(model, tokenizer, device, config)
    all_results = {}
    
    for lang in languages:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Mr.TyDi - {lang.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Load test set
            logger.info(f"Loading Mr.TyDi {lang} test set...")
            test_data = load_dataset('castorini/mr-tydi', lang, split='test')
            corpus_data = load_dataset('castorini/mr-tydi-corpus', lang, split='train')
            
            # Create dataset
            dataset = MrTyDiEvalDataset(
                queries_data=test_data,
                corpus_data=corpus_data,
                tokenizer=tokenizer,
                query_max_len=config.MAX_QUERY_LEN,
                doc_max_len=config.MAX_DOC_LEN
            )
            
            logger.info(f"  Dataset: {len(dataset)} queries, {len(dataset.corpus_lookup)} docs")
            
            # Evaluate
            metrics = retriever.retrieve_and_evaluate(dataset, corpus_size=None)
            all_results[lang] = metrics
            
            # Log results
            logger.info(f"\n  Results for {lang}:")
            for metric_name, value in metrics.items():
                logger.info(f"    {metric_name}: {value:.4f}")
        
        except Exception as e:
            logger.error(f"  Error evaluating {lang}: {str(e)}")
            all_results[lang] = {}
    
    return all_results

def evaluate_mmarco(model: DualEncoderModel, tokenizer, device: torch.device,
                   config: Config, languages: List[str] = None) -> Dict:
    """Evaluate on mMARCO dataset"""
    logger = logging.getLogger(__name__)
    
    if languages is None:
        languages = config.MMARCO_LANGUAGES
    
    retriever = DenseRetriever(model, tokenizer, device, config)
    all_results = {}
    
    for lang in languages:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating mMARCO - {lang.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Load dataset components
            logger.info(f"Loading mMARCO {lang} dataset...")
            queries = load_dataset('unicamp-dl/mmarco', f'queries-{lang}', split='dev')
            collection = load_dataset('unicamp-dl/mmarco', f'collection-{lang}', split='collection')
            
            # Load qrels (relevance judgments) - using English as reference
            # Note: You may need to adjust this based on actual qrels availability
            try:
                qrels = load_dataset('unicamp-dl/mmarco', 'qrels', split='dev')
            except:
                logger.warning(f"  Qrels not found for {lang}, skipping...")
                continue
            
            # Create dataset
            dataset = MMarcoEvalDataset(
                queries_data=queries,
                collection_data=collection,
                qrels_data=qrels,
                tokenizer=tokenizer,
                query_max_len=config.MAX_QUERY_LEN,
                doc_max_len=config.MAX_DOC_LEN
            )
            
            logger.info(f"  Dataset: {len(dataset)} queries, {len(dataset.collection_lookup)} docs")
            
            # Evaluate (use sampled corpus for efficiency)
            metrics = retriever.retrieve_and_evaluate(dataset, corpus_size=100000)
            all_results[lang] = metrics
            
            # Log results
            logger.info(f"\n  Results for {lang}:")
            for metric_name, value in metrics.items():
                logger.info(f"    {metric_name}: {value:.4f}")
        
        except Exception as e:
            logger.error(f"  Error evaluating {lang}: {str(e)}")
            all_results[lang] = {}
    
    return all_results

def plot_results(slm_results: Dict, llm_results: Dict, 
                dataset_name: str, output_dir: str):
    """Plot comparison between SLM and LLM"""
    
    # Extract metrics
    languages = list(slm_results.keys())
    metrics_to_plot = ['MRR@100', 'Recall@100', 'NDCG@10']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
    fig.suptitle(f'{dataset_name} - SLM vs LLM Comparison', fontsize=16)
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        slm_scores = [slm_results.get(lang, {}).get(metric, 0) for lang in languages]
        llm_scores = [llm_results.get(lang, {}).get(metric, 0) for lang in languages]
        
        x = np.arange(len(languages))
        width = 0.35
        
        ax.bar(x - width/2, slm_scores, width, label='SLM', alpha=0.8)
        ax.bar(x + width/2, llm_scores, width, label='LLM', alpha=0.8)
        
        ax.set_xlabel('Language')
        ax.set_ylabel('Score')
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(languages, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_results_table(slm_results: Dict, llm_results: Dict, 
                      dataset_name: str, output_dir: str):
    """Save results as CSV table"""
    
    # Create DataFrame
    data = []
    for lang in slm_results.keys():
        row = {'Language': lang}
        
        # Add SLM metrics
        for metric, value in slm_results[lang].items():
            row[f'SLM_{metric}'] = value
        
        # Add LLM metrics
        for metric, value in llm_results[lang].items():
            row[f'LLM_{metric}'] = value
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, f'{dataset_name.lower()}_results.csv'), 
              index=False)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Evaluate SLM and LLM on Mr.TyDi and mMARCO')
    parser.add_argument('--slm_model', type=str, required=True,
                       help='Path to distilled BERT (SLM) model')
    parser.add_argument('--llm_model', type=str, default='bert-base-multilingual-cased',
                       help='Path to BERT multilingual (LLM) model')
    parser.add_argument('--datasets', nargs='+', default=['mrtydi', 'mmarco'],
                       choices=['mrtydi', 'mmarco'],
                       help='Datasets to evaluate on')
    parser.add_argument('--languages', nargs='+', default=None,
                       help='Specific languages to evaluate (default: all)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for encoding')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.SLM_MODEL_PATH = args.slm_model
    config.LLM_MODEL_PATH = args.llm_model
    config.OUTPUT_DIR = args.output_dir
    config.BATCH_SIZE = args.batch_size
    
    # Setup logging
    logger = setup_logging(config.OUTPUT_DIR)
    logger.info("="*60)
    logger.info("MULTILINGUAL IR EVALUATION")
    logger.info("="*60)
    logger.info(f"SLM Model: {config.SLM_MODEL_PATH}")
    logger.info(f"LLM Model: {config.LLM_MODEL_PATH}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Device: {args.device}")
    logger.info("="*60)
    
    device = torch.device(args.device)
    
    # Load models
    logger.info("\nLoading models...")
    logger.info("  Loading SLM (Distilled BERT)...")
    slm_tokenizer = AutoTokenizer.from_pretrained(config.SLM_MODEL_PATH)
    slm_model = DualEncoderModel(config.SLM_MODEL_PATH, share_weights=False).to(device)
    
    logger.info("  Loading LLM (BERT Multilingual)...")
    llm_tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_PATH)
    llm_model = DualEncoderModel(config.LLM_MODEL_PATH, share_weights=False).to(device)
    
    # Evaluate on datasets
    results = {
        'slm': {},
        'llm': {}
    }
    
    if 'mrtydi' in args.datasets:
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ON MR.TYDI")
        logger.info("="*60)
        
        logger.info("\n--- SLM Evaluation ---")
        results['slm']['mrtydi'] = evaluate_mrtydi(
            slm_model, slm_tokenizer, device, config, args.languages
        )
        
        logger.info("\n--- LLM Evaluation ---")
        results['llm']['mrtydi'] = evaluate_mrtydi(
            llm_model, llm_tokenizer, device, config, args.languages
        )
        
        # Visualize and save
        plot_results(results['slm']['mrtydi'], results['llm']['mrtydi'], 
                    'Mr.TyDi', config.OUTPUT_DIR)
        save_results_table(results['slm']['mrtydi'], results['llm']['mrtydi'], 
                          'Mr.TyDi', config.OUTPUT_DIR)
    
    if 'mmarco' in args.datasets:
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ON mMARCO")
        logger.info("="*60)
        
        logger.info("\n--- SLM Evaluation ---")
        results['slm']['mmarco'] = evaluate_mmarco(
            slm_model, slm_tokenizer, device, config, args.languages
        )
        
        logger.info("\n--- LLM Evaluation ---")
        results['llm']['mmarco'] = evaluate_mmarco(
            llm_model, llm_tokenizer, device, config, args.languages
        )
        
        # Visualize and save
        plot_results(results['slm']['mmarco'], results['llm']['mmarco'], 
                    'mMARCO', config.OUTPUT_DIR)
        save_results_table(results['slm']['mmarco'], results['llm']['mmarco'], 
                          'mMARCO', config.OUTPUT_DIR)
    
    # Save complete results
    logger.info("\n" + "="*60)
    logger.info("SAVING RESULTS")
    logger.info("="*60)
    
    with open(os.path.join(config.OUTPUT_DIR, 'complete_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ All results saved to {config.OUTPUT_DIR}")
    logger.info("="*60)

if __name__ == "__main__":
    main()


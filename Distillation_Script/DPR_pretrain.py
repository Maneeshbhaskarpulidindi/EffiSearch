"""Distillation=>Only-MSMARCO.ipynb=>Pre-trained
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import load_dataset
from typing import Iterable, Dict
import logging

logging.basicConfig(level=logging.INFO)

class MSECosineDistillationLoss(nn.Module):
    """
    Combined MSE + Cosine similarity loss for knowledge distillation.
    Student learns to mimic teacher's embeddings on MS MARCO queries.
    """
    def __init__(self, teacher_model: SentenceTransformer, student_model: SentenceTransformer, alpha: float = 0.5):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha
        self.mse = nn.MSELoss()

        # Freeze teacher - no gradients needed
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        """
        Compute distillation loss between teacher and student embeddings.
        """
        # Get teacher embeddings (frozen, no grad)
        with torch.no_grad():
            teacher_emb = self.teacher(sentence_features[0])['sentence_embedding']

        # Get student embeddings (trainable)
        student_emb = self.student(sentence_features[0])['sentence_embedding']

        # MSE Loss - L2 distance between embeddings
        mse_loss = self.mse(student_emb, teacher_emb)

        # Cosine Similarity Loss - directional alignment
        student_norm = F.normalize(student_emb, p=2, dim=1)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=1)
        cosine_sim = torch.sum(student_norm * teacher_norm, dim=1)
        cosine_loss = 1 - torch.mean(cosine_sim)

        # Combined loss
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * cosine_loss

        return total_loss

print("Loading models from local directory...")

# Load BOTH query and context encoders
teacher_query_path = 'data/trained_models/pretrained/DPR-bert-msmarco/query_encoder'
student_query_path = 'data/trained_models/pretrained/DPR-distilbert-msmarco/query_encoder'

teacher_context_path = 'data/trained_models/pretrained/DPR-bert-msmarco/context_encoder'
student_context_path = 'data/trained_models/pretrained/DPR-distilbert-msmarco/context_encoder'

try:
    # Load query encoders
    print("Loading query encoders...")
    teacher_query = SentenceTransformer(teacher_query_path)
    student_query = SentenceTransformer(student_query_path)

    # Load context encoders
    print("Loading context encoders...")
    teacher_context = SentenceTransformer(teacher_context_path)
    student_context = SentenceTransformer(student_context_path)

    print("✓ All models loaded successfully from local DPR directories\n")
except Exception as e:
    print(f"❌ Error loading from local: {e}")
    print("Falling back to downloading from HuggingFace...\n")
    teacher_query = SentenceTransformer('bert-base-multilingual-cased')
    student_query = SentenceTransformer('distilbert-base-multilingual-cased')
    teacher_context = SentenceTransformer('bert-base-multilingual-cased')
    student_context = SentenceTransformer('distilbert-base-multilingual-cased')

# Print statistics for query encoder
print("Query Encoder:")
print(f"  Teacher params: {sum(p.numel() for p in teacher_query.parameters()):,}")
print(f"  Student params: {sum(p.numel() for p in student_query.parameters()):,}")
query_compression = sum(p.numel() for p in teacher_query.parameters()) / sum(p.numel() for p in student_query.parameters())
print(f"  Compression ratio: {query_compression:.2f}x smaller\n")

# Print statistics for context encoder
print("Context Encoder:")
print(f"  Teacher params: {sum(p.numel() for p in teacher_context.parameters()):,}")
print(f"  Student params: {sum(p.numel() for p in student_context.parameters()):,}")
context_compression = sum(p.numel() for p in teacher_context.parameters()) / sum(p.numel() for p in student_context.parameters())
print(f"  Compression ratio: {context_compression:.2f}x smaller\n")

print("Loading MS MARCO dataset...")

train_examples_queries = []
train_examples_passages = []

try:
    # Load MS MARCO v2.1 - passage ranking task
    msmarco = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)

    # Sample size - adjust based on your needs

    print(f"Extracting {MAX_SAMPLES} query-passage pairs from MS MARCO...")

    count = 0
    for example in msmarco:

        # MS MARCO has queries and passages
        query = example.get("query", "")
        passages = example.get("passages", {})

        if query and passages:
            # Extract passage text
            passage_texts = passages.get("passage_text", [])
            if passage_texts and len(passage_texts) > 0:
                # Add query for query encoder distillation
                train_examples_queries.append(InputExample(texts=[query]))

                # Add passages for context encoder distillation
                # Take up to 3 passages per query to balance dataset
                for passage in passage_texts[:3]:
                    if passage and len(passage.strip()) > 0:
                        train_examples_passages.append(InputExample(texts=[passage]))

                count += 1

        if count % 10000 == 0 and count > 0:
            print(f"  Loaded {count} examples...")

    print(f"✓ Successfully loaded {len(train_examples_queries)} queries")
    print(f"✓ Successfully loaded {len(train_examples_passages)} passages\n")

except Exception as e:
    print(f"❌ Error loading MS MARCO: {e}")
    print("Creating dummy MS MARCO-style data for demonstration...\n")

    # Fallback dummy data
    dummy_queries = [
        "what is the definition of machine learning",
        "how does neural network work",
        "what are the benefits of deep learning",
        "explain transformer architecture",
        "what is attention mechanism in nlp",
    ] * 200

    dummy_passages = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "Deep learning provides advantages like automatic feature extraction and handling of unstructured data.",
        "The transformer architecture uses self-attention mechanisms to process sequential data efficiently.",
        "Attention mechanism allows models to focus on relevant parts of input when making predictions.",
    ] * 200

    train_examples_queries = [InputExample(texts=[query]) for query in dummy_queries]
    train_examples_passages = [InputExample(texts=[passage]) for passage in dummy_passages]
    print(f"✓ Created {len(train_examples_queries)} dummy queries")
    print(f"✓ Created {len(train_examples_passages)} dummy passages\n")

# Create DataLoaders for both
BATCH_SIZE = 16

query_dataloader = DataLoader(
    train_examples_queries,
    shuffle=True,
    batch_size=BATCH_SIZE
)

passage_dataloader = DataLoader(
    train_examples_passages,
    shuffle=True,
    batch_size=BATCH_SIZE
)

print(f"Query DataLoader:")
print(f"  Total examples: {len(train_examples_queries)}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Number of batches: {len(query_dataloader)}")

print(f"\nPassage DataLoader:")
print(f"  Total examples: {len(train_examples_passages)}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Number of batches: {len(passage_dataloader)}\n")

print("Setting up distillation losses...")

# Query encoder distillation loss
query_distillation_loss = MSECosineDistillationLoss(
    teacher_model=teacher_query,
    student_model=student_query,
    alpha=0.5  # 50% MSE + 50% Cosine
)

# Context encoder distillation loss
context_distillation_loss = MSECosineDistillationLoss(
    teacher_model=teacher_context,
    student_model=student_context,
    alpha=0.5  # 50% MSE + 50% Cosine
)

print(f"Query Encoder Loss: {query_distillation_loss.alpha:.0%} MSE + {1-query_distillation_loss.alpha:.0%} Cosine")
print(f"Context Encoder Loss: {context_distillation_loss.alpha:.0%} MSE + {1-context_distillation_loss.alpha:.0%} Cosine")
print("Teacher models frozen ✓\n")

print("="*60)
print("STARTING DISTILLATION TRAINING - QUERY ENCODER")
print("="*60)

# Training hyperparameters
NUM_EPOCHS = 3
WARMUP_STEPS = 100
LEARNING_RATE = 2e-5

print(f"\nTraining Configuration:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: 0.01")
print(f"  Mixed precision: Enabled")
print(f"  Dataset: MS MARCO queries\n")

# Train query encoder
student_query.fit(
    train_objectives=[(query_dataloader, query_distillation_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    optimizer_params={'lr': LEARNING_RATE},
    weight_decay=0.01,
    use_amp=True,
    show_progress_bar=True,
    checkpoint_path='./checkpoints/query_encoder',
    checkpoint_save_steps=500,
    checkpoint_save_total_limit=2
)

print("\n" + "="*60)
print("QUERY ENCODER TRAINING COMPLETE")
print("="*60 + "\n")

print("="*60)
print("STARTING DISTILLATION TRAINING - CONTEXT ENCODER")
print("="*60)

print(f"\nTraining Configuration:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: 0.01")
print(f"  Mixed precision: Enabled")
print(f"  Dataset: MS MARCO passages\n")

# Train context encoder
student_context.fit(
    train_objectives=[(passage_dataloader, context_distillation_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    optimizer_params={'lr': LEARNING_RATE},
    weight_decay=0.01,
    use_amp=True,
    show_progress_bar=True,
    checkpoint_path='./checkpoints/context_encoder',
    checkpoint_save_steps=500,
    checkpoint_save_total_limit=2
)

print("\n" + "="*60)
print("CONTEXT ENCODER TRAINING COMPLETE")
print("="*60 + "\n")

print("="*60)
print("ALL TRAINING COMPLETE")
print("="*60 + "\n")

# Save both encoders
query_output_path = "distilled-msmarco-query-encoder"
context_output_path = "distilled-msmarco-context-encoder"

student_query.save(query_output_path)
print(f"✅ Query encoder saved to: {query_output_path}")

student_context.save(context_output_path)
print(f"✅ Context encoder saved to: {context_output_path}\n")

print("="*60)
print("POST-TRAINING EVALUATION")
print("="*60 + "\n")

# Sample MS MARCO-style queries
test_queries = [
    "what is machine learning",
    "how does a neural network work",
    "what are the applications of NLP",
    "explain deep learning",
    "what is transfer learning"
]

# Sample passages
test_passages = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "A neural network is a series of algorithms that endeavors to recognize underlying relationships in data.",
    "Natural language processing applications include machine translation, sentiment analysis, and chatbots.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
    "Transfer learning is a machine learning method where a model developed for one task is reused as the starting point."
]

print("="*60)
print("QUERY ENCODER EVALUATION")
print("="*60 + "\n")
print(f"{'Query':<40} {'Similarity':<12} {'MSE':<10}")
print("-" * 65)

total_query_similarity = 0
total_query_mse = 0

for query in test_queries:
    teacher_emb = teacher_query.encode(query, convert_to_tensor=True)
    student_emb = student_query.encode(query, convert_to_tensor=True)

    similarity = F.cosine_similarity(teacher_emb.unsqueeze(0), student_emb.unsqueeze(0)).item()
    mse = F.mse_loss(student_emb, teacher_emb).item()

    total_query_similarity += similarity
    total_query_mse += mse

    print(f"{query[:38]:<40} {similarity:>6.4f}      {mse:>8.4f}")

print("-" * 65)
avg_query_similarity = total_query_similarity / len(test_queries)
avg_query_mse = total_query_mse / len(test_queries)
print(f"{'AVERAGE':<40} {avg_query_similarity:>6.4f}      {avg_query_mse:>8.4f}\n")

print("="*60)
print("CONTEXT ENCODER EVALUATION")
print("="*60 + "\n")
print(f"{'Passage':<40} {'Similarity':<12} {'MSE':<10}")
print("-" * 65)

total_context_similarity = 0
total_context_mse = 0

for passage in test_passages:
    teacher_emb = teacher_context.encode(passage, convert_to_tensor=True)
    student_emb = student_context.encode(passage, convert_to_tensor=True)

    similarity = F.cosine_similarity(teacher_emb.unsqueeze(0), student_emb.unsqueeze(0)).item()
    mse = F.mse_loss(student_emb, teacher_emb).item()

    total_context_similarity += similarity
    total_context_mse += mse

    print(f"{passage[:38]:<40} {similarity:>6.4f}      {mse:>8.4f}")

print("-" * 65)
avg_context_similarity = total_context_similarity / len(test_passages)
avg_context_mse = total_context_mse / len(test_passages)
print(f"{'AVERAGE':<40} {avg_context_similarity:>6.4f}      {avg_context_mse:>8.4f}\n")

print("="*60)
print("OVERALL RESULTS")
print("="*60 + "\n")

print("Query Encoder:")
print(f"  • Cosine Similarity: {avg_query_similarity:.4f}")
print(f"  • MSE: {avg_query_mse:.4f}")

print("\nContext Encoder:")
print(f"  • Cosine Similarity: {avg_context_similarity:.4f}")
print(f"  • MSE: {avg_context_mse:.4f}\n")

print("Model Efficiency:")
print(f"  • Query Teacher: {sum(p.numel() for p in teacher_query.parameters()):,} parameters")
print(f"  • Query Student: {sum(p.numel() for p in student_query.parameters()):,} parameters")
print(f"  • Context Teacher: {sum(p.numel() for p in teacher_context.parameters()):,} parameters")
print(f"  • Context Student: {sum(p.numel() for p in student_context.parameters()):,} parameters\n")

print("="*60)
print("DISTILLATION PIPELINE COMPLETE! 🎉")
print("="*60)

print("\nNext steps:")
print("  1. Evaluate on MS MARCO dev set for retrieval metrics (MRR@10, etc.)")
print("  2. Test end-to-end retrieval with both encoders")
print("  3. Compare inference speed: teacher vs student")
print("  4. Fine-tune further if needed")
print(f"\nModels ready at:")
print(f"  - Query: {query_output_path}")
print(f"  - Context: {context_output_path}")

# Full MS MARCO evaluation with BOTH encoders
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
from typing import List

print("Loading MS MARCO dev set...")
try:
    msmarco_dev = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
except Exception as e:
    print(f"❌ Error loading MS MARCO dev set: {e}")
    raise SystemExit("Evaluation requires MS MARCO dev set.")

def compute_mrr_at_k(ranks: List[int], k: int = 10) -> float:
    """Compute Mean Reciprocal Rank at k."""
    mrr = 0.0
    for rank in ranks:
        if rank <= k:
            mrr += 1.0 / rank
    return mrr / len(ranks) if ranks else 0.0

def compute_precision_recall_at_k(relevant: List[bool], k: int = 10) -> tuple:
    """Compute Precision@k and Recall@k."""
    if not relevant:
        return 0.0, 0.0
    top_k_relevant = relevant[:k]
    precision = sum(top_k_relevant) / k
    total_relevant = sum(relevant)
    recall = sum(top_k_relevant) / total_relevant if total_relevant > 0 else 0.0
    return precision, recall

print("="*60)
print("EVALUATING DPR SYSTEM ON MS MARCO DEV SET")
print("="*60 + "\n")

teacher_mrr, student_mrr = [], []
teacher_precisions, teacher_recalls = [], []
student_precisions, student_recalls = [], []

count = 0

for example in msmarco_dev:

    query = example.get("query", "")
    passages = example.get("passages", {})
    if not query or not passages:
        continue

    passage_texts = passages.get("passage_text", [])
    is_selected = passages.get("is_selected", [])
    if len(passage_texts) == 0 or len(is_selected) == 0:
        continue

    # Encode query with query encoder
    teacher_query_emb = teacher_query.encode(query, convert_to_tensor=True)
    student_query_emb = student_query.encode(query, convert_to_tensor=True)

    # Encode passages with context encoder
    teacher_passage_embs = teacher_context.encode(passage_texts, convert_to_tensor=True)
    student_passage_embs = student_context.encode(passage_texts, convert_to_tensor=True)

    # Compute cosine similarities
    teacher_scores = F.cosine_similarity(teacher_query_emb.unsqueeze(0), teacher_passage_embs).cpu().numpy()
    student_scores = F.cosine_similarity(student_query_emb.unsqueeze(0), student_passage_embs).cpu().numpy()

    # Rank passages
    teacher_ranking = np.argsort(-teacher_scores)
    student_ranking = np.argsort(-student_scores)

    # Find first relevant passage rank
    teacher_first_relevant = float('inf')
    student_first_relevant = float('inf')
    relevant_passages = [bool(is_selected[i]) for i in range(len(is_selected))]

    for rank, idx in enumerate(teacher_ranking, 1):
        if relevant_passages[idx]:
            teacher_first_relevant = rank
            break
    for rank, idx in enumerate(student_ranking, 1):
        if relevant_passages[idx]:
            student_first_relevant = rank
            break

    if teacher_first_relevant != float('inf'):
        teacher_mrr.append(teacher_first_relevant)
    if student_first_relevant != float('inf'):
        student_mrr.append(student_first_relevant)

    # Compute Precision@10 and Recall@10
    teacher_top_k = [relevant_passages[idx] for idx in teacher_ranking[:10]]
    student_top_k = [relevant_passages[idx] for idx in student_ranking[:10]]

    teacher_precision, teacher_recall = compute_precision_recall_at_k(teacher_top_k)
    student_precision, student_recall = compute_precision_recall_at_k(student_top_k)

    teacher_precisions.append(teacher_precision)
    teacher_recalls.append(teacher_recall)
    student_precisions.append(student_precision)
    student_recalls.append(student_recall)

    count += 1
    if count % 100 == 0:
        print(f"Processed {count} queries...")

print(f"\n✓ Evaluated {count} queries\n")

# Aggregate metrics
teacher_mrr_score = compute_mrr_at_k(teacher_mrr)
student_mrr_score = compute_mrr_at_k(student_mrr)
teacher_precision_avg = np.mean(teacher_precisions)
student_precision_avg = np.mean(student_precisions)
teacher_recall_avg = np.mean(teacher_recalls)
student_recall_avg = np.mean(student_recalls)

print("="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"{'Metric':<20} {'Teacher':<15} {'Student':<15} {'Difference':<15}")
print("-"*65)
print(f"{'MRR@10':<20} {teacher_mrr_score:>6.4f}      {student_mrr_score:>6.4f}      {teacher_mrr_score - student_mrr_score:>6.4f}")
print(f"{'Precision@10':<20} {teacher_precision_avg:>6.4f}      {student_precision_avg:>6.4f}      {teacher_precision_avg - student_precision_avg:>6.4f}")
print(f"{'Recall@10':<20} {teacher_recall_avg:>6.4f}      {student_recall_avg:>6.4f}      {teacher_recall_avg - student_recall_avg:>6.4f}")
print()

print("Model Efficiency:")
teacher_params = sum(p.numel() for p in teacher_query.parameters()) + sum(p.numel() for p in teacher_context.parameters())
student_params = sum(p.numel() for p in student_query.parameters()) + sum(p.numel() for p in student_context.parameters())
compression_ratio = teacher_params / student_params

print(f"  • Teacher (both encoders): {teacher_params:,} parameters")
print(f"  • Student (both encoders): {student_params:,} parameters")
print(f"  • Compression: {compression_ratio:.2f}x smaller, {(1-1/compression_ratio)*100:.1f}% reduction")
print()

print("="*60)
print("EVALUATION COMPLETE")
print("="*60)

print(f"\n• Student retains {student_mrr_score/teacher_mrr_score*100:.1f}% of teacher's MRR@10 performance.")
print(f"• Using separate query and context encoders for proper DPR architecture.")
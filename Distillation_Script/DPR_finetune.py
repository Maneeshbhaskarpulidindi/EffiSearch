"""Distillation=>Finetune-MrTyDi.ipynb=>Hybrid Loss
Fine-tuning with combined distillation + task-specific loss on Mr. TyDi dataset
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

class HybridDistillationLoss(nn.Module):
    """
    Hybrid loss combining:
    1. Distillation loss (MSE + Cosine) - learn from teacher
    2. Task-specific loss (Multiple Negatives Ranking) - learn from labels
    
    This allows the student to both mimic the teacher AND optimize for retrieval task.
    """
    def __init__(
        self, 
        teacher_model: SentenceTransformer, 
        student_model: SentenceTransformer, 
        distillation_weight: float = 0.5,
        mse_cosine_alpha: float = 0.5
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.distillation_weight = distillation_weight  # Weight for distillation vs task loss
        self.mse_cosine_alpha = mse_cosine_alpha  # Weight for MSE vs Cosine in distillation
        self.mse = nn.MSELoss()
        
        # Task-specific loss (Multiple Negatives Ranking Loss)
        self.task_loss = losses.MultipleNegativesRankingLoss(student_model)

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        """
        Compute hybrid loss:
        - Distillation component: MSE + Cosine similarity with teacher
        - Task component: Multiple Negatives Ranking Loss with labels
        """
        # === DISTILLATION LOSS ===
        with torch.no_grad():
            teacher_emb = self.teacher(sentence_features[0])['sentence_embedding']

        student_emb = self.student(sentence_features[0])['sentence_embedding']

        # MSE Loss
        mse_loss = self.mse(student_emb, teacher_emb)

        # Cosine Similarity Loss
        student_norm = F.normalize(student_emb, p=2, dim=1)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=1)
        cosine_sim = torch.sum(student_norm * teacher_norm, dim=1)
        cosine_loss = 1 - torch.mean(cosine_sim)

        # Combined distillation loss
        distillation_loss = self.mse_cosine_alpha * mse_loss + (1 - self.mse_cosine_alpha) * cosine_loss

        # === TASK-SPECIFIC LOSS ===
        # Multiple Negatives Ranking Loss for retrieval task
        task_loss_value = self.task_loss(sentence_features, labels)

        # === HYBRID LOSS ===
        total_loss = (
            self.distillation_weight * distillation_loss + 
            (1 - self.distillation_weight) * task_loss_value
        )

        return total_loss

print("Loading fine-tuned models from local directory...")

# Load BOTH query and context encoders (fine-tuned from distillation)
teacher_query_path = 'data/trained_models/finetuned/DPR-bert-mrtydi/query_encoder'
student_query_path = 'distilled-mrtydi-query-encoder'  # From pre-training distillation

teacher_context_path = 'data/trained_models/finetuned/DPR-bert-mrtydi/context_encoder'
student_context_path = 'distilled-mrtydi-context-encoder'  # From pre-training distillation

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

print("Loading Mr. TyDi dataset for fine-tuning...")

train_examples_queries = []
train_examples_passages = []

try:
    # Load Mr. TyDi - multilingual question answering dataset
    LANGUAGES = ["arabic", "bengali", "english", "finnish", "indonesian", "japanese", "korean", "russian", "swahili", "telugu", "thai"]    
    
    mrtydi = load_dataset("castorini/mr-tydi", LANGUAGE, split="train", streaming=True)

    print(f"Extracting {MAX_SAMPLES} query-passage pairs from Mr. TyDi ({LANGUAGE})...")

    count = 0
    for example in mrtydi:

        # Mr. TyDi structure: query, positive_passages, negative_passages
        query = example.get("query", "")
        positive_passages = example.get("positive_passages", [])
        
        if query and positive_passages and len(positive_passages) > 0:
            # Get the first positive passage
            positive_passage = positive_passages[0]
            passage_text = positive_passage.get("text", "")
            
            if passage_text and len(passage_text.strip()) > 0:
                # Add query-positive pair for query encoder
                train_examples_queries.append(
                    InputExample(texts=[query, passage_text])
                )

                # Add query-positive pair for context encoder
                train_examples_passages.append(
                    InputExample(texts=[query, passage_text])
                )
                
                count += 1

        if count % 1000 == 0 and count > 0:
            print(f"  Loaded {count} examples...")

    print(f"✓ Successfully loaded {len(train_examples_queries)} query-positive pairs")
    print(f"✓ Successfully loaded {len(train_examples_passages)} passage pairs\n")

except Exception as e:
    print(f"❌ Error loading Mr. TyDi: {e}")
    print("Creating dummy Mr. TyDi-style data for demonstration...\n")

    # Fallback dummy data with multilingual query-positive pairs
    dummy_query_pairs = [
        ("what is artificial intelligence", 
         "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems."),
        ("how do computers learn", 
         "Computers learn through machine learning algorithms that identify patterns in data and make predictions."),
        ("what are neural networks", 
         "Neural networks are computing systems vaguely inspired by the biological neural networks in animal brains."),
        ("explain natural language processing", 
         "Natural language processing is a branch of AI that helps computers understand, interpret and manipulate human language."),
        ("what is deep learning", 
         "Deep learning is a subset of machine learning that uses neural networks with multiple layers to analyze data."),
    ] * 200

    train_examples_queries = [InputExample(texts=[q, p]) for q, p in dummy_query_pairs]
    train_examples_passages = [InputExample(texts=[q, p]) for q, p in dummy_query_pairs]
    print(f"✓ Created {len(train_examples_queries)} dummy query-positive pairs")
    print(f"✓ Created {len(train_examples_passages)} dummy passage pairs\n")

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

print("Setting up hybrid distillation losses...")

# Query encoder hybrid loss
query_hybrid_loss = HybridDistillationLoss(
    teacher_model=teacher_query,
    student_model=student_query,
    distillation_weight=0.5,  # 50% distillation + 50% task loss
    mse_cosine_alpha=0.5  # Within distillation: 50% MSE + 50% Cosine
)

# Context encoder hybrid loss
context_hybrid_loss = HybridDistillationLoss(
    teacher_model=teacher_context,
    student_model=student_context,
    distillation_weight=0.5,  # 50% distillation + 50% task loss
    mse_cosine_alpha=0.5  # Within distillation: 50% MSE + 50% Cosine
)

print(f"Query Encoder Hybrid Loss:")
print(f"  Distillation weight: {query_hybrid_loss.distillation_weight:.0%}")
print(f"  Task loss weight: {1-query_hybrid_loss.distillation_weight:.0%}")
print(f"  Within distillation: {query_hybrid_loss.mse_cosine_alpha:.0%} MSE + {1-query_hybrid_loss.mse_cosine_alpha:.0%} Cosine")

print(f"\nContext Encoder Hybrid Loss:")
print(f"  Distillation weight: {context_hybrid_loss.distillation_weight:.0%}")
print(f"  Task loss weight: {1-context_hybrid_loss.distillation_weight:.0%}")
print(f"  Within distillation: {context_hybrid_loss.mse_cosine_alpha:.0%} MSE + {1-context_hybrid_loss.mse_cosine_alpha:.0%} Cosine")
print("Teacher models frozen ✓\n")

print("="*60)
print("STARTING FINE-TUNING - QUERY ENCODER (HYBRID LOSS)")
print("="*60)

# Fine-tuning hyperparameters
NUM_EPOCHS = 3
WARMUP_STEPS = 100
LEARNING_RATE = 2e-5

print(f"\nFine-tuning Configuration:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: 0.01")
print(f"  Mixed precision: Enabled")
print(f"  Dataset: Mr. TyDi query-positive pairs")
print(f"  Loss: Hybrid (Distillation + Multiple Negatives Ranking)\n")

# Fine-tune query encoder
student_query.fit(
    train_objectives=[(query_dataloader, query_hybrid_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    optimizer_params={'lr': LEARNING_RATE},
    weight_decay=0.01,
    use_amp=True,
    show_progress_bar=True,
    checkpoint_path='./checkpoints/query_encoder_finetuned',
    checkpoint_save_steps=500,
    checkpoint_save_total_limit=2
)

print("\n" + "="*60)
print("QUERY ENCODER FINE-TUNING COMPLETE")
print("="*60 + "\n")

print("="*60)
print("STARTING FINE-TUNING - CONTEXT ENCODER (HYBRID LOSS)")
print("="*60)

print(f"\nFine-tuning Configuration:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: 0.01")
print(f"  Mixed precision: Enabled")
print(f"  Dataset: Mr. TyDi passage pairs")
print(f"  Loss: Hybrid (Distillation + Multiple Negatives Ranking)\n")

# Fine-tune context encoder
student_context.fit(
    train_objectives=[(passage_dataloader, context_hybrid_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    optimizer_params={'lr': LEARNING_RATE},
    weight_decay=0.01,
    use_amp=True,
    show_progress_bar=True,
    checkpoint_path='./checkpoints/context_encoder_finetuned',
    checkpoint_save_steps=500,
    checkpoint_save_total_limit=2
)

print("\n" + "="*60)
print("CONTEXT ENCODER FINE-TUNING COMPLETE")
print("="*60 + "\n")

print("="*60)
print("ALL FINE-TUNING COMPLETE")
print("="*60 + "\n")

# Save both fine-tuned encoders
query_output_path = "finetuned-mrtydi-query-encoder"
context_output_path = "finetuned-mrtydi-context-encoder"

student_query.save(query_output_path)
print(f"✅ Fine-tuned query encoder saved to: {query_output_path}")

student_context.save(context_output_path)
print(f"✅ Fine-tuned context encoder saved to: {context_output_path}\n")

print("="*60)
print("POST-FINE-TUNING EVALUATION")
print("="*60 + "\n")

# Sample Mr. TyDi-style queries (multilingual examples)
test_queries = [
    "what is artificial intelligence",
    "how do neural networks work",
    "what are the applications of machine learning",
    "explain deep learning algorithms",
    "what is natural language processing"
]

# Sample passages
test_passages = [
    "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
    "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
    "Machine learning applications include image recognition, speech recognition, medical diagnosis, and recommendation systems.",
    "Deep learning algorithms use multiple layers of neural networks to progressively extract higher-level features from raw input.",
    "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with interactions between computers and human language."
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
print("FINE-TUNING PIPELINE COMPLETE! 🎉")
print("="*60)

print("\nNext steps:")
print("  1. Evaluate on Mr. TyDi dev set for retrieval metrics (MRR@10, etc.)")
print("  2. Test end-to-end retrieval with both encoders")
print("  3. Compare inference speed: teacher vs student")
print("  4. Compare with pre-training distillation results")
print("  5. Test on multiple languages if using multilingual models")
print(f"\nFine-tuned models ready at:")
print(f"  - Query: {query_output_path}")
print(f"  - Context: {context_output_path}")

# Full Mr. TyDi evaluation with BOTH fine-tuned encoders
import numpy as np
from typing import List

print("Loading Mr. TyDi dev set...")
try:
    mrtydi_dev = load_dataset("castorini/mr-tydi", LANGUAGE, split="dev", streaming=True)
except Exception as e:
    print(f"❌ Error loading Mr. TyDi dev set: {e}")
    raise SystemExit("Evaluation requires Mr. TyDi dev set.")

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
print("EVALUATING FINE-TUNED DPR SYSTEM ON MR. TYDI DEV SET")
print("="*60 + "\n")

MAX_QUERIES = 1000
teacher_mrr, student_mrr = [], []
teacher_precisions, teacher_recalls = [], []
student_precisions, student_recalls = [], []

count = 0

for example in mrtydi_dev:
    if count >= MAX_QUERIES:
        break

    query = example.get("query", "")
    positive_passages = example.get("positive_passages", [])
    negative_passages = example.get("negative_passages", [])
    
    if not query or not positive_passages:
        continue

    # Combine positive and negative passages for ranking
    all_passages = []
    relevance_labels = []
    
    # Add positive passages
    for pos_passage in positive_passages:
        passage_text = pos_passage.get("text", "")
        if passage_text:
            all_passages.append(passage_text)
            relevance_labels.append(True)
    
    # Add negative passages (up to 10 for evaluation)
    for neg_passage in negative_passages[:10]:
        passage_text = neg_passage.get("text", "")
        if passage_text:
            all_passages.append(passage_text)
            relevance_labels.append(False)
    
    if len(all_passages) == 0:
        continue

    # Encode query with query encoder
    teacher_query_emb = teacher_query.encode(query, convert_to_tensor=True)
    student_query_emb = student_query.encode(query, convert_to_tensor=True)

    # Encode passages with context encoder
    teacher_passage_embs = teacher_context.encode(all_passages, convert_to_tensor=True)
    student_passage_embs = student_context.encode(all_passages, convert_to_tensor=True)

    # Compute cosine similarities
    teacher_scores = F.cosine_similarity(teacher_query_emb.unsqueeze(0), teacher_passage_embs).cpu().numpy()
    student_scores = F.cosine_similarity(student_query_emb.unsqueeze(0), student_passage_embs).cpu().numpy()

    # Rank passages
    teacher_ranking = np.argsort(-teacher_scores)
    student_ranking = np.argsort(-student_scores)

    # Find first relevant passage rank
    teacher_first_relevant = float('inf')
    student_first_relevant = float('inf')

    for rank, idx in enumerate(teacher_ranking, 1):
        if relevance_labels[idx]:
            teacher_first_relevant = rank
            break
    for rank, idx in enumerate(student_ranking, 1):
        if relevance_labels[idx]:
            student_first_relevant = rank
            break

    if teacher_first_relevant != float('inf'):
        teacher_mrr.append(teacher_first_relevant)
    if student_first_relevant != float('inf'):
        student_mrr.append(student_first_relevant)

    # Compute Precision@10 and Recall@10
    teacher_top_k = [relevance_labels[idx] for idx in teacher_ranking[:10]]
    student_top_k = [relevance_labels[idx] for idx in student_ranking[:10]]

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
print("FINE-TUNING EVALUATION RESULTS")
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
print("FINE-TUNING EVALUATION COMPLETE")
print("="*60)

print(f"\n• Fine-tuned student retains {student_mrr_score/teacher_mrr_score*100:.1f}% of teacher's MRR@10 performance.")
print(f"• Hybrid loss combines distillation knowledge + task-specific optimization.")
print(f"• Using separate query and context encoders for proper DPR architecture.")
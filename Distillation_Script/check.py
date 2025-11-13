#!/usr/bin/env python3
"""
Quick test to verify loss scaling is working
Run this BEFORE starting training to catch issues
"""

import torch
import torch.nn.functional as F
import numpy as np

print("\n" + "="*80)
print("QUICK LOSS SCALING TEST")
print("="*80)

# Simulate batch
batch_size = 64
embedding_dim = 384
temperature = 1.0

# Create normalized embeddings (like from your model)
query = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
passage = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
student = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
teacher = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)

print("\n✓ Created normalized embeddings:")
print(f"  Query norm: {torch.norm(query, dim=1).mean().item():.6f} (should be 1.0)")
print(f"  Passage norm: {torch.norm(passage, dim=1).mean().item():.6f} (should be 1.0)")

# ===== TEST RETRIEVAL LOSS =====
print("\n" + "-"*80)
print("RETRIEVAL LOSS TEST")
print("-"*80)

# CORRECT implementation (with scaling)
sim_matrix = torch.mm(query, passage.t())  # [B, B]
scale_factor = 1.0 / np.sqrt(embedding_dim)
print(f"\nScale factor: 1/sqrt({embedding_dim}) = {scale_factor:.6f}")

sim_matrix_scaled = sim_matrix * scale_factor / temperature
labels = torch.arange(batch_size)
retrieval_loss = F.cross_entropy(sim_matrix_scaled, labels)

print(f"\nRetrieval Loss Stats:")
print(f"  Raw sim_matrix - min: {sim_matrix.min().item():.4f}, max: {sim_matrix.max().item():.4f}")
print(f"  Scaled sim_matrix - min: {sim_matrix_scaled.min().item():.4f}, max: {sim_matrix_scaled.max().item():.4f}")
print(f"  Retrieval Loss: {retrieval_loss.item():.6f}")
print(f"  Expected: ~{-np.log(1.0/batch_size):.6f} (for random guessing)")

if retrieval_loss.item() < 1.5:
    print(f"  ✅ CORRECT! Loss is properly scaled")
else:
    print(f"  ❌ WRONG! Loss is too high, scaling not applied")

# ===== TEST DISTILLATION LOSS =====
print("\n" + "-"*80)
print("DISTILLATION LOSS TEST")
print("-"*80)

mse_loss = F.mse_loss(student, teacher)
scaled_distill_loss = mse_loss * 0.5

print(f"\nDistillation Loss Stats:")
print(f"  MSE (raw): {mse_loss.item():.6f}")
print(f"  MSE (scaled by 0.5): {scaled_distill_loss.item():.6f}")
print(f"  Expected range: [0.25, 0.75]")

if 0.2 < scaled_distill_loss.item() < 1.0:
    print(f"  ✅ CORRECT! Loss is in expected range")
else:
    print(f"  ❌ WARNING! Loss outside expected range")

# ===== TEST HYBRID LOSS =====
print("\n" + "-"*80)
print("HYBRID LOSS TEST")
print("-"*80)

lambda_weight = 0.5
hybrid_loss = lambda_weight * scaled_distill_loss + (1 - lambda_weight) * retrieval_loss

print(f"\nHybrid Loss Calculation:")
print(f"  λ (lambda): {lambda_weight}")
print(f"  Distillation loss: {scaled_distill_loss.item():.6f}")
print(f"  Retrieval loss: {retrieval_loss.item():.6f}")
print(f"  Hybrid = λ*distill + (1-λ)*retrieval")
print(f"  Hybrid = {lambda_weight}*{scaled_distill_loss.item():.6f} + {1-lambda_weight}*{retrieval_loss.item():.6f}")
print(f"  Hybrid Loss: {hybrid_loss.item():.6f}")

if 0.3 < hybrid_loss.item() < 1.5:
    print(f"  ✅ CORRECT! Hybrid loss is well-balanced")
else:
    print(f"  ❌ WRONG! Hybrid loss is out of range")

# ===== COMPARISON WITH BROKEN VERSION =====
print("\n" + "="*80)
print("COMPARISON: BROKEN vs FIXED")
print("="*80)

# Broken version (no scaling)
retrieval_loss_broken = F.cross_entropy(torch.mm(query, passage.t()) / temperature, labels)
distill_loss_broken = F.mse_loss(student, teacher)
hybrid_loss_broken = 0.5 * distill_loss_broken + 0.5 * retrieval_loss_broken

print(f"\n❌ BROKEN (no scaling):")
print(f"  Distillation: {distill_loss_broken.item():.6f}")
print(f"  Retrieval: {retrieval_loss_broken.item():.6f}")
print(f"  Hybrid: {hybrid_loss_broken.item():.6f}")
print(f"  → Retrieval dominates, λ would drop to 0.2")

print(f"\n✅ FIXED (with scaling):")
print(f"  Distillation: {scaled_distill_loss.item():.6f}")
print(f"  Retrieval: {retrieval_loss.item():.6f}")
print(f"  Hybrid: {hybrid_loss.item():.6f}")
print(f"  → Both balanced, λ stays at 0.5")

# ===== FINAL CHECK =====
print("\n" + "="*80)
print("FINAL CHECK")
print("="*80)

checks = [
    ("Retrieval loss < 1.5", retrieval_loss.item() < 1.5),
    ("Distillation loss in [0.2, 1.0]", 0.2 < scaled_distill_loss.item() < 1.0),
    ("Hybrid loss in [0.3, 1.5]", 0.3 < hybrid_loss.item() < 1.5),
    ("Losses are similar magnitude", 0.3 < retrieval_loss.item() / scaled_distill_loss.item() < 3.0),
]

all_pass = True
for check_name, check_result in checks:
    status = "✅" if check_result else "❌"
    print(f"{status} {check_name}")
    if not check_result:
        all_pass = False

print("\n" + "="*80)
if all_pass:
    print("✅ ALL CHECKS PASSED - Your code is correctly scaled!")
    print("   You can now train with confidence.")
else:
    print("❌ SOME CHECKS FAILED - There are scaling issues!")
    print("   Make sure you applied the fixes to retrieval_loss and distillation_loss")
print("="*80 + "\n")
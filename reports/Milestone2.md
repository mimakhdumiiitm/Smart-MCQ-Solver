# Milestone 2 Report – Transformer-Based Semantic Ranking

## Objective

The objective of Milestone 2 is to improve answer ranking using transformer-based semantic models. Unlike the lexical retrieval methods used in Phase 1 (TF-IDF, Word2Vec, SBERT), this milestone evaluates deep contextual language models capable of understanding semantic relationships between the question and answer options.

Two transformer-based approaches were implemented:

- **Zero-Shot Natural Language Inference (NLI) Ranker**
- **Transformer Embedding Similarity Ranker**

The generated score matrices are later reused by the ensemble model in subsequent milestones.

---

# Methodology

## 1. Zero-Shot NLI Ranker

The Zero-Shot model treats every answer option as a hypothesis and the question as the premise.

For every question:

1. Construct premise–hypothesis pairs.
2. Pass each pair through a pretrained NLI model.
3. Use the entailment probability as the score.
4. Rank answer choices according to their entailment confidence.

This approach requires **no additional training**, making it suitable for semantic reasoning tasks.

---

## 2. Transformer Embedding Ranker

The embedding-based approach computes dense sentence embeddings using a pretrained Sentence Transformer.

The workflow is:

1. Encode the question.
2. Encode each answer option.
3. Compute cosine similarity between the question embedding and every option embedding.
4. Rank options according to similarity.

This method captures semantic similarity beyond exact word matching.

---

## 3. Artifact Reuse

To reduce execution time, the pipeline first checks whether previously computed score files are available.

If found, the following cached artifacts are reused:

- `zs_val_scores.npy`
- `zs_test_scores.npy`
- `transformer_val_scores.npy`
- `transformer_test_scores.npy`

When cached artifacts are unavailable, the models generate the scores from scratch and save them for future executions.

This significantly reduces runtime during repeated experiments.

---

# Experimental Setup

| Component | Value |
|-----------|-------|
| GPU | 2 × Tesla T4 |
| Device | CUDA |
| Validation Samples | 400 |
| Test Samples | 500 |
| Artifact Reuse | Enabled |
| Experiment Tracking | Weights & Biases |

---

# Weights & Biases Integration

Three W&B runs were executed:

1. **Zero-Shot NLI**
   - Validation metrics logged
   - Cached prediction scores reused

2. **Transformer Embedding**
   - Validation metrics logged
   - Cached prediction scores reused

3. **Baseline Comparison**
   - Comparison table generated
   - Results logged for experiment tracking

Artifact reuse avoided recomputation while preserving experiment reproducibility.

---

# Results

## Zero-Shot NLI

| Metric | Value |
|---------|-------|
| MAP@3 | **0.5496** |
| Accuracy | **0.3825** |
| Macro F1 | **0.3803** |

The Zero-Shot approach achieved the strongest performance among the transformer-based models.

---

## Transformer Embedding

| Metric | Value |
|---------|-------|
| MAP@3 | **0.2762** |
| Accuracy | **0.1500** |
| Macro F1 | **0.1482** |

The embedding similarity approach captured semantic information but performed substantially below the Zero-Shot model.

---

# Overall Comparison

| Method | MAP@3 | Accuracy | Macro F1 |
|---------|-------|----------|----------|
| Zero-Shot NLI | **0.5496** | **0.3825** | **0.3803** |
| Transformer Embedding | **0.2762** | **0.1500** | **0.1482** |

---

# Observations

- Zero-Shot NLI significantly outperformed the embedding similarity approach.
- Cached `.npy` score files were successfully reused, avoiding repeated inference.
- W&B successfully tracked all experiments and generated comparison summaries.
- Transformer embeddings based only on cosine similarity were less effective than inference-based semantic reasoning.
- The generated score matrices are reused in later milestones for ensemble learning and Retrieval-Augmented Generation (RAG).

---

# Conclusion

Milestone 2 demonstrates the effectiveness of transformer-based semantic ranking for multiple-choice question answering.

Among the evaluated approaches, **Zero-Shot NLI** achieved the best overall performance with a **MAP@3 of 0.5496**, substantially outperforming the transformer embedding similarity model.

The implementation also incorporates artifact reuse and experiment tracking, making the pipeline efficient, reproducible, and suitable for integration into the subsequent ensemble and RAG stages of the Smart MCQ Solver.
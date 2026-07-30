# Milestone 5 Report: Ensemble Learning and Final Model Selection

**Course:** Smart MCQ Solver Challenge  
**Milestone:** 5 – Ensemble Learning and Final Model Selection  
**Environment:** Kaggle Notebook (GPU: Tesla T4 ×2)

---

# 1. Objective

The objective of Milestone 5 was to integrate the prediction scores produced by all previous models into a unified ensemble framework capable of selecting the best-performing prediction strategy.

Instead of relying on a single ranking model, this milestone combines multiple independent prediction sources, evaluates several ensemble methods, and automatically selects the best-performing ensemble based on validation MAP@3.

The pipeline also emphasizes reproducibility by automatically reusing previously generated artifacts and logging every experiment to Weights & Biases (W&B).

---

# 2. System Configuration

| Component | Value |
|-----------|-------|
| Platform | Kaggle |
| Device | CUDA |
| GPUs | 2 × Tesla T4 |
| GPU Memory | 15.6 GB each |
| Fine-tuning Backbone | microsoft/deberta-v3-base |
| LoRA Configuration | r = 16, α = 32 |
| Validation Samples | 400 |
| Test Samples | 500 |
| Evaluation Metric | MAP@3 |

---

# 3. Input Prediction Sources

The ensemble pipeline automatically discovered and loaded prediction score matrices generated during previous milestones.

| Model | Validation Shape | Test Shape |
|--------|-----------------:|-----------:|
| TF-IDF | (400, 5) | (500, 5) |
| Word2Vec | (400, 5) | (500, 5) |
| SBERT | (400, 5) | (500, 5) |
| RAG Semantic | (400, 5) | (500, 5) |
| Transformer | (400, 5) | (500, 5) |
| Zero-shot | (400, 5) | (500, 5) |
| Fine-tuned DeBERTa | (400, 5) | (500, 5) |
| Fine-tuned RoBERTa | (400, 5) | (500, 5) |

All score matrices contain ranking scores for the five answer options (A–E).

---

# 4. Artifact Discovery and Reuse

To avoid unnecessary computation, the pipeline automatically searches for previously generated score artifacts.

Search priority:

1. Kaggle artifact directory
2. Local output directory

During execution, all required artifacts were successfully loaded from the Kaggle artifact repository.

Loaded artifacts included:

- `tfidf_val_scores.npy`
- `tfidf_test_scores.npy`
- `w2v_val_scores.npy`
- `w2v_test_scores.npy`
- `sbert_val_scores.npy`
- `sbert_test_scores.npy`
- `rag_semantic_val.npy`
- `rag_semantic_test.npy`
- `transformer_val_scores.npy`
- `transformer_test_scores.npy`
- `zs_val_scores.npy`
- `zs_test_scores.npy`
- `ft_val_logits.npy`
- `ft_test_logits.npy`
- `roberta_val_logits.npy`
- `roberta_test_logits.npy`

If the final ensemble artifacts already exist, the pipeline skips the entire ensemble computation and directly loads:

- `ensemble_val.npy`
- `ensemble_test.npy`

This significantly reduces execution time during repeated experiments.

---

# 5. Individual Model Evaluation

Before building the ensemble, every model was evaluated independently.

For each model the following metrics were computed:

- Accuracy
- Precision
- Recall
- Macro F1 Score
- MAP@3

Top-1 predictions were used for classification metrics, while Top-3 predictions were used for MAP@3 evaluation.

---

# 6. Individual Model Performance

| Model | Accuracy | F1 Score | MAP@3 |
|--------|---------:|---------:|------:|
| TF-IDF | 0.2000 | 0.1950 | 0.3700 |
| Word2Vec | 0.2250 | 0.2182 | 0.3779 |
| SBERT | 0.2250 | 0.2244 | 0.3796 |
| Transformer | 0.2450 | 0.2445 | 0.4017 |
| Zero-shot | 0.2050 | 0.2035 | 0.3825 |
| Fine-tuned DeBERTa | 0.2150 | 0.2138 | 0.3725 |
| Fine-tuned RoBERTa | 0.1875 | 0.1789 | 0.3625 |
| **RAG Semantic** | **0.9875** | **0.9875** | **0.9938** |

Among all individual models, the Retrieval-Augmented Generation (RAG) semantic model achieved the highest validation performance with a MAP@3 score of **0.9938**.

---

# 7. Ensemble Methodology

Three different ensemble strategies were evaluated.

## 7.1 Weighted Score Fusion

Each model's prediction scores were calibrated using temperature scaling.

The validation MAP@3 of each model was then used to optimize ensemble weights.

The optimized weights were:

| Model | Weight |
|--------|-------:|
| TF-IDF | 0.0123 |
| Word2Vec | 0.0488 |
| SBERT | 0.0288 |
| **RAG Semantic** | **0.7351** |
| Transformer | 0.0205 |
| Zero-shot | 0.0391 |
| DeBERTa FT | 0.0649 |
| RoBERTa FT | 0.0505 |

As expected, the strongest-performing model (RAG Semantic) received the largest contribution.

---

## 7.2 Rank Averaging

Each model independently ranked the five answer options.

The final ranking was obtained by averaging option ranks across all models.

---

## 7.3 Soft Voting

Each model contributed normalized prediction scores.

The final prediction was obtained by averaging probabilities across all models.

---

# 8. Ensemble Comparison

The three ensemble strategies were evaluated on the validation set.

| Ensemble Strategy | MAP@3 |
|-------------------|-------:|
| **Weighted Score Fusion** | **0.9617** |
| Rank Averaging | 0.5321 |
| Soft Voting | 0.5367 |

The weighted score fusion strategy achieved the highest overall ensemble performance and was automatically selected as the final prediction method.

---

# 9. Weights & Biases Integration

A dedicated W&B experiment run was created for every individual model.

Each run logged:

- Accuracy
- Precision
- Recall
- Macro F1 Score
- MAP@3
- Per-class precision
- Per-class recall
- Per-class F1 score
- Score distribution histograms

A final W&B run was created for the selected ensemble.

The ensemble run additionally logged:

- Best ensemble strategy
- Ensemble comparison table
- Final validation metrics
- Ensemble artifacts

This enables complete experiment tracking and comparison across all models.

---

# 10. Final Results

## Individual Model MAP@3

| Model | MAP@3 |
|--------|------:|
| TF-IDF | 0.3700 |
| Word2Vec | 0.3779 |
| SBERT | 0.3796 |
| Transformer | 0.4017 |
| Zero-shot | 0.3825 |
| Fine-tuned DeBERTa | 0.3725 |
| Fine-tuned RoBERTa | 0.3625 |
| **RAG Semantic** | **0.9938** |

---

## Best Ensemble

| Metric | Value |
|--------|------:|
| Ensemble Strategy | Weighted Score Fusion |
| Accuracy | 0.9275 |
| Precision | 0.9302 |
| Recall | 0.9251 |
| Macro F1 | 0.9271 |
| MAP@3 | **0.9617** |

---

# 11. Artifact Generation

The final ensemble prediction matrices were saved for future inference.

Generated artifacts:

- `ensemble_val.npy`
- `ensemble_test.npy`

These artifacts are automatically reused in subsequent executions, eliminating redundant ensemble computation.

---

# 12. Key Achievements

- Successfully integrated prediction scores from eight different ranking models.
- Implemented automatic discovery and reuse of prediction artifacts.
- Evaluated every model using standardized classification and ranking metrics.
- Implemented three complementary ensemble strategies.
- Applied temperature scaling before weighted score fusion.
- Automatically optimized ensemble weights using validation performance.
- Selected the best ensemble strategy without manual intervention.
- Logged comprehensive experiments to Weights & Biases for both individual models and the final ensemble.
- Generated reusable ensemble prediction artifacts for future inference.
- Built a modular ensemble framework that can easily incorporate additional models.

---

# 13. Conclusion

Milestone 5 completes the Smart MCQ Solver pipeline by integrating all previously developed ranking models into a unified ensemble framework.

The system automatically discovers reusable prediction artifacts, evaluates each model independently, compares multiple ensemble strategies, and selects the best-performing approach using validation MAP@3. The weighted score fusion strategy produced the strongest ensemble performance (MAP@3 = **0.9617**) by assigning higher importance to better-performing models, particularly the Retrieval-Augmented Generation (RAG) semantic model.

The complete workflow is fully reproducible through artifact reuse and comprehensive Weights & Biases experiment tracking, providing a scalable foundation for future ensemble optimization and competition submissions.
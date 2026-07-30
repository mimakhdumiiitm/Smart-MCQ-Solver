# Milestone 4 Report: Transformer Fine-Tuning with LoRA

**Course:** Smart MCQ Solver Challenge  
**Milestone:** 4 – Transformer Fine-Tuning with LoRA  
**Environment:** Kaggle Notebook (GPU: Tesla T4 ×2)

---

# 1. Objective

The objective of Milestone 4 was to improve answer ranking by fine-tuning pretrained transformer models for the multiple-choice question answering task using Low-Rank Adaptation (LoRA).

Instead of training the entire transformer, LoRA updates only a small subset of trainable parameters while keeping the pretrained backbone frozen. This significantly reduces GPU memory usage and training time while maintaining strong performance.

Two transformer backbones were evaluated:

- DeBERTa-v3 Base
- RoBERTa Base

Their prediction logits were also combined through an ensemble to evaluate whether combining multiple fine-tuned models improves ranking performance.

---

# 2. System Configuration

| Component | Value |
|-----------|-------|
| Platform | Kaggle |
| Device | CUDA |
| GPUs | 2 × Tesla T4 |
| GPU Memory | 15.6 GB each |
| Fine-Tuning Method | LoRA (PEFT) |
| Primary Backbone | microsoft/deberta-v3-base |
| Secondary Backbone | roberta-base |
| LoRA Rank (r) | 16 |
| LoRA Alpha (α) | 32 |
| Number of Choices | 5 |
| Sequence Length | 512 |
| Training Samples | 2000 |
| Validation Samples | 400 |
| Test Samples | 500 |
| Epochs | 3 |

---

# 3. Methodology

The Milestone 4 pipeline consists of five major stages.

## Stage 1 — Dataset Preparation

The Smart MCQ dataset was converted into the Hugging Face Multiple Choice format.

Each training example contains:

- Question
- Five answer options
- Correct answer label

The generated datasets were:

```
Training Samples   : 2000
Validation Samples : 400
Test Samples       : 500
```

Each batch has the following tensor shape:

```
Input IDs Shape

(batch_size, 5, 512)
```

where:

- batch dimension
- five answer options
- maximum sequence length of 512 tokens

---

## Stage 2 — LoRA Fine-Tuning

Instead of updating all transformer parameters, LoRA inserts trainable low-rank matrices into attention layers.

Configuration:

```
Rank (r)     : 16
Alpha (α)    : 32
Epochs       : 3
```

### Primary Model

```
microsoft/deberta-v3-base
```

Trainable Parameters

```
2,679,553
```

Total Parameters

```
187,102,466
```

Trainable Percentage

```
1.4321%
```

---

### Secondary Model

```
roberta-base
```

Trainable Parameters

```
590,593
```

Total Parameters

```
125,236,994
```

Trainable Percentage

```
0.4716%
```

Only the LoRA adapters were optimized while the pretrained backbone remained frozen.

---

## Stage 3 — Validation Prediction

After fine-tuning, both transformer models generated logits for:

- Validation set
- Test set

Generated logits:

```
DeBERTa Validation Logits : (400, 5)
DeBERTa Test Logits       : (500, 5)

RoBERTa Validation Logits : (400, 5)
RoBERTa Test Logits       : (500, 5)
```

---

## Stage 4 — Ensemble Prediction

The validation and test logits from both transformer models were combined to form an ensemble prediction.

Generated logits:

```
Ensemble Validation Logits : (400, 5)
Ensemble Test Logits       : (500, 5)
```

The ensemble prediction was evaluated alongside the individual transformer models.

---

## Stage 5 — Model Comparison

Three prediction strategies were compared:

- DeBERTa Fine-Tuned
- RoBERTa Fine-Tuned
- Ensemble

The comparison was automatically logged to Weights & Biases.

---

# 4. Artifact Reuse

To avoid unnecessary retraining, the pipeline automatically searches for previously generated transformer artifacts.

Search priority:

1. Kaggle artifact directory
2. Local output directory
3. Train from scratch

The following artifacts are reused whenever available:

- `ft_val_logits.npy`
- `ft_test_logits.npy`
- `roberta_val_logits.npy`
- `roberta_test_logits.npy`

If artifacts are unavailable, the models are trained from scratch and the generated logits are saved for future experiments.

This significantly reduces execution time for repeated runs.

---

# 5. Experimental Results

## Hardware

```
Device : CUDA
GPUs   : 2 × Tesla T4
GPU Memory : 15.6 GB each
```

---

## Primary Transformer

```
microsoft/deberta-v3-base
```

---

## Secondary Transformer

```
roberta-base
```

---

## LoRA Configuration

```
Rank (r)  : 16
Alpha (α) : 32
Epochs    : 3
```

---

## Dataset Size

```
Training   : 2000
Validation : 400
Test       : 500
```

---

# 6. Training Summary

## DeBERTa-v3 Base

| Epoch | Training Loss | Validation Loss | MAP@3 | Accuracy |
|-------:|--------------:|----------------:|------:|---------:|
| 1 | 12.159180 | 2.917969 | 0.725417 | 0.607500 |
| 2 | 10.654687 | 2.548828 | 0.704583 | 0.567500 |
| 3 | 9.277832 | 2.427734 | 0.696667 | 0.557500 |

Training Runtime

```
1575.57 seconds
```

---

## RoBERTa Base

| Epoch | Training Loss | Validation Loss | MAP@3 | Accuracy |
|-------:|--------------:|----------------:|------:|---------:|
| 1 | 12.866060 | 3.217368 | 0.540833 | 0.367500 |
| 2 | 12.857254 | 3.216100 | 0.638333 | 0.482500 |
| 3 | 12.851961 | 3.215462 | 0.675417 | 0.532500 |

Training Runtime

```
859.90 seconds
```

---

# 7. Model Comparison

| Model | MAP@3 |
|--------|-------:|
| DeBERTa Fine-Tuned | **0.3725** |
| RoBERTa Fine-Tuned | 0.3625 |
| Ensemble | 0.3700 |

---

# 8. Discussion

### DeBERTa Fine-Tuning

The LoRA fine-tuned DeBERTa model achieved the highest validation performance with a MAP@3 score of **0.3725**. Among the evaluated models, it provided the strongest ranking quality and served as the best standalone transformer.

### RoBERTa Fine-Tuning

The LoRA fine-tuned RoBERTa model achieved a validation MAP@3 of **0.3625**. Although its performance was slightly lower than DeBERTa, it demonstrated competitive ranking capability while requiring fewer trainable parameters.

### Ensemble

The ensemble of DeBERTa and RoBERTa achieved a validation MAP@3 of **0.3700**. While combining both models produced stable predictions, it did not outperform the best individual model (DeBERTa) on the validation dataset.

---

# 9. Key Achievements

- Successfully implemented parameter-efficient transformer fine-tuning using LoRA.
- Fine-tuned two pretrained transformer backbones for multiple-choice question answering.
- Reduced trainable parameters to approximately 1.43% for DeBERTa and 0.47% for RoBERTa.
- Generated reusable validation and test logits for both models.
- Implemented automatic artifact reuse to avoid redundant computation.
- Saved reusable logits for future ensemble experiments.
- Logged training metrics, model artifacts, and comparison tables using Weights & Biases.
- Compared individual transformer models with an ensemble approach.
- Achieved the best validation MAP@3 of **0.3725** using the fine-tuned DeBERTa model.

---

# 10. Conclusion

Milestone 4 successfully extends the Smart MCQ Solver by introducing parameter-efficient transformer fine-tuning using LoRA. Two pretrained transformer backbones, DeBERTa-v3 Base and RoBERTa Base, were adapted for the multiple-choice ranking task while training only a small fraction of the total model parameters.

The pipeline includes automated dataset preparation, LoRA-based fine-tuning, validation and test logit generation, artifact reuse, model checkpointing, Weights & Biases integration, and ensemble evaluation. Experimental results show that the fine-tuned DeBERTa model achieved the highest validation performance with a MAP@3 score of **0.3725**, while the ensemble produced competitive but slightly lower performance. The modular design enables efficient experimentation and provides a strong foundation for future improvements in answer ranking.
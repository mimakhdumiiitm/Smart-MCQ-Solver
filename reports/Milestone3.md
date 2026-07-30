# Milestone 3 Report: Retrieval-Augmented Generation (RAG)

**Course:** Smart MCQ Solver Challenge  
**Milestone:** 3 – Retrieval-Augmented Generation (RAG)  
**Environment:** Kaggle Notebook (GPU: Tesla T4 ×2)

---

# 1. Objective

The objective of Milestone 3 was to improve multiple-choice answer ranking by incorporating Retrieval-Augmented Generation (RAG). Rather than relying solely on semantic similarity between the question and answer options, this milestone retrieves the most relevant training examples and uses the retrieved knowledge to improve answer prediction.

The implementation combines dense retrieval using Sentence-BERT with a FAISS vector index and introduces two complementary scoring methods:

- Retrieval Vote Scoring
- Semantic Context Scoring

The effectiveness of each approach was evaluated individually and in combination using an ablation study.

---

# 2. System Configuration

| Component | Value |
|-----------|-------|
| Platform | Kaggle |
| Device | CUDA |
| GPUs | 2 × Tesla T4 |
| GPU Memory | 15.6 GB each |
| Retrieval Model | sentence-transformers/all-mpnet-base-v2 |
| Vector Index | FAISS |
| Embedding Dimension | 768 |
| Retrieval Method | Cosine Similarity |
| Training Corpus Size | 2,000 Questions |
| Top-K Retrieval | 5 |

---

# 3. Methodology

The RAG pipeline consists of four major stages.

## Stage 1 — Dense Embedding Generation

All training questions were encoded using the pretrained Sentence-BERT model:

**Model**
```
sentence-transformers/all-mpnet-base-v2
```

Each question was converted into a 768-dimensional dense embedding.

The embeddings were normalized so that cosine similarity could be computed efficiently using inner-product search.

---

## Stage 2 — FAISS Index Construction

After generating embeddings, a FAISS similarity index was created.

Execution log:

```
Embedding Dimension : 768
Training Samples    : 2000
FAISS Index         : CPU
Vectors Indexed     : 2000
```

Although two Tesla T4 GPUs were available, GPU FAISS was unavailable in the Kaggle environment, so the pipeline automatically switched to the CPU implementation without affecting correctness.

---

## Stage 3 — Retrieval-Augmented Scoring

For every validation and test question:

1. Encode the question
2. Retrieve the Top-5 most similar training questions
3. Generate two independent score matrices

### 3.1 Retrieval Vote Score

Each retrieved example contributes a weighted vote according to its cosine similarity.

Questions with higher similarity contribute larger weights.

The accumulated votes become ranking scores for options:

```
A
B
C
D
E
```

---

### 3.2 Semantic Context Score

Instead of voting directly, the retrieved correct answers are collected.

The pipeline:

- extracts the correct answer text
- embeds those answers
- computes their mean embedding
- compares every current answer option with this semantic context

This produces a semantic similarity score for every option.

---

### 3.3 Combined RAG Score

The final RAG score is computed by combining both retrieval strategies:

```
Combined Score =
Vote Score +
Semantic Context Score
```

This allows retrieval evidence and semantic similarity to complement one another.

---

# 4. Retrieval Context Generation

The pipeline also generates retrieval contexts for every question.

Each context contains the most relevant retrieved question-answer pairs in the format:

```
Q: Retrieved Question
A: Retrieved Correct Answer
```

Example:

> Q: What is the relationship between mass, force, and acceleration according to Newton's laws of motion?  
> A: Mass is an inertial property...

These contexts can later be used for:

- Prompt augmentation
- Retrieval-based prompting
- Fine-tuning
- Generative models

---

# 5. Artifact Reuse

To avoid unnecessary computation, the pipeline automatically checks for previously generated RAG artifacts.

Search priority:

1. Kaggle artifact directory
2. Local output directory
3. Compute from scratch

The following artifacts are reused whenever available:

- `rag_vote_val.npy`
- `rag_vote_test.npy`
- `rag_semantic_val.npy`
- `rag_semantic_test.npy`

If artifacts are unavailable, the pipeline computes them and saves them for future runs.

This significantly reduces execution time for repeated experiments.

---

# 6. Experimental Results

## Hardware

```
Device : CUDA
GPUs   : 2 × Tesla T4
```

---

## Retrieval Model

```
Sentence-BERT
all-mpnet-base-v2
```

---

## Training Corpus

```
2000 Questions
```

---

## Embedding Dimension

```
768
```

---

## FAISS Index

```
CPU Index
2000 vectors indexed
```

---

# 7. Ablation Study

Three different retrieval strategies were evaluated.

| RAG Variant | MAP@3 |
|-------------|-------:|
| Retrieval Vote Only | **1.0000** |
| Semantic Context Only | **0.9938** |
| Combined RAG | **1.0000** |

---

# 8. Discussion

### Retrieval Vote Scoring

The retrieval voting strategy achieved a perfect MAP@3 score of **1.0000** on the validation dataset. This indicates that the retrieved neighbors consistently provided strong evidence for identifying the correct answer.

### Semantic Context Scoring

Semantic context scoring achieved a MAP@3 of **0.9938**, demonstrating that the semantic representations of retrieved correct answers were highly informative, although slightly less effective than direct vote aggregation.

### Combined Scoring

Combining vote-based retrieval with semantic context also achieved a perfect MAP@3 score of **1.0000**. While no additional improvement over vote-only scoring was observed on the validation split, the combined approach provides complementary information and offers a more robust retrieval framework for future extensions.

---

# 9. Key Achievements

- Successfully implemented a complete Retrieval-Augmented Generation (RAG) pipeline.
- Built a dense retrieval system using Sentence-BERT embeddings.
- Constructed a FAISS index for efficient nearest-neighbor search.
- Implemented retrieval vote scoring based on similarity-weighted answer aggregation.
- Developed semantic context scoring using retrieved correct-answer embeddings.
- Generated reusable retrieval contexts for downstream prompt augmentation.
- Added automatic artifact reuse to avoid redundant computation.
- Saved reusable RAG score artifacts for future experiments.
- Performed an ablation study comparing different retrieval strategies.
- Achieved perfect MAP@3 performance using retrieval voting and combined RAG scoring on the validation dataset.

---

# 10. Conclusion

Milestone 3 successfully extends the Smart MCQ Solver by integrating retrieval-based reasoning into the prediction pipeline. Instead of relying solely on semantic similarity between questions and answer options, the system leverages knowledge from similar training examples to improve answer ranking.

The combination of Sentence-BERT embeddings, FAISS retrieval, retrieval vote aggregation, and semantic context scoring provides a modular and extensible RAG framework. Automatic artifact reuse further improves efficiency by minimizing redundant computation, making the pipeline suitable for iterative experimentation and future enhancements.
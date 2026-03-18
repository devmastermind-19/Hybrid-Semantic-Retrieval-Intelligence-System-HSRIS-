HSRIS — Hybrid Semantic Retrieval & Intelligence System
A customer support ticket retrieval engine built from scratch using PyTorch + NumPy only — no scikit-learn.

What it does
Given a new support ticket, HSRIS surfaces the most similar historical tickets by fusing two retrieval signals:

Sparse (TF-IDF) — keyword overlap, stored as torch.sparse_coo_tensor
Dense (GloVe 300-d) — semantic meaning via TF-IDF weighted word vector averaging
Hybrid score — α × TF-IDF + (1 − α) × GloVe, α tunable at runtime


Dataset
Customer Support Ticket Dataset · ~8,470 records
Focus columns: Ticket Subject, Ticket Description, Ticket Priority, Ticket Type, Ticket Channel

Stack
ComponentImplementationTokenizerCustom regex pipelineVocabularyTop-5000 tokens, bigrams & trigramsTF-IDFManual TF · smoothed IDF · torch.sparse_coo_tensorEncodersLabelEncoderCustom (priority) · OneHotEncoderCustom (channel)EmbeddingsGloVe 6B 300-d via nn.Embedding (frozen)SimilarityCosine via pre-normalised dot product on GPUMulti-GPUnn.DataParallel across dual T4sInterfaceGradio app with alpha slider

Quickstart
python# Run on Kaggle — Dual T4 x2, internet enabled
# Dataset mounted at /kaggle/input/customer-support-ticket-dataset/

results = retrieve_tickets(
    query="I cannot login and my password reset email never arrived",
    alpha=0.4,   # 0 = pure semantic | 1 = pure keyword
    top_k=5
)

Evaluation
Precision@5 over 500 sampled queries, relevance = matching Ticket Type:
MethodMean P@5TF-IDF only (α=1.0)~0.62GloVe only (α=0.0)~0.58Hybrid (α=0.4)~0.71

Project structure
notebook.ipynb

├── Section 1  — Environment setup & GPU config

├── Section 2  — Dataset loading

├── Section 3  — Custom preprocessing pipeline

├── Section 4  — Tokenization & vocabulary (N-grams)

├── Section 5  — Categorical encoders (label + one-hot)

├── Section 6  — TF-IDF + sparse tensor storage

├── Section 7  — GloVe embeddings + weighted sentence vectors

├── Section 8  — Hybrid retrieval function

├── Section 9  — Dual GPU benchmarking

├── Section 10 — Precision@5 evaluation

├── Section 11 — Visualization (comparison plots)

└── Section 12 — Gradio deployment interface

Platform

Kaggle Notebook · Dual NVIDIA T4 x2 · Python 3.10 · PyTorch 2.x

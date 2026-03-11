# Comprehensive Literature Review: Mechanistic Interpretability for Time Series Foundation Models (TSFMs)

**Last Updated:** March 11, 2026
**Purpose:** Complete literature review for building a NotebookLLM library for consultation on SOTA papers, implementation, and methodologies.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Survey and Review Papers](#survey-and-review-papers)
3. [Core Time Series Foundation Models](#core-time-series-foundation-models)
4. [Mechanistic Interpretability for TSFMs](#mechanistic-interpretability-for-tsfms)
5. [Interpretability Methodologies and Techniques](#interpretability-methodologies-and-techniques)
6. [Implementation Resources](#implementation-resources)
7. [Key Architectures and Paradigms](#key-architectures-and-paradigms)
8. [Benchmarks and Datasets](#benchmarks-and-datasets)
9. [Open Problems and Future Directions](#open-problems-and-future-directions)
10. [References](#references)

---

## 1. Introduction

### What are Time Series Foundation Models (TSFMs)?

Time Series Foundation Models represent a paradigm shift from traditional task-specific models to general-purpose, pre-trained models capable of zero-shot or few-shot learning across diverse time series tasks. These models leverage large-scale pre-training on massive time series datasets to learn universal temporal representations.

### Key Characteristics:
- **Zero-shot capabilities**: Ability to forecast on unseen datasets without task-specific training
- **Transfer learning**: Pre-trained representations applicable across domains
- **Multi-task learning**: Single model handling forecasting, classification, anomaly detection, etc.
- **Scalability**: Built on transformer architectures scaled to billions of parameters

### The Interpretability Challenge

While TSFMs achieve impressive performance, they introduce a **"transparency debt"** - operating as opaque black boxes where standard attribution methods fail. This creates critical challenges for:
- Safety-critical applications (healthcare, finance, manufacturing)
- Regulatory compliance
- Debugging and model improvement
- Scientific understanding

---

## 2. Survey and Review Papers

### Primary Survey Papers

#### **Foundation Models for Time Series Analysis: A Tutorial and Survey**
- **arXiv:** [2403.14735](https://arxiv.org/abs/2403.14735)
- **Conference:** KDD 2024
- **Authors:** Multiple contributors
- **Key Contributions:**
  - Comprehensive consolidation of FMs for time series
  - Theoretical underpinnings and methodological frameworks
  - Taxonomy of existing approaches
  - Tutorial website: [FM4TS](https://sites.google.com/view/fm4ts/home)

#### **Empowering Time Series Analysis with Foundation Models**
- **arXiv:** [2405.02358](https://ui.adsabs.harvard.edu/abs/2024arXiv240502358Y/abstract)
- **Focus:** Modality-aware, challenge-oriented perspective
- **Key Insight:** Examines FMs pre-trained on different data types for time series

#### **Foundation Models for Time Series: A Survey**
- **arXiv:** [2504.04011v1](https://arxiv.org/html/2504.04011v1)
- **Contribution:** Novel taxonomy for categorizing pre-trained foundation models
- **Coverage:** From pre-training to post-training approaches

#### **A Survey on Time Series Foundation Models**
- **ResearchGate:** [Link](https://www.researchgate.net/publication/400275994_From_Pre-training_to_Post-training_A_Survey_on_Time_Series_Foundation_Models)
- **Focus:** Complete journey from pre-training to post-training

### Tutorial Resources

1. **[FM4TS Tutorial Website](https://wenhaomin.github.io/FM4TS.github.io/)** - Official companion site
2. **[Foundation Models for Time Series: Theory, Algorithms, and Applications](https://fm4ts.netlify.app/)** - Comprehensive review
3. **[KDD 2024 Video Abstract](https://www.youtube.com/watch?v=HvqFUmksd_M)** - Conference presentation

---

## 3. Core Time Series Foundation Models

### 3.1 TimeGPT-1

**The First Foundation Model for Time Series**

- **Paper:** [TimeGPT-1](https://arxiv.org/abs/2310.03589)
- **Organization:** Nixtla
- **Year:** 2023 (August - milestone announcement)
- **GitHub:** [Nixtla/nixtla](https://github.com/Nixtla/nixtla)

**Key Features:**
- First generative pre-trained transformer specifically for time series
- Production-ready deployment
- Zero-shot forecasting across diverse domains (retail, finance, healthcare, etc.)
- Pre-trained on 100 billion data points
- Transformer-based architecture adapted for temporal data

**Benchmarking:**
- [Benchmarking TimeGPT for Real-World Forecasting Applications](https://www.researchgate.net/publication/398033397_Benchmarking_a_Time-Series_Foundation_Model_TimeGPT_for_Real-World_Forecasting_Applications) (November 2025)

**Resources:**
- [Nixtla Documentation](https://www.nixtla.io/docs/introduction/introduction)
- [Medium Article](https://medium.com/the-forecaster/timegpt-the-first-foundation-model-for-time-series-forecasting-bf0a75e63b3a)

---

### 3.2 TimesFM (Google)

**A Decoder-Only Foundation Model for Time-Series Forecasting**

- **Paper:** [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)
- **Organization:** Google Research
- **Conference:** ICML 2024
- **GitHub:** [google-research/timesfm](https://github.com/google-research/timesfm)

**Key Features:**
- Decoder-only transformer architecture (inspired by LLMs)
- Pre-trained on **100 billion real-world time-points**
- Training data sources:
  - Google Trends
  - Wikipedia Pageviews
  - Synthetic data
- Zero-shot forecasting capabilities
- Performance competitive with state-of-the-art supervised models

**Resources:**
- [Google Research Blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [PMLR Proceedings](https://proceedings.mlr.press/v235/das24c.html)

**Architecture Highlights:**
- Similar to GPT-style decoder-only models
- Adapted for continuous time series data
- Patch-based input processing

---

### 3.3 Chronos (Amazon)

**Learning the Language of Time Series**

- **Paper:** [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)
- **Organization:** Amazon Web Services + UC San Diego + collaborators
- **Year:** 2024
- **GitHub:** [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)

**Key Innovation:**
- **Tokenization approach:** Transforms continuous time series into discrete tokens
- Adapts existing language model architectures (T5 family)
- Model sizes: 20M to 710M parameters
- Zero-shot probabilistic forecasting

**Chronos-Bolt (Improved Version):**
- 5% more accurate (lower error)
- Up to **250x faster** inference
- **20x more memory efficient**
- [AWS Blog](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/)

**Resources:**
- [Amazon Science Page](https://www.amazon.science/code-and-datasets/chronos-learning-the-language-of-time-series)
- [OpenReview](https://openreview.net/forum?id=gerNCVqqtR)

**Methodology:**
1. Time series values → Tokenization → Discrete tokens
2. Process with language model architecture
3. De-tokenization → Probabilistic forecasts

---

### 3.4 MOMENT

**A Family of Open Time-Series Foundation Models**

- **Paper:** [MOMENT: A Family of Open Time-Series Foundation Models](https://raw.githubusercontent.com/mlresearch/v235/main/assets/goswami24a/goswami24a.pdf)
- **Authors:** Mononito Goswami, Konrad Szafer, Arjun Choudhry, et al.
- **Conference:** ICML 2024

**Key Features:**
- High-capacity transformer models
- Pre-trained using **masked time series prediction tasks**
- Large-scale time series data for pre-training
- Open-source and accessible

**Architecture:**
- Transformer-based encoder
- Masked autoencoding objective (similar to BERT)
- Focus on learning general temporal representations

**Note:** This is the model used in TimeSAE interpretability research.

---

### 3.5 Time-LLM

**Time Series Forecasting by Reprogramming Large Language Models**

- **Paper:** [arXiv:2310.01728](https://arxiv.org/abs/2310.01728)
- **Conference:** ICLR 2024
- **Authors:** Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, et al.
- **GitHub:** [KimMeen/Time-LLM](https://github.com/KimMeen/Time-LLM)

**Key Innovation:**
- **Reprogramming framework** for LLMs to handle time series
- **Keeps backbone LLM frozen** (no modifications to pre-trained models)
- Transforms time series into text prototypes
- Uses prompt enhancement for context understanding

**Methodology:**
1. Time series → Text prototype transformation
2. Prompt engineering with domain context
3. Process through frozen LLM
4. Output → Time series prediction

**Resources:**
- [ICLR 2024 Paper (PDF)](https://proceedings.iclr.cc/paper_files/paper/2024/file/680b2a8135b9c71278a09cafb605869e-Paper-Conference.pdf)
- [OpenReview](https://openreview.net/forum?id=Unb5CVPtae)
- [Presentation Slides](https://mingjin.dev/other/iclr-24-jin-slides-mllm-talk.pdf)

---

### 3.6 UniTS

**A Unified Multi-Task Time Series Model**

- **Paper:** [arXiv:2403.00131](https://arxiv.org/abs/2403.00131)
- **Conference:** NeurIPS 2024
- **Organization:** Harvard Medical School (Zitnik Lab)
- **GitHub:** [mims-harvard/UniTS](https://github.com/mims-harvard/UniTS)

**Key Innovation:**
- **Universal Task Formulation:** Single model for all time series tasks
- **Task Tokenization:** Expresses predictive and generative tasks uniformly
- **No task-specific modules:** Shared parameters across all tasks
- **Transfer learning:** Captures universal time series representations

**Supported Tasks:**
- Forecasting
- Classification
- Anomaly detection
- Imputation
- Generation

**Resources:**
- [Project Page](https://zitniklab.hms.harvard.edu/projects/UniTS/)
- [NeurIPS 2024 Poster](https://neurips.cc/virtual/2024/poster/93709)
- [OpenReview](https://openreview.net/forum?id=nBOdYBptWW)

**Architecture:**
- Modified transformer blocks
- Task tokens as special tokens
- Universal task specification language

---

### 3.7 Lag-Llama

**Probabilistic Foundation Model for Time Series Forecasting**

- **Paper:** Mentioned in multiple surveys (detailed search limited by rate limits)
- **Focus:** Probabilistic forecasting with uncertainty quantification
- **Architecture:** Transformer-based with probabilistic output heads

**Key Features:**
- Probabilistic predictions (distribution forecasting)
- Uncertainty quantification
- Generalization to new datasets

*Note: Need to search for more detailed information on Lag-Llama by Rasul et al.*

---

### 3.8 TimeMAE / ExtraMAE

**Masked Autoencoder Approaches for Time Series**

#### **TimeMAE: Self-Supervised Representations of Time Series**
- **Source:** [ACM Digital Library](https://dl.acm.org/doi/10.1145/3773966.3778007)
- **Key Contribution:** Reformulates masked modeling for time series via semantic unit elevation

#### **ExtraMAE: Time Series Generation with Masked Autoencoder**
- **Paper:** [arXiv:2201.07006](https://arxiv.org/abs/2201.07006)
- **GitHub:** [Dolores2333/ExtraMAE](https://github.com/Dolores2333/ExtraMAE)
- **Key Feature:** Scalable self-supervised model for time series generation

#### **Related Masked Autoencoder Works:**
- **TS-MAE:** [Masked Autoencoder for Time Series Representation Learning](https://www.sciencedirect.com/science/article/abs/pii/S0020025524014907)
- **MTS-MAE:** Masked Autoencoders for Multivariate Time-Series Forecasting
- **TFMAE:** [Temporal-Frequency Masked Autoencoders (ICDE 2024)](https://github.com/LMissher/TFMAE)
- **VisionTS:** [Visual Masked Autoencoders for Zero-Shot Time Series Forecasting](https://openreview.net/forum?id=5DSj3MfWrB)

---

### 3.9 TTM (Tiny Time Mixer)

**IBM's Efficient Foundation Model**

- **Organization:** IBM
- **Focus:** Lightweight, efficient time series foundation model
- **Part of:** IBM Granite TSFM ecosystem

**Resources:**
- [IBM Granite TSFM GitHub](https://github.com/ibm-granite/granite-tsfm)
- Public notebooks and serving components
- Open-source implementation

*Note: Need more detailed search for TTM-specific papers*

---

### 3.10 PatchTST

**Transformer-based Time Series Forecasting with Patching**

- **Key Innovation:** Patch-based approach for time series transformers
- **Methodology:** Divides time series into patches (similar to Vision Transformer)
- **Benefits:**
  - Reduced computational complexity
  - Better local pattern capture
  - Improved long-range dependencies

*Note: PatchTST is foundational for many subsequent TSFMs. Need to search for detailed arXiv paper.*

---

## 4. Mechanistic Interpretability for TSFMs

This is the core section focusing on interpretability research specifically for time series foundation models.

### 4.1 TimeSAE: Mechanistic Interpretability for Time-Series Foundation Models

**The First Rigorous Application of SAEs to TSFMs**

- **Paper:** [OpenReview PDF](https://openreview.net/pdf?id=Ojd6YjHpyE)
- **Venue:** ICLR 2026 Workshop on Time Series in the Age of Large Models (TSALM)
- **Status:** Under double-blind review (Anonymous authors)

**Abstract:**
TimeSAE addresses the "transparency debt" in TSFMs like MOMENT-1-large by adapting Sparse Autoencoders (SAEs) to decompose dense representations into interpretable, monosemantic features.

**Key Contributions:**
1. **First rigorous application of SAEs to TSFMs**
2. **Verified protocol for latent activation steering in continuous domains**
3. **Open-source framework for transforming black-box forecasters into auditable systems**

**Methodology:**

1. **Sparse Decomposition:**
   - Uses Top-K SAE architecture
   - Maps dense activations x ∈ R^d_model to sparse latent z ∈ R^d_SAE
   - Expansion factor: d_SAE ≫ d_model
   - Achieves 99.9% sparsity ratio

2. **Causal Verification:**
   - Interventional Causal Shift (Δ_i)
   - Steering experiments: do(x ← x + αv_i)
   - Monotonic dose-response relationship validation

3. **Data Engineering:**
   - PHM 2018 Ion Mill Etching dataset
   - Z-score normalization with outlier clipping
   - Activation harvesting from MOMENT-1-large encoder (layer l* = 12)

**Architecture:**
```
Encoder: z = TopK(W_enc(x - b_dec) + b_enc, k)
Decoder: x̂ = W_dec * z + b_dec
```

Where:
- k = 32 (sparsity parameter)
- d = 32,768 (latent dimension)
- d_model = 1,024 (MOMENT embedding dimension)

**Results:**

| Metric | PCA (Baseline) | TimeSAE | Improvement |
|--------|----------------|---------|-------------|
| Latent Dimension | 32 | 32,768 | +3100% |
| Active Neurons | 32 (Dense) | 32 (Sparse) | 0% |
| Sparsity Ratio | 0.0% | 99.9% | +99.9% |
| Reconstruction R² | 0.6902 | 0.79 | +14.5% |
| Feature Kurtosis | 4.88 | ≥ 20 | ≥ +310% |

**Key Findings:**
- **Linear Representation Hypothesis confirmed** for time series
- **Causal Steerability:** Predictable forecast control via latent intervention
- **Feature Discovery:** Isolates sparse physical primitives (e.g., cycles)
- **Monosemantic features:** High firing density (62.5%) across unsupervised features

**Automated Interpretation:**
- Point-biserial correlation for feature labeling
- "Etch Cycle Detectors" identified through correlation with ground truth
- Correlation threshold ρ_min for feature classification

**Limitations:**
1. Absence of semantic dictionary in time series (vs. NLP)
2. 32× latent expansion restricts to offline auditing
3. Linear Representation Hypothesis may underrepresent chaotic dynamics

**Future Directions:**
- Multi-layer circuit analysis
- Automated feature labeling
- Gated SAEs for non-linear probing

---

### 4.2 Mechanistic Interpretability for Transformer-based Time Series Classification

**Systematic Probing of Internal Causal Structures**

- **Paper:** [arXiv:2511.21514](https://arxiv.org/abs/2511.21514)
- **Authors:** Matīss Kalnāre, Sofoklis Kitharidis, Thomas Bäck, Niki van Stein
- **Date:** November 26, 2025

**Abstract:**
Addresses the gap in understanding transformer-based time series classification by adapting mechanistic interpretability techniques from NLP, including activation patching, attention saliency, and sparse autoencoders.

**Key Contributions:**
1. **Adaptation of mechanistic interpretability techniques** from NLP to time series
2. **Systematic probing** of attention heads and timesteps
3. **Causal graph construction** for information propagation
4. **Sparse autoencoder application** for interpretable latent features

**Methodologies Applied:**

1. **Activation Patching:**
   - Causal intervention on specific activations
   - Identifies critical components for classification

2. **Attention Saliency:**
   - Analyzes attention patterns
   - Identifies important temporal positions

3. **Sparse Autoencoders:**
   - Decomposes representations into interpretable features
   - Uncovers latent structure

**Experimental Setup:**
- Benchmark time series classification datasets
- Transformer architectures explicitly designed for time series
- Causal graph construction methodology

**Key Findings:**
- Reveals **causal structures** within transformer-based models
- Identifies **key attention heads** driving correct classifications
- Highlights **critical temporal positions** in the input
- Demonstrates potential of SAEs for uncovering interpretable latent features

**Significance:**
- Bridges gap between high-performance models and mechanistic auditability
- Provides methodological contributions to transformer interpretability
- Novel insights into functional mechanics of time series transformers

---

## 5. Interpretability Methodologies and Techniques

This section covers the broader interpretability toolkit applicable to TSFMs.

### 5.1 Sparse Autoencoders (SAEs)

**Decomposing Polysemantic Representations**

**Core Concept:**
- SAEs decompose dense neural network activations into sparse, monosemantic features
- Treat activations as superposition of interpretable concepts
- Act as "microscopes" for examining neural network representations

**Key Resources:**
- **Anthropic's Learning Roadmap:** [Anthropic 机械可解释性学习路线](https://juejin.cn/post/7577438119559266355)
- **Implementation:** [mechanistic-interpretability-saelens](https://ai.codefather.cn/skills/2014279386386608131)

**Architecture:**
```
Input: x ∈ R^d (dense activations)
Encoder: z = f(W_enc * x + b_enc)  where z is sparse
Decoder: x̂ = W_dec * z + b_dec
Loss: L = ||x - x̂||^2 + λ * sparsity_penalty(z)
```

**Variants:**
1. **Top-K SAE:** Hard sparsity constraint (k active features)
2. **JumpReLU SAE:** Threshold-based activation
3. **Standard SAE:** L1 regularization for sparsity

**Applications in TSFMs:**
- Feature discovery and labeling
- Causal intervention and steering
- Model auditing and debugging

**Theoretical Foundations:**
- **Linear Representation Hypothesis** (Elhage et al., 2022)
- **Superposition principle:** Dense representations encode many features in superposition
- **Monosemanticity:** Sparse features correspond to single interpretable concepts

**Key Papers:**
- **Anthropic's Toy Models of Superposition:** [Transformer Circuits Thread](https://transformer-circuits.pub/2022/toy_model/index.html)
- **Towards Monosemanticity:** [Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- **Sparse Autoencoders Find Highly Interpretable Features:** [Cunningham et al., ICLR 2024](https://openreview.net/forum?id=...)

---

### 5.2 Activation Patching / Causal Tracing

**Identifying Causal Mechanisms**

**Core Concept:**
- Intervene on specific activations to test their causal role
- "Patching" activations from one input to another
- Measure downstream effects on model behavior

**Methodology:**
1. **Run clean input:** Get baseline activations and output
2. **Run corrupted input:** Get corrupted activations
3. **Patch:** Replace specific corrupted activations with clean ones
4. **Measure:** Observe recovery of correct behavior

**Applications:**
- Identify critical attention heads
- Trace information flow through layers
- Validate causal importance of features

**Key Papers:**
- **Anthropic's Induction Heads Work:** Demonstrates activation patching for identifying induction circuits
- **Interpretability in the Wild:** Uses causal tracing to locate factual knowledge storage

**Relevance to TSFMs:**
- Identify which timesteps are critical for forecasting
- Understand temporal attention patterns
- Validate feature importance discovered through other methods

---

### 5.3 Attention Saliency and Visualization

**Understanding What the Model Attends To**

**Techniques:**
1. **Raw Attention Weights:** Visualize attention matrices
2. **Attention Rollout:** Aggregate attention across layers
3. **Gradient-based Saliency:** Attention weighted by gradients
4. **Attention Flow:** Model attention as flow network

**Applications in Time Series:**
- Identify which historical points influence predictions
- Understand seasonal pattern capture
- Detect lag dependencies
- Validate domain-specific temporal patterns

**Limitations:**
- Attention is not necessarily explanation (Jain & Wallace, 2019)
- Needs to be combined with causal methods for validity

---

### 5.4 Probing Classifiers

**Testing What Information is Encoded**

**Core Concept:**
- Train simple classifiers on model representations
- Test what linguistic/semantic properties are encoded
- Probe for specific features (e.g., trend, seasonality, anomalies)

**Methodology:**
1. Extract representations from specific layers
2. Define probe task (e.g., classify trend direction)
3. Train linear/non-linear classifier on representations
4. Evaluate probe performance

**Types of Probes:**
- **Linear probes:** Test linear separability
- **Non-linear probes:** Capture complex relationships
- **Structural probes:** Test for hierarchical structure

**Applications in TSFMs:**
- Test if representations encode trend/seasonality
- Probe for anomaly indicators
- Verify temporal dependency capture
- Validate learned physical dynamics

---

### 5.5 Integrated Gradients and Attribution Methods

**Input-Output Attribution**

**Standard Methods:**
1. **Integrated Gradients** (Sundararajan et al., 2017)
2. **SHAP** (Lundberg & Lee, 2017)
3. **LIME** (Ribeiro et al., 2016)
4. **Saliency Maps** (Simonyan et al., 2013)

**Limitations for TSFMs:**
- Focus on input-output relationships, not internal mechanisms
- Often provide "illusion of understanding" (Rudin, 2019)
- Fail randomization tests (Adebayo et al., 2018)
- Correlation vs. causation issue

**When to Use:**
- Initial exploration of model behavior
- Feature importance ranking
- Communication with non-technical stakeholders
- Complement to mechanistic methods

---

### 5.6 Logit Lens and Token Lens

**Interpreting Intermediate Predictions**

**Core Concept:**
- Apply output layer to intermediate representations
- See how predictions evolve through layers
- Identify when certain concepts emerge

**Applications in TSFMs:**
- Track forecast evolution through decoder layers
- Identify when specific patterns are recognized
- Understand temporal hierarchy in representations

---

## 6. Implementation Resources

### 6.1 Foundation Model Repositories

#### **TimeGPT**
- **GitHub:** [Nixtla/nixtla](https://github.com/Nixtla/nixtla)
- **Documentation:** [nixtla.io](https://www.nixtla.io/docs/introduction/introduction)
- **Type:** Production-ready API

#### **TimesFM (Google)**
- **GitHub:** [google-research/timesfm](https://github.com/google-research/timesfm)
- **Type:** Open-source, pre-trained model
- **Inference:** Zero-shot forecasting

#### **Chronos (Amazon)**
- **GitHub:** [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
- **Models:** Chronos and Chronos-Bolt
- **Type:** Open-source with pre-trained weights

#### **MOMENT**
- **Paper:** Available via ICML 2024 proceedings
- **Type:** Open time-series foundation model
- **Focus:** Masked time series prediction

#### **Time-LLM**
- **GitHub:** [KimMeen/Time-LLM](https://github.com/KimMeen/Time-LLM)
- **Type:** Reprogramming framework for existing LLMs
- **LLMs Supported:** LLaMA, GPT, etc.

#### **UniTS**
- **GitHub:** [mims-harvard/UniTS](https://github.com/mims-harvard/UniTS)
- **Type:** Unified multi-task model
- **Tasks:** Forecasting, classification, anomaly detection, imputation

#### **IBM Granite TSFM**
- **GitHub:** [ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm)
- **Contents:** Notebooks, utilities, serving components
- **Models:** TTM (Tiny Time Mixer) and others

#### **ExtraMAE**
- **GitHub:** [Dolores2333/ExtraMAE](https://github.com/Dolores2333/ExtraMAE)
- **Type:** Masked autoencoder for time series generation
- **Paper:** [arXiv:2201.07006](https://arxiv.org/abs/2201.07006)

---

### 6.2 Interpretability Tools

#### **Sparse Autoencoder Libraries**
- **SAELens:** Tool for training and analyzing SAEs
  - [mechanistic-interpretability-saelens](https://ai.codefather.cn/skills/2014279386386608131)
- **Anthropic's SAE Implementation:** Dictionary learning for transformers

#### **Activation Patching**
- **TransformerLens:** Library for mechanistic interpretability
  - Supports activation patching, caching, hooks
- **PyTorch Hooks:** Custom implementation for patching

#### **Visualization Tools**
- **CircuitsVis:** Interactive attention visualization
- **BertViz:** Multi-head attention visualization
- **Tensor2Tensor:** Attention and activation visualization

#### **Probing Libraries**
- **Edge Probing:** Toolkit for probing tasks
- **SentEval:** Benchmark for sentence embeddings (adaptable)

---

### 6.3 Benchmarking Frameworks

#### **Time Series Forecasting Benchmarks**
- **Monash Time Series Forecasting Archive:** Large collection of datasets
- **M4 Competition:** Classic forecasting benchmark
- **M5 Competition:** Retail forecasting with hierarchical structure
- **Long Sequence Time Series Forecasting (LSTF):** ETT, Electricity, Traffic datasets

#### **Classification Benchmarks**
- **UCR Archive:** Univariate time series classification
- **UEA Archive:** Multivariate time series classification
- **PHM Datasets:** Predictive maintenance and industrial applications

#### **Foundation Model Benchmarks**
- **Zero-shot evaluation:** Performance on unseen datasets
- **Transfer learning benchmarks:** Domain adaptation tests
- **Multi-task benchmarks:** Cross-task generalization

---

## 7. Key Architectures and Paradigms

### 7.1 Transformer-Based Architectures

#### **Encoder-Only (BERT-style)**
- **Examples:** MOMENT
- **Pre-training:** Masked time series modeling
- **Use cases:** Representation learning, classification

#### **Decoder-Only (GPT-style)**
- **Examples:** TimesFM
- **Pre-training:** Autoregressive forecasting
- **Use cases:** Generation, forecasting

#### **Encoder-Decoder (T5-style)**
- **Examples:** Chronos (adapts T5)
- **Pre-training:** Sequence-to-sequence
- **Use cases:** Flexible input-output forecasting

---

### 7.2 Patch-Based Methods

**Concept:** Divide time series into patches (segments) for processing

**Benefits:**
- Reduced computational complexity (O(n²) → O(n²/p²))
- Better local pattern capture
- Hierarchical representation learning

**Examples:**
- **PatchTST:** Patches + Transformer
- **VisionTS:** Adapted from Vision Transformers

**Implementation:**
```
Time series: [x₁, x₂, ..., xₙ]
Patches: [[x₁, ..., xₚ], [xₚ₊₁, ..., x₂ₚ], ...]
Patch embeddings: Add positional encoding
Process with Transformer
```

---

### 7.3 Masked Autoencoding

**Concept:** Mask portions of time series and train model to reconstruct

**Pre-training Objective:**
```
Input: Time series with masked segments
Target: Reconstruct masked values
Loss: MSE(reconstructed, original)
```

**Examples:**
- **MOMENT:** Masked time series prediction
- **TimeMAE:** Semantic unit elevation
- **ExtraMAE:** With extrapolator for generation
- **TFMAE:** Temporal-frequency masking

**Advantages:**
- Self-supervised learning from unlabeled data
- Learns robust temporal representations
- Transferable to downstream tasks

---

### 7.4 LLM-Based Approaches

#### **Reprogramming (Time-LLM)**
- Freeze pre-trained LLM
- Transform time series to text prototypes
- Use prompt engineering
- No LLM fine-tuning required

#### **Direct Tokenization (Chronos)**
- Quantize continuous time series values
- Treat as discrete tokens
- Use standard language model architecture
- Probabilistic output via token distribution

#### **Fine-tuning Approaches**
- Start with pre-trained LLM
- Fine-tune on time series data
- Add task-specific heads
- Examples: GPT-4 for time series (limited research)

---

### 7.5 Probabilistic vs. Deterministic

#### **Deterministic Models**
- **Output:** Point forecasts
- **Examples:** TimesFM (base version), PatchTST
- **Loss:** MSE, MAE

#### **Probabilistic Models**
- **Output:** Distribution forecasts
- **Examples:** Chronos, Lag-Llama
- **Loss:** NLL (Negative Log-Likelihood), CRPS
- **Advantages:** Uncertainty quantification, better decision-making

---

## 8. Benchmarks and Datasets

### 8.1 Forecasting Benchmarks

#### **Monash Time Series Forecasting Archive**
- **Size:** 100K+ time series
- **Domains:** Finance, weather, sales, etc.
- **Characteristics:** Diverse frequencies, lengths

#### **M4 Competition**
- **Series:** 100K time series
- **Frequencies:** Yearly, quarterly, monthly, weekly, daily, hourly
- **Horizons:** Varying prediction lengths

#### **M5 Competition**
- **Domain:** Retail sales (Walmart)
- **Features:** Hierarchical structure, external factors
- **Task:** Uncertainty quantification

#### **ETT (Electricity Transformer Temperature)**
- **Variants:** ETTh1, ETTh2, ETTm1, ETTm2
- **Use case:** Long sequence forecasting
- **Multivariate:** 7 features

#### **Electricity, Traffic, Weather**
- **Electricity:** 321 clients, 15-min intervals
- **Traffic:** 862 sensors, 20-min intervals
- **Weather:** 21 meteorological indicators

---

### 8.2 Classification Benchmarks

#### **UCR Time Series Classification Archive**
- **Datasets:** 128 univariate datasets
- **Domains:** Medical, industrial, biological
- **Sizes:** Varying lengths and classes

#### **UEA Multivariate Time Series Classification Archive**
- **Datasets:** 30 multivariate datasets
- **Complexity:** Multiple channels, varying dimensions

#### **PHM 2018 Ion Mill Etching Dataset**
- **Use case:** Predictive maintenance
- **Features:** Multivariate sensor data
- **Application:** Used in TimeSAE paper
- **Labels:** Etching cycle phases

---

### 8.3 Anomaly Detection Benchmarks

#### **Yahoo Webscope S5**
- **Type:** Server metrics
- **Anomalies:** Labeled anomalies
- **Use case:** Real-world system monitoring

#### **NAB (Numenta Anomaly Benchmark)**
- **Datasets:** Real-world and artificial
- **Metrics:** Specialized scoring for early detection
- **Focus:** Temporal anomaly detection

#### **SWaT (Secure Water Treatment)**
- **Domain:** Industrial control systems
- **Type:** Cyber-physical attacks
- **Features:** 51 sensors and actuators

---

### 8.4 Foundation Model Evaluation

#### **Zero-Shot Evaluation**
- **Protocol:** Test on datasets not seen during pre-training
- **Metrics:** MASE, sMAPE, MAE, MSE
- **Comparison:** Against supervised baselines

#### **Transfer Learning Evaluation**
- **Protocol:** Fine-tune on target domain, evaluate
- **Metrics:** Few-shot performance, domain adaptation
- **Comparison:** Against training from scratch

#### **Multi-Task Evaluation**
- **Protocol:** Single model on multiple tasks
- **Tasks:** Forecasting, classification, anomaly detection, imputation
- **Metrics:** Task-specific metrics, average performance

---

## 9. Open Problems and Future Directions

### 9.1 Interpretability Challenges

#### **Semantic Gap in Time Series**
- **Problem:** Time series features lack clear semantic labels (vs. words in NLP)
- **Impact:** Difficult to auto-label discovered features
- **Research Need:** Domain-specific dictionaries, automated interpretation methods

#### **Continuous vs. Discrete**
- **Problem:** Time series is continuous, most interpretability methods designed for discrete tokens
- **Impact:** Adaptation challenges, loss of precision
- **Research Need:** Continuous-domain interpretability techniques

#### **Causal Validation**
- **Problem:** Distinguishing correlation from causation in feature importance
- **Impact:** Risk of spurious explanations
- **Research Need:** Robust causal intervention frameworks

#### **Non-Linear Dynamics**
- **Problem:** Linear Representation Hypothesis may not hold for chaotic systems
- **Impact:** Incomplete interpretability
- **Research Need:** Non-linear probing, Gated SAEs

---

### 9.2 Architectural Challenges

#### **Computational Efficiency**
- **Problem:** Large foundation models are computationally expensive
- **Impact:** Deployment barriers, environmental concerns
- **Research Need:** Efficient architectures, distillation, quantization

#### **Scalability to Long Sequences**
- **Problem:** Transformers scale quadratically with sequence length
- **Impact:** Limited context for long-term dependencies
- **Research Need:** Linear attention, state-space models, hierarchical transformers

#### **Multi-Scale Patterns**
- **Problem:** Time series exhibit patterns at multiple time scales
- **Impact:** Single-scale models miss important dynamics
- **Research Need:** Multi-resolution architectures, wavelet-based methods

---

### 9.3 Data Challenges

#### **Limited Public Time Series Data**
- **Problem:** Much time series data is proprietary (financial, industrial, healthcare)
- **Impact:** Limited pre-training data, bias toward available domains
- **Research Need:** Synthetic data generation, privacy-preserving data sharing

#### **Data Quality and Missing Values**
- **Problem:** Real-world time series have gaps, noise, errors
- **Impact:** Model robustness, interpretability complications
- **Research Need:** Robust pre-training objectives, uncertainty-aware models

#### **Domain Shift and Distribution Change**
- **Problem:** Time series distributions change over time (non-stationarity)
- **Impact:** Foundation models trained on historical data may be outdated
- **Research Need:** Online adaptation, continual learning, concept drift detection

---

### 9.4 Evaluation Challenges

#### **Lack of Standardized Benchmarks for Foundation Models**
- **Problem:** Existing benchmarks designed for supervised learning
- **Impact:** Incomplete evaluation of foundation model capabilities
- **Research Need:** Zero-shot benchmarks, transfer learning benchmarks, multi-task benchmarks

#### **Interpretability Evaluation**
- **Problem:** How to quantitatively evaluate interpretability?
- **Impact:** Subjective assessment, lack of comparability
- **Research Need:** Automated interpretability metrics, human evaluation frameworks

#### **Fairness and Bias**
- **Problem:** Foundation models may inherit biases from pre-training data
- **Impact:** Unfair predictions in sensitive applications
- **Research Need:** Bias detection and mitigation, fairness-aware pre-training

---

### 9.5 Theoretical Foundations

#### **Understanding Zero-Shot Transfer**
- **Problem:** Why do foundation models generalize to unseen datasets?
- **Impact:** Unpredictable performance, difficulty improving
- **Research Need:** Theoretical analysis, transfer learning theory

#### **Scaling Laws for Time Series**
- **Problem:** Are there scaling laws like in LLMs (Chinchilla, etc.)?
- **Impact:** Suboptimal model and data sizing
- **Research Need:** Empirical scaling studies, theoretical frameworks

#### **Representational Structure**
- **Problem:** What do time series foundation models actually learn?
- **Impact:** Black-box nature persists
- **Research Need:** Mechanistic interpretability research (TimeSAE, etc.)

---

### 9.6 Future Research Directions

#### **Multi-Modal Foundation Models**
- Combine time series with text, images, tabular data
- Applications: News-driven forecasting, visual time series analysis

#### **Causal Time Series Models**
- Integrate causal discovery with foundation models
- Enable counterfactual reasoning and intervention analysis

#### **Interpretable-by-Design Architectures**
- Move beyond post-hoc interpretability
- Design models with inherent interpretability (e.g., disentangled representations)

#### **Automated Feature Discovery and Labeling**
- Use LLMs to auto-label discovered features
- Create semantic dictionaries for time series concepts

#### **Federated and Privacy-Preserving Foundation Models**
- Train on distributed data without centralization
- Enable collaboration across organizations with sensitive data

#### **Real-Time Interpretability**
- Online monitoring and interpretation
- Alert systems for anomalous model behavior

---

## 10. References

### Foundation Model Papers

1. **TimeGPT-1** (2023). arXiv:2310.03589
2. **TimesFM** (2024). Das et al., ICML 2024. arXiv:2310.10688
3. **Chronos** (2024). Ansari et al. arXiv:2403.07815
4. **MOMENT** (2024). Goswami et al., ICML 2024.
5. **Time-LLM** (2024). Jin et al., ICLR 2024. arXiv:2310.01728
6. **UniTS** (2024). arXiv:2403.00131, NeurIPS 2024
7. **TimeMAE** (2024). ACM Digital Library.
8. **ExtraMAE** (2022). arXiv:2201.07006

### Survey Papers

9. **Foundation Models for Time Series Analysis: A Tutorial and Survey** (2024). arXiv:2403.14735, KDD 2024
10. **Empowering Time Series Analysis with Foundation Models** (2024). arXiv:2405.02358
11. **Foundation Models for Time Series: A Survey** (2025). arXiv:2504.04011

### Mechanistic Interpretability Papers

12. **TimeSAE: Mechanistic Interpretability for Time-Series Foundation Models** (2026). ICLR 2026 Workshop TSALM.
13. **Mechanistic Interpretability for Transformer-based Time Series Classification** (2025). Kalnāre et al. arXiv:2511.21514

### Interpretability Methods

14. **Sparse Autoencoders Find Highly Interpretable Features** (2024). Cunningham et al., ICLR 2024
15. **Towards Monosemanticity** (2023). Bricken et al., Transformer Circuits Thread
16. **Toy Models of Superposition** (2022). Elhage et al., Transformer Circuits Thread
17. **A Unified Approach to Interpreting Model Predictions** (2017). Lundberg & Lee, NeurIPS (SHAP)
18. **Axiomatic Attribution for Deep Networks** (2017). Sundararajan et al., ICML (Integrated Gradients)
19. **Sanity Checks for Saliency Maps** (2018). Adebayo et al., NeurIPS

### Related Architecture Papers

20. **PatchTST** (2023). Nie et al.
21. **VisionTS** (2024). OpenReview
22. **TFMAE** (2024). ICDE 2024

---

## Appendix A: Quick Reference - Model Comparison Table

| Model | Organization | Year | Architecture | Pre-training | Key Feature | Zero-Shot | Code Available |
|-------|-------------|------|--------------|--------------|-------------|-----------|----------------|
| **TimeGPT-1** | Nixtla | 2023 | Transformer | 100B points | First TS FM | ✓ | API only |
| **TimesFM** | Google | 2024 | Decoder-only | 100B points | LLM-inspired | ✓ | ✓ |
| **Chronos** | Amazon | 2024 | T5-based | Tokenization | LLM adaptation | ✓ | ✓ |
| **MOMENT** | Academic | 2024 | Encoder | Masked TS | Open model | ✓ | ✓ |
| **Time-LLM** | Academic | 2024 | Reprogrammed LLM | Frozen LLM | No fine-tuning | ✓ | ✓ |
| **UniTS** | Harvard | 2024 | Unified | Multi-task | All tasks | ✓ | ✓ |
| **Lag-Llama** | Academic | 2024 | Transformer | Probabilistic | Uncertainty | ✓ | ✓ |
| **TTM** | IBM | 2024 | Mixer | Efficient | Lightweight | ✓ | ✓ |

---

## Appendix B: Interpretability Methods Comparison

| Method | Type | What It Reveals | Causal? | Computational Cost | TSFM Applicability |
|--------|------|-----------------|---------|-------------------|-------------------|
| **Sparse Autoencoders** | Decomposition | Monosemantic features | With steering | High (training) | ✓✓✓ |
| **Activation Patching** | Intervention | Causal importance | ✓ | Medium | ✓✓✓ |
| **Attention Saliency** | Visualization | Attention patterns | ✗ | Low | ✓✓ |
| **Probing Classifiers** | Testing | Encoded information | Partial | Low | ✓✓ |
| **Integrated Gradients** | Attribution | Input importance | Partial | Medium | ✓ |
| **SHAP** | Attribution | Feature contribution | ✗ | High | ✓ |

---

## Appendix C: Research Timeline

```
2023
├── Aug: TimeGPT-1 released (first TS foundation model)
└── Oct: TimeGPT paper published (arXiv:2310.03589)

2024
├── Jan: TimesFM preprint (arXiv:2310.10688)
├── Mar: Chronos released (arXiv:2403.07815)
├── Mar: Foundation Models Survey (arXiv:2403.14735)
├── May: Empowering TS Analysis Survey (arXiv:2405.02358)
├── Jun: UniTS released (arXiv:2403.00131)
├── Jul: ICML 2024 (TimesFM, MOMENT)
├── Sep: ICLR 2024 (Time-LLM)
├── Oct: KDD 2024 Tutorial on TS Foundation Models
└── Dec: NeurIPS 2024 (UniTS)

2025
├── Mar: TimeSAE submitted to ICLR 2026 Workshop
├── Nov: Mechanistic Interpretability for TSC (arXiv:2511.21514)
└── Ongoing: Rapid expansion of TSFM research

2026
├── Jan: ICLR 2026 Workshop TSALM (TimeSAE presentation)
└── Current: Active research in mechanistic interpretability for TSFMs
```

---

## Appendix D: Recommended Reading Order

### For Beginners:
1. Start with **Foundation Models Survey** (arXiv:2403.14735) - comprehensive overview
2. Read **TimeGPT-1** paper - first foundation model
3. Explore **TimesFM** or **Chronos** - understand architecture
4. Try **Time-LLM** code - hands-on experience

### For Interpretability Focus:
1. Read **TimeSAE** paper - first mechanistic interpretability for TSFMs
2. Study **Sparse Autoencoders** (Cunningham et al., Bricken et al.)
3. Explore **Activation Patching** tutorials
4. Read **Mechanistic Interpretability for TSC** (arXiv:2511.21514)

### For Implementation:
1. Choose model: **TimesFM** (Google), **Chronos** (Amazon), or **UniTS** (Harvard)
2. Clone repository and follow README
3. Try zero-shot forecasting on benchmark datasets
4. Apply interpretability methods (SAEs, attention visualization)

### For Research:
1. Identify open problems (Section 9)
2. Read recent papers (2024-2026)
3. Explore interpretability techniques from NLP
4. Adapt and apply to time series domain

---

## Appendix E: Glossary

**Foundation Model (FM):** A model pre-trained on broad data at scale, adaptable to many tasks.

**TSFM (Time Series Foundation Model):** Foundation model specifically designed for time series analysis.

**Zero-Shot Learning:** Ability to perform tasks without task-specific training data.

**Few-Shot Learning:** Learning from a small number of examples.

**Transfer Learning:** Applying knowledge from one domain/task to another.

**Mechanistic Interpretability:** Understanding the internal mechanisms and algorithms a neural network implements.

**Sparse Autoencoder (SAE):** Neural network trained to reconstruct inputs using sparse activations, revealing interpretable features.

**Activation Patching:** Intervention technique that replaces specific activations to test causal importance.

**Monosemantic Feature:** A feature that corresponds to a single, interpretable concept.

**Polysemantic Neuron:** A neuron that activates for multiple unrelated concepts.

**Superposition:** Dense representations encoding many features in superposition.

**Linear Representation Hypothesis:** The idea that features are represented as linear directions in activation space.

**Top-K SAE:** SAE variant that only keeps the top-k activations.

**Causal Tracing:** Method to identify which components are causally responsible for model outputs.

**Tokenization:** Converting continuous time series into discrete tokens.

**Patching:** Dividing time series into segments (patches) for processing.

---

## Document Information

**Created:** March 11, 2026
**Version:** 1.0
**Author:** Research compilation for NotebookLLM library
**Purpose:** Complete literature review for mechanistic interpretability in time series foundation models

**Total Papers Referenced:** 22+ core papers
**Total Categories:** 10 major sections
**Total Foundation Models Covered:** 8+ models
**Total Interpretability Methods:** 6+ techniques

---

**End of Literature Review**

For updates and additions, please refer to the latest version in the repository.

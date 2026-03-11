# Research Roadmap: Mechanistic Interpretability for TSFMs

This document outlines potential research directions, open problems, and methodological approaches for advancing mechanistic interpretability in Time Series Foundation Models.

---

## 🎯 High-Priority Research Directions

### 1. Automated Feature Interpretation

**Problem:** Time series features lack semantic labels (unlike words in NLP), making manual interpretation tedious.

**Research Questions:**
- Can we use LLMs to auto-label discovered time series features?
- What domain-specific dictionaries can we create?
- How do we validate automated interpretations?

**Potential Approaches:**
- **LLM-based labeling:** Use GPT-4/Claude to describe feature patterns
- **Template matching:** Match features to known time series primitives (trend, seasonality, noise)
- **Cross-domain transfer:** Use labels from well-studied domains for similar features

**Datasets:** PHM 2018, UCR Archive, Monash Archive

**Metrics:** Human agreement rate, downstream task performance, interpretation consistency

---

### 2. Non-Linear Dynamics and Chaotic Systems

**Problem:** Linear Representation Hypothesis may not hold for chaotic time series.

**Research Questions:**
- What percentage of time series features are linear vs. non-linear?
- Do SAEs fail to capture chaotic dynamics?
- What alternative architectures could work?

**Potential Approaches:**
- **Gated SAEs:** Non-linear gating mechanisms
- **Manifold learning:** Isomap, t-SNE, UMAP for feature discovery
- **Hamiltonian Neural Networks:** Physics-informed architectures
- **Polynomial feature expansion:** Capture higher-order interactions

**Datasets:** Lorenz system, Rossler attractor, real chaotic systems (weather, ECG)

**Metrics:** Reconstruction error on chaotic systems, Lyapunov exponent preservation

---

### 3. Real-Time Interpretability

**Problem:** Current interpretability methods require offline analysis.

**Research Questions:**
- Can we maintain interpretability during online inference?
- How to update feature dictionaries in streaming scenarios?
- What's the latency budget for real-time interpretability?

**Potential Approaches:**
- **Online SAE training:** Incremental updates
- **Cached feature dictionaries:** Pre-computed interpretations
- **Lightweight probes:** Fast surrogate models
- **Hierarchical interpretation:** Coarse-to-fine analysis

**Datasets:** Streaming sensor data, financial tick data, IoT telemetry

**Metrics:** Latency (ms), interpretability quality, drift detection

---

### 4. Multi-Layer Circuit Analysis (I like this one)

**Problem:** Current work focuses on single layers; complex circuits span multiple layers.

**Research Questions:**
- How do features compose across layers?
- What circuits emerge for specific time series patterns?
- Can we identify universal circuits across TSFMs?

**Potential Approaches:**
- **Path patching:** Trace information flow across layers
- **Circuit induction:** Identify repeating circuit patterns
- **Layer-wise relevance propagation:** Attribute predictions to layers
- **Causal scrubbing:** Test minimal sufficient circuits

**Datasets:** Models with many layers (TimesFM, large MOMENT variants)

**Metrics:** Circuit completeness, cross-model transfer, prediction accuracy

---

### 5. Interpretability for Multi-Variate Time Series

**Problem:** Most interpretability work focuses on univariate time series.

**Research Questions:**
- How do features represent interactions between variables?
- Can we decompose multivariate representations into channel-specific and cross-channel features?
- What visualization techniques work for high-dimensional features?

**Potential Approaches:**
- **Channel-wise SAEs:** Separate encoder per channel
- **Attention-based decomposition:** Use cross-attention patterns
- **Graph neural networks:** Model variable interactions
- **Tensor factorization:** PARAFAC, Tucker decomposition

**Datasets:** Electricity, Traffic, Weather, SWaT, WADI

**Metrics:** Cross-channel feature interpretability, interaction detection accuracy

---

## 🔬 Methodological Innovations

### 1. Causal Intervention Protocols

**Current State:** TimeSAE demonstrates steering, but lacks standardized protocol.

**Innovation Needed:**
- **Standardized benchmark:** Dataset + evaluation protocol
- **Multiple intervention types:** Activation, weight, and input interventions
- **Safety guarantees:** Ensure interventions don't break model

**Implementation:**
```python
# Pseudocode for standardized protocol
class CausalInterventionBenchmark:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def evaluate_feature(self, feature_idx):
        # 1. Identify feature activation pattern
        pattern = self.identify_pattern(feature_idx)

        # 2. Intervene with different strengths
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            steered_output = self.steer(feature_idx, alpha)

        # 3. Measure dose-response relationship
        linearity_score = self.measure_linearity(steered_outputs)

        # 4. Validate semantic change
        semantic_match = self.validate_semantics(pattern, steered_output)

        return linearity_score, semantic_match
```

---

### 2. Interpretability-Aware Model Design (Train TimesFM so it is interpretable via circuit analasys)

**Current State:** Interpretability is post-hoc.

**Innovation Needed:**
- **Disentangled representations:** Train models with built-in interpretability
- **Monosemantic regularization:** Encourage monosemantic neurons during training
- **Interpretable attention:** Constrain attention patterns to be interpretable

**Potential Approaches:**
- **Beta-VAE for time series:** Disentangled latent factors
- **Concept bottleneck models:** Force predictions through interpretable concepts
- **Attention supervision:** Guide attention to interpretable patterns

**Research Questions:**
- Does interpretability-aware training hurt performance?
- What's the trade-off between interpretability and accuracy?
- Can we achieve both zero-shot capability and interpretability?

---

### 3. Cross-Model Interpretability Transfer (I like this as well)

**Current State:** Interpretability analysis is model-specific.

**Innovation Needed:**
- **Universal feature dictionaries:** Features that transfer across TSFMs
- **Interpretability distillation:** Transfer interpretations from teacher to student
- **Feature alignment:** Match features across different models

**Potential Approaches:**
- **Canonical Correlation Analysis (CCA):** Align representations
- **Optimal transport:** Match feature distributions
- **Meta-learning:** Learn to transfer interpretations

**Research Questions:**
- Do different TSFMs learn similar features?
- Can we create a "universal time series feature dictionary"?
- How to handle architectural differences?

---

## 📊 Benchmark and Evaluation Frameworks

### 1. Interpretability Benchmark Suite

**Components:**
- **Datasets:** Diverse time series (finance, healthcare, industry, weather)
- **Tasks:** Forecasting, classification, anomaly detection, imputation
- **Models:** Multiple TSFMs (TimesFM, Chronos, MOMENT, UniTS)
- **Interpretability methods:** SAEs, attention, probing, patching

**Evaluation Metrics:**
- **Reconstruction fidelity:** R², MSE for SAEs
- **Sparsity:** L0 norm, active feature ratio
- **Causality:** Intervention success rate, dose-response linearity
- **Interpretability:** Human evaluation, auto-labeling accuracy
- **Efficiency:** Training time, inference latency

**Implementation:**
```python
class TSFMInterpretabilityBenchmark:
    def evaluate(self, model, interpretability_method):
        results = {}

        # 1. Reconstruction fidelity
        results['reconstruction_r2'] = self.measure_reconstruction()

        # 2. Causal intervention success
        results['causal_success_rate'] = self.test_interventions()

        # 3. Human interpretability study
        results['human_score'] = self.human_evaluation()

        # 4. Downstream task performance
        results['downstream_accuracy'] = self.test_downstream()

        # 5. Computational efficiency
        results['inference_latency'] = self.measure_latency()

        return results
```

---

### 2. Human Evaluation Protocol

**Challenge:** Interpretability is subjective; need standardized human evaluation.

**Protocol:**
1. **Feature description task:** Humans describe what feature captures
2. **Feature prediction task:** Given feature description, predict when it activates
3. **Feature steering task:** Describe desired change, find feature to steer
4. **Feature comparison task:** Rate interpretability of different features

**Metrics:**
- Inter-rater agreement (Fleiss' Kappa)
- Description accuracy
- Prediction accuracy
- Steering success rate

**Tools:**
- Amazon Mechanical Turk / Prolific for crowd-sourcing
- Domain experts for specialized datasets
- Interactive visualization tools

---

## 🌟 Novel Application Domains

### 1. Healthcare and Medical Time Series

**Opportunities:**
- ECG interpretation for arrhythmia detection
- EEG analysis for seizure prediction
- Vital signs monitoring in ICU

**Challenges:**
- High stakes: errors can be fatal
- Regulatory requirements (FDA approval)
- Privacy concerns (HIPAA)

**Research Directions:**
- **Clinician-AI collaboration:** Interpretability for doctor-AI teaming
- **Regulatory-grade interpretability:** Satisfy FDA requirements
- **Privacy-preserving interpretability:** Interpret without exposing patient data

---

### 2. Financial Time Series

**Opportunities:**
- Algorithmic trading interpretability
- Risk model explanation for regulators
- Fraud detection transparency

**Challenges:**
- Adversarial actors may exploit interpretations
- High frequency: latency constraints
- Non-stationarity: concept drift

**Research Directions:**
- **Adversarial robustness:** Interpretations that don't reveal trading strategies
- **Real-time interpretability:** Sub-millisecond latency
- **Drift-aware interpretation:** Update interpretations as markets change

---

### 3. Industrial IoT and Predictive Maintenance

**Opportunities:**
- Equipment failure prediction
- Quality control in manufacturing
- Energy optimization

**Challenges:**
- Multi-sensor fusion (high dimensionality)
- Harsh environments (noisy data)
- Long time horizons (slow degradation)

**Research Directions:**
- **Multi-sensor interpretability:** Cross-sensor feature discovery
- **Noise-robust interpretation:** Extract signal from noise
- **Long-term circuit analysis:** Understand multi-year patterns

---

### 4. Climate and Environmental Science

**Opportunities:**
- Weather forecasting interpretability
- Climate change pattern discovery
- Extreme event prediction

**Challenges:**
- Multi-scale patterns (hours to decades)
- Sparse observations (satellite data)
- Chaotic dynamics

**Research Directions:**
- **Multi-scale interpretability:** Features at different temporal scales
- **Physics-informed interpretation:** Respect physical laws
- **Uncertainty quantification:** Probabilistic interpretations

---

## 🛠️ Tooling and Infrastructure

### 1. Interpretability Toolkit for TSFMs

**Vision:** Scikit-learn-style API for TSFM interpretability.

```python
from tsfm_interpret import SparseAutoencoder, ActivationPatcher, ProbingClassifier

# Load TSFM
model = load_timesfm("google/timesfm-1.0")

# Extract activations
activations = model.extract_activations(dataset, layer=12)

# Train SAE
sae = SparseAutoencoder(latent_dim=32768, sparsity=32)
sae.fit(activations)

# Interpret features
feature_descriptions = sae.interpret_features(top_k=100)

# Causal intervention
patcher = ActivationPatcher(model)
important_features = patcher.identify_important_features(test_data)

# Probing
probe = ProbingClassifier(task="trend_detection")
probe_performance = probe.evaluate(activations)
```

**Components:**
- Activation extraction (universal across TSFMs)
- SAE training and analysis
- Activation patching framework
- Probing classifier library
- Visualization tools
- Benchmark suite

---

### 2. Visualization Platform

**Features:**
- **Interactive feature explorer:** Browse discovered features
- **Activation visualizer:** See when features fire
- **Causal graph viewer:** Explore feature interactions
- **Steering dashboard:** Test interventions in real-time
- **Comparison tool:** Compare features across models

**Technology:**
- Frontend: React + D3.js
- Backend: FastAPI + PostgreSQL
- Deployment: Docker + Kubernetes

---

### 3. Collaborative Research Platform

**Vision:** Shared resource for TSFM interpretability research.

**Components:**
- **Feature database:** Community-contributed feature interpretations
- **Model zoo:** Pre-analyzed TSFMs with cached interpretations
- **Benchmark leaderboard:** Compare interpretability methods
- **Paper repository:** Curated papers with annotations
- **Discussion forum:** Research community discussions

**Incentives:**
- Co-authorship for significant contributions
- Citation tracking for feature usage
- Grants for infrastructure development

---

## 📅 Research Timeline

### Short-Term (6 months)
- [ ] Reproduce TimeSAE results on additional datasets
- [ ] Implement standardized causal intervention benchmark
- [ ] Create initial visualization toolkit
- [ ] Survey domain experts on interpretability needs

### Medium-Term (1-2 years)
- [ ] Develop automated feature interpretation pipeline
- [ ] Publish multi-layer circuit analysis
- [ ] Release open-source interpretability toolkit
- [ ] Establish interpretability benchmark suite

### Long-Term (3-5 years)
- [ ] Achieve real-time interpretability
- [ ] Deploy in safety-critical applications (healthcare, finance)
- [ ] Develop interpretability-aware TSFM architectures
- [ ] Create universal time series feature dictionary

---

## 🎓 PhD Thesis Ideas

### 1. "Mechanistic Interpretability for Probabilistic Time Series Forecasting"
**Focus:** Extend interpretability to probabilistic models (Chronos, Lag-Llama)
**Contribution:** Understand uncertainty quantification mechanisms

### 2. "Causal Interpretability of Time Series Foundation Models"
**Focus:** Develop rigorous causal intervention frameworks
**Contribution:** From correlation to causation in feature interpretation

### 3. "Interpretable-by-Design Time Series Foundation Models"
**Focus:** Architectures with built-in interpretability
**Contribution:** Eliminate post-hoc analysis, achieve native interpretability

### 4. "Multi-Modal Time Series Interpretability"
**Focus:** Time series + text/images/tabular data
**Contribution:** Cross-modal feature discovery and interpretation

### 5. "Real-Time Mechanistic Interpretability for Streaming Time Series"
**Focus:** Online interpretation with latency constraints
**Contribution:** Production-ready interpretability for industrial applications

---

## 🤝 Collaboration Opportunities

### Academic Collaborations
- **Anthropic / OpenAI:** Interpretability methods transfer from LLMs
- **Google Research:** Access to TimesFM internals, compute resources
- **Amazon Science:** Chronos architecture insights
- **Harvard / MIT / Stanford:** Academic research partnerships
- **Domain experts:** Healthcare, finance, climate scientists

### Industry Partnerships
- **Healthcare systems:** Access to medical time series
- **Financial institutions:** Trading and risk data
- **Manufacturing:** IoT sensor data
- **Energy companies:** Smart grid data

### Open-Source Communities
- **Hugging Face:** Model hosting and sharing
- **Weights & Biases:** Experiment tracking
- **PyTorch / TensorFlow:** Framework development

---

## 📝 Grant Proposal Ideas

### 1. NSF: "Interpretable AI for Critical Infrastructure"
- Focus: Power grid, water systems, transportation
- Amount: $500K - $1M
- Duration: 3 years

### 2. NIH: "Mechanistic Interpretability for Medical Time Series"
- Focus: ECG, EEG, vital signs monitoring
- Amount: $1M - $2M
- Duration: 4 years

### 3. DARPA: "Explainable AI for Defense Applications"
- Focus: Sensor fusion, threat detection
- Amount: $2M - $5M
- Duration: 4 years

### 4. Industry: "Real-Time Interpretability for Financial Markets"
- Focus: Algorithmic trading, risk management
- Amount: $500K - $1M
- Duration: 2 years

---

## 🎯 Impact Metrics

### Academic Impact
- **Publications:** Top-tier venues (NeurIPS, ICML, ICLR, KDD)
- **Citations:** Track citation count and h-index
- **Code usage:** GitHub stars, forks, downloads
- **Reproducibility:** Number of successful reproductions

### Practical Impact
- **Adoption:** Number of companies using interpretability tools
- **Regulatory approval:** FDA, SEC acceptance of interpretations
- **Safety incidents:** Reduction in AI-related accidents
- **Efficiency gains:** Time saved in model debugging

### Community Impact
- **Open-source contributions:** Number of contributors
- **Tutorial attendance:** Workshop and tutorial participation
- **Educational materials:** Courses, blog posts, videos created
- **Diversity:** Participation from underrepresented groups

---

## 🚀 Getting Started with Research

### Step 1: Choose a Direction
- Read this roadmap
- Identify what excites you most
- Assess your skills and resources
- Pick 1-2 directions to focus on

### Step 2: Deep Dive
- Read all relevant papers (see literature review)
- Reproduce key results
- Identify gaps and limitations
- Formulate specific research questions

### Step 3: Design Experiments
- Define success criteria
- Choose datasets and models
- Plan computational resources
- Set timeline and milestones

### Step 4: Execute and Iterate
- Run experiments
- Analyze results
- Refine hypotheses
- Repeat

### Step 5: Share and Collaborate
- Write papers and blog posts
- Release code and data
- Present at conferences
- Build collaborations

---

## 💭 Final Thoughts

The field of mechanistic interpretability for time series foundation models is in its infancy. The TimeSAE paper (2026) is just the beginning. There are vast opportunities for impactful research that could:

1. **Make AI safer:** Understand and control black-box models
2. **Advance science:** Discover new patterns in time series data
3. **Enable regulation:** Satisfy requirements for high-stakes applications
4. **Democratize AI:** Make powerful models accessible and trustworthy

The time to act is now. The foundations are being laid, and early contributions will shape the field for years to come.

**Your research matters. Start today.**

---

*Last updated: March 11, 2026*
*Contact: [Your research community forum]*

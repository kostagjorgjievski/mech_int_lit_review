# Quick Start Guide: Mechanistic Interpretability for Time Series Foundation Models

This guide provides a curated reading path and implementation checklist for researchers and practitioners interested in mechanistic interpretability for Time Series Foundation Models (TSFMs).

---

## 🎯 Target Audiences

### For Researchers
Focus: Understanding state-of-the-art and identifying open problems

### For Practitioners
Focus: Implementing and interpreting TSFMs in production

### For Beginners
Focus: Getting started with time series foundation models

---

## 📚 Reading Paths

### Path 1: Foundation Model Basics (Beginners)

**Estimated Time: 1-2 weeks**

1. **Start Here: Survey Paper**
   - Read: [Foundation Models for Time Series Analysis: A Tutorial and Survey](https://arxiv.org/abs/2403.14735)
   - Time: 2-3 hours
   - Focus: Sections 1-3 (Introduction, Background, Taxonomy)

2. **First Foundation Model: TimeGPT**
   - Read: [TimeGPT-1](https://arxiv.org/abs/2310.03589) (Introduction and Methods)
   - Time: 1 hour
   - Try: [Nixtla API](https://github.com/Nixtla/nixtla) (1-2 hours hands-on)

3. **Understand Architecture: TimesFM**
   - Read: [TimesFM paper](https://arxiv.org/abs/2310.10688) (Architecture section)
   - Time: 1-2 hours
   - Try: [Google Research Colab](https://github.com/google-research/timesfm) (2 hours)

4. **Alternative Approach: Chronos**
   - Read: [Chronos paper](https://arxiv.org/abs/2403.07815) (Tokenization section)
   - Time: 1 hour
   - Try: [Chronos Demo](https://github.com/amazon-science/chronos-forecasting) (2 hours)

**Checkpoint:** You should now understand what TSFMs are, their key architectures, and how to use them for zero-shot forecasting.

---

### Path 2: Interpretability Focus (Researchers)

**Estimated Time: 2-3 weeks**

1. **Background: Interpretability in NLP**
   - Read: [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
   - Time: 2 hours
   - Note: Foundation for understanding SAEs

2. **Core Method: Sparse Autoencoders**
   - Read: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
   - Time: 2-3 hours
   - Read: [Sparse Autoencoders Find Highly Interpretable Features](https://openreview.net/forum?id=...)
   - Time: 1-2 hours

3. **Application to TSFMs: TimeSAE**
   - Read: [TimeSAE Paper](https://openreview.net/pdf?id=Ojd6YjHpyE) - FULL PAPER
   - Time: 3-4 hours
   - Focus: Methods, Results, and Limitations sections
   - Take notes on: Causal intervention protocol

4. **Complementary Work: Classification Interpretability**
   - Read: [Mechanistic Interpretability for Transformer-based TSC](https://arxiv.org/abs/2511.21514)
   - Time: 2 hours
   - Compare: Methods vs. TimeSAE

5. **Hands-On Implementation**
   - Implement: SAE on a simple TSFM (e.g., MOMENT)
   - Time: 1-2 days
   - Dataset: Use PHM 2018 or UCR Archive

**Checkpoint:** You should understand mechanistic interpretability techniques and be ready to apply them to TSFMs.

---

### Path 3: Production Implementation (Practitioners)

**Estimated Time: 1 week**

1. **Choose Your Model**
   - Option A: [TimesFM](https://github.com/google-research/timesfm) - Best for forecasting
   - Option B: [Chronos](https://github.com/amazon-science/chronos-forecasting) - Best for probabilistic
   - Option C: [UniTS](https://github.com/mims-harvard/UniTS) - Best for multi-task
   - Time: 1-2 hours to decide

2. **Set Up Environment**
   ```bash
   # Example for TimesFM
   git clone https://github.com/google-research/timesfm
   cd timesfm
   pip install -r requirements.txt
   ```
   - Time: 1-2 hours

3. **Run Zero-Shot Forecasting**
   - Load pre-trained model
   - Test on your data
   - Evaluate metrics
   - Time: 2-4 hours

4. **Add Interpretability Layer**
   - Extract activations
   - Train SAE (optional)
   - Visualize attention patterns
   - Time: 4-8 hours

5. **Production Deployment**
   - Model serving (Docker, FastAPI)
   - Monitoring and logging
   - Performance optimization
   - Time: 1-2 days

**Checkpoint:** You have a working TSFM with basic interpretability in production.

---

## 🛠️ Implementation Checklist

### Phase 1: Setup

- [ ] Choose TSFM model based on requirements
- [ ] Clone repository and install dependencies
- [ ] Download pre-trained weights
- [ ] Test on sample dataset
- [ ] Verify zero-shot performance

### Phase 2: Data Preparation

- [ ] Format data according to model requirements
- [ ] Split into train/validation/test
- [ ] Normalize/standardize if required
- [ ] Create data loaders
- [ ] Validate data quality

### Phase 3: Model Evaluation

- [ ] Run baseline evaluation
- [ ] Calculate metrics (MASE, sMAPE, MAE)
- [ ] Compare to benchmarks
- [ ] Analyze failure cases
- [ ] Document results

### Phase 4: Interpretability Integration

- [ ] Extract model activations
- [ ] Choose interpretability method:
  - [ ] Attention visualization (easiest)
  - [ ] Probing classifiers (medium)
  - [ ] Sparse Autoencoders (advanced)
  - [ ] Activation patching (research)
- [ ] Implement chosen method
- [ ] Validate interpretability results

### Phase 5: Documentation & Deployment

- [ ] Document model architecture
- [ ] Document interpretability findings
- [ ] Create API endpoints
- [ ] Set up monitoring
- [ ] Prepare user guide

---

## 📊 Key Metrics to Track

### Forecasting Performance
- **MASE** (Mean Absolute Scaled Error)
- **sMAPE** (Symmetric Mean Absolute Percentage Error)
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **CRPS** (Continuous Ranked Probability Score) - for probabilistic models

### Interpretability Quality
- **Reconstruction R²** (for SAEs)
- **Sparsity Ratio** (for SAEs)
- **Feature Kurtosis** (for SAEs)
- **Causal Effect Size** (for activation patching)
- **Human Evaluation** (feature interpretability)

### Computational Efficiency
- **Inference Time** (ms per sample)
- **Memory Usage** (GB)
- **Model Size** (MB/GB)
- **Training Time** (if fine-tuning)

---

## 🔧 Code Templates

### Template 1: Load and Use Pre-trained TSFM

```python
# Example: TimesFM
import timesfm

# Load model
model = timesfm.TimesFm(
    context_len=512,
    horizon_len=128,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
)

# Load checkpoint
model.load_from_checkpoint("path/to/checkpoint")

# Forecast
forecast = model.forecast(
    inputs=time_series_data,
    freq="D"  # Daily frequency
)
```

### Template 2: Extract Activations for Interpretability

```python
import torch
from collections import defaultdict

def extract_activations(model, input_data):
    """Extract activations from specific layers"""
    activations = defaultdict(list)

    def hook_fn(name):
        def hook(module, input, output):
            activations[name].append(output.detach())
        return hook

    # Register hooks
    for name, layer in model.named_modules():
        if "layer_12" in name:  # Example: middle layer
            layer.register_forward_hook(hook_fn(name))

    # Forward pass
    with torch.no_grad():
        _ = model(input_data)

    return activations

# Usage
activations = extract_activations(model, time_series_batch)
```

### Template 3: Train Sparse Autoencoder

```python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        # Top-K sparsity
        k = 32
        top_k_values, top_k_indices = torch.topk(z, k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, top_k_indices, top_k_values)

        x_recon = self.decoder(z_sparse)
        return x_recon, z_sparse

# Training loop
sae = SparseAutoencoder(input_dim=1024, latent_dim=32768)
optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

for epoch in range(100):
    for batch in dataloader:
        x_recon, z = sae(batch)
        loss = nn.MSELoss()(x_recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Template 4: Activation Patching

```python
def activation_patching(model, clean_input, corrupted_input, layer_name):
    """Test causal importance of specific activation"""
    # Get clean activations
    clean_acts = {}
    hook = model.get_layer(layer_name).register_forward_hook(
        lambda m, i, o: clean_acts.update({"clean": o})
    )
    clean_output = model(clean_input)
    hook.remove()

    # Get corrupted activations
    corrupted_acts = {}
    hook = model.get_layer(layer_name).register_forward_hook(
        lambda m, i, o: corrupted_acts.update({"corrupted": o})
    )
    corrupted_output = model(corrupted_input)
    hook.remove()

    # Patch: Replace corrupted with clean
    def patch_hook(module, input, output):
        return clean_acts["clean"]

    hook = model.get_layer(layer_name).register_forward_hook(patch_hook)
    patched_output = model(corrupted_input)
    hook.remove()

    # Measure recovery
    recovery = torch.norm(patched_output - clean_output) / torch.norm(corrupted_output - clean_output)
    return recovery.item()
```

---

## 🎓 Learning Resources

### Online Courses
- **Deep Learning for Time Series** (Coursera)
- **Interpretability in Machine Learning** (various university courses)
- **Transformer Architecture Deep Dive** (fast.ai)

### Video Tutorials
- [KDD 2024 Tutorial: Foundation Models for Time Series](https://www.youtube.com/watch?v=HvqFUmksd_M)
- [Anthropic's Interpretability Research](https://www.youtube.com/results?search_query=anthropic+interpretability)
- [Time Series Forecasting with Transformers](https://www.youtube.com/results?search_query=time+series+transformer)

### Blog Posts
- [Google Research: TimesFM Blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [Amazon: Chronos Announcement](https://www.amazon.science/code-and-datasets/chronos-learning-the-language-of-time-series)
- [Nixtla: TimeGPT Introduction](https://medium.com/the-forecaster/timegpt-the-first-foundation-model-for-time-series-forecasting-bf0a75e63b3a)

### Research Communities
- **r/MachineLearning** - Reddit community
- **Papers With Code** - Time Series tag
- **Hugging Face** - Time series models
- **OpenReview** - Latest papers and reviews

---

## 🚨 Common Pitfalls

### 1. **Data Mismatch**
- **Problem:** Pre-trained model expects certain data format
- **Solution:** Check model documentation for input requirements
- **Prevention:** Validate data before model loading

### 2. **Computational Constraints**
- **Problem:** Foundation models are large and slow
- **Solution:** Use smaller variants, quantization, or cloud compute
- **Prevention:** Profile memory and time requirements first

### 3. **Interpretability Illusion**
- **Problem:** Attribution methods show correlation, not causation
- **Solution:** Use causal intervention (activation patching)
- **Prevention:** Validate interpretability claims with multiple methods

### 4. **Zero-Shot Overconfidence**
- **Problem:** Zero-shot performance may not match benchmarks
- **Solution:** Fine-tune on domain-specific data if needed
- **Prevention:** Always evaluate on your specific data

### 5. **Feature Polysemanticity**
- **Problem:** Single neurons activate for multiple concepts
- **Solution:** Use Sparse Autoencoders to decompose
- **Prevention:** Check feature interpretability, not just accuracy

---

## 📅 Weekly Research Update Template

Use this template to track your progress:

```markdown
# Week of [Date]

## Papers Read
- [ ] Paper 1: [Title] - Key insight: [1-2 sentences]
- [ ] Paper 2: [Title] - Key insight: [1-2 sentences]

## Code Implemented
- [ ] Feature 1: [Description]
- [ ] Feature 2: [Description]

## Experiments Run
- [ ] Experiment 1: [Result summary]
- [ ] Experiment 2: [Result summary]

## Open Questions
1. [Question 1]
2. [Question 2]

## Next Week Goals
- [ ] Goal 1
- [ ] Goal 2

## Blockers
- [Blocker 1] - [Proposed solution]
```

---

## 🔗 Essential Links

### Model Repositories
- [TimesFM](https://github.com/google-research/timesfm)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
- [Time-LLM](https://github.com/KimMeen/Time-LLM)
- [UniTS](https://github.com/mims-harvard/UniTS)
- [IBM Granite TSFM](https://github.com/ibm-granite/granite-tsfm)

### Interpretability Tools
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Activation patching
- [CircuitsVis](https://github.com/alan-cooney/CircuitsVis) - Visualization
- [SAELens](https://github.com/jbloom-mats/SAELens) - Sparse autoencoders

### Benchmark Datasets
- [Monash Time Series Archive](https://forecastingdata.org/)
- [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
- [UEA Multivariate Archive](http://www.timeseriesclassification.com/)

### Survey Papers
- [Foundation Models for Time Series](https://arxiv.org/abs/2403.14735)
- [Time Series Forecasting Survey](https://arxiv.org/abs/2405.02358)

---

## 💡 Tips for Success

1. **Start Simple:** Begin with pre-trained models before training from scratch
2. **Read Actively:** Take notes, summarize, and question as you read
3. **Implement Incrementally:** Build small components and test frequently
4. **Visualize Everything:** Plots reveal patterns tables hide
5. **Collaborate:** Join research communities and discuss ideas
6. **Document Thoroughly:** Future you will thank present you
7. **Reproduce Results:** Validate your implementation against benchmarks
8. **Stay Updated:** Follow key researchers on Twitter/GitHub

---

## 🎯 Success Criteria

You know you're on the right track when:

- [ ] You can explain what TSFMs are to a non-expert
- [ ] You've successfully run zero-shot forecasting with at least one model
- [ ] You understand the difference between attention and causal importance
- [ ] You've implemented at least one interpretability method
- [ ] You can identify open problems in the field
- [ ] You've read and understood TimeSAE paper
- [ ] You have ideas for extending current research

---

## 📞 Getting Help

### Stack Overflow
- Tags: `[time-series]`, `[transformer]`, `[interpretability]`

### GitHub Issues
- Check model repositories for common issues
- Search closed issues before opening new ones

### Research Forums
- **OpenReview** - Discuss papers with authors
- **Reddit r/MachineLearning** - Community Q&A
- **Twitter/X** - Follow researchers for updates

### Academic Support
- Contact authors via email (polite, specific questions)
- Attend conference Q&A sessions
- Join university reading groups

---

**Good luck with your journey into mechanistic interpretability for time series foundation models!**

Remember: The field is rapidly evolving. What's state-of-the-art today may be obsolete in 6 months. Stay curious, stay learning, and contribute back to the community when you can.

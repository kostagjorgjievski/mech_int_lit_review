# Multi-Layer Circuit Analysis in Time Series Foundation Models
## A Deep Research Expansion

### Executive Summary

This document provides an in-depth research agenda for investigating how neural circuits compose across multiple layers in Time Series Foundation Models (TSFMs). Unlike current mechanistic interpretability work that focuses on single-layer analysis, this research direction aims to understand the emergence, composition, and function of complex circuits that span the entire depth of modern time series models.

---

## 1. Theoretical Framework

### 1.1 Why Multi-Layer Circuits Matter

Current mechanistic interpretability research in time series models predominantly focuses on:
- Individual attention heads and their patterns
- Single-layer neuron activations
- Local feature extraction mechanisms

However, this approach misses the fundamental insight from deep learning theory: **representation emerges through layer-wise composition**. In time series models specifically:

1. **Hierarchical Feature Extraction**: Lower layers detect basic patterns (trends, seasonality, anomalies), while higher layers compose these into complex concepts (regime changes, causal relationships, long-term dependencies)

2. **Information Refinement**: Each layer progressively refines and abstracts temporal representations, similar to how vision models build from edges → shapes → objects

3. **Cross-Layer Dependencies**: Circuits often span multiple layers, with information flowing through specific paths rather than being processed independently at each layer

### 1.2 Core Hypotheses

**Hypothesis 1: Compositionality**
*Complex time series understanding emerges from predictable compositions of simple circuits across layers. Lower-layer circuits (primitive detectors) feed into higher-layer circuits (compositional reasoners) in systematic ways.*

**Hypothesis 2: Specialization**
*Different layers specialize in different temporal abstractions. Early layers handle local patterns (immediate temporal context), middle layers handle medium-range dependencies (intermediate temporal reasoning), and late layers handle global structure (long-horizon forecasting and causal inference).*

**Hypothesis 3: Universality**
*Certain circuit motifs are universal across TSFM architectures, representing fundamental time series operations (trend extraction, seasonal decomposition, change point detection) that all models must implement.*

---

## 2. Methodological Foundations

### 2.1 Circuit Definition in Multi-Layer Context

A **multi-layer circuit** is defined as:
```
C = {(L₁, N₁, A₁), (L₂, N₂, A₂), ..., (Lₖ, Nₖ, Aₖ), E}
```
Where:
- `Lᵢ` = Layer index
- `Nᵢ` = Set of neurons/attention heads at layer Lᵢ
- `Aᵢ` = Activation conditions or computation performed
- `E` = Set of edges (connections) between components

**Key Properties:**
1. **Causality**: Components in E must have a causal relationship (earlier layers affect later layers)
2. **Minimality**: C contains only necessary components for the computation
3. **Completeness**: C fully explains a specific behavior or capability

### 2.2 Analytical Techniques

#### A. Path Patching (Multi-Layer)

**Goal**: Trace specific information flow across layers

**Method:**
1. **Intervention**: For a given input example, identify a specific feature or piece of information (e.g., "the trend component")
2. **Corruption**: Add noise to that feature at layer L
3. **Patching**: Restore the feature using activations from a different input at layer L' > L
4. **Measurement**: Observe if the behavior is restored at the output

**Analysis:**
- If patching at layer L₁ restores behavior → Information flows through L₁
- If patching requires layer L₁ AND L₂ → Circuit spans both layers
- Path tracing identifies critical layers for specific features

**Extension to Time Series:**
```python
# Pseudocode for multi-layer path patching
def path_patching(model, input_series, feature_of_interest, target_layers):
    # Forward pass with clean input
    clean_activations = forward_pass(model, input_series)

    # Forward pass with corrupted feature
    corrupted_input = corrupt_feature(input_series, feature_of_interest)
    corrupted_output = forward_pass(model, corrupted_input)

    results = {}
    for layer in target_layers:
        # Patch activations at this layer
        patched_output = forward_with_patch(
            model,
            corrupted_input,
            layer,
            clean_activations[layer]
        )
        # Measure restoration
        restoration = similarity(patched_output, clean_output)
        results[layer] = restoration

    return results
```

#### B. Circuit Induction

**Goal**: Discover repeating circuit patterns that perform specific computations

**Method:**
1. **Behavioral Clustering**: Group examples that exhibit similar behaviors (e.g., "detects seasonality of period 7")
2. **Activation Analysis**: For each cluster, analyze which components activate across layers
3. **Pattern Mining**: Use subgraph isomorphism to find common activation patterns
4. **Validation**: Test if induced circuits are causally responsible

**Example Circuit Discovery:**
- **Seasonality Detection Circuit** might involve:
  - Layer 2: Attention heads attending to lag-k positions (k = candidate period)
  - Layer 4: Neurons that compute periodicity scores
  - Layer 6: Attention heads that weight seasonality component
  - Layer 8: Output neurons that scale seasonal component

#### C. Layer-wise Relevance Propagation (LRP)

**Goal**: Attribute predictions to specific layers and components

**Method:**
1. **Backward Propagation**: Start from output prediction
2. **Redistribution**: Assign relevance to each layer based on contribution
3. **Conservation**: Ensure total relevance is conserved across redistribution
4. **Layer Analysis**: Identify which layers contribute most to specific predictions

**Time Series Adaptation:**
- Relevance for trend prediction → Early layers (local pattern extraction)
- Relevance for long-horizon forecasting → Late layers (global reasoning)
- Relevance for anomaly detection → Middle layers (deviation computation)

#### D. Causal Scrubbing (Multi-Layer)

**Goal**: Test if a hypothesized circuit is sufficient and necessary

**Method:**
1. **Hypothesis**: Propose a circuit C for behavior B
2. **Ablation**: Randomize components in C on input examples
3. **Intervention**: Replace with activations from counterfactual examples
4. **Validation**: If behavior B degrades in predictable ways, circuit is validated

**Multi-Layer Extension:**
- Scrub across multiple layers simultaneously
- Test if information at layer L is necessary for computation at layer L+1
- Validate interdependencies between layers

---

## 3. Research Questions (Deep Dive)

### 3.1 How do features compose across layers?

**Sub-questions:**

1. **Feature Transformation Trajectory**
   - How does a primitive feature (e.g., "autocorrelation at lag 1") transform across layers?
   - Do features become more abstract, more specific, or change in nature?
   - What mathematical operations transform features between layers?

2. **Composition Rules**
   - Are there algebraic rules governing feature composition?
   - Can we predict higher-layer features from lower-layer features?
   - Do features combine additively, multiplicatively, or through more complex operations?

3. **Feature Emergence**
   - At which layer do complex features (e.g., "regime change point") first emerge?
   - Are emergent features predictable from architecture or learned from data?
   - Do different models develop similar features at similar layers?

4. **Feature Interaction**
   - How do features from different pathways interact within a layer?
   - Do features compete or cooperate in influencing downstream computation?
   - Are there gating mechanisms that control feature flow?

**Investigation Approach:**
```python
# Example: Tracking a feature across layers
def track_feature_across_layers(model, input_series, feature_definition):
    """
    Track how a specific feature is represented across all layers
    """
    feature_trajectory = []

    for layer in model.layers:
        # Get activations at this layer
        activations = get_layer_activations(model, input_series, layer)

        # Measure presence/similarity to feature definition
        feature_strength = compute_feature_similarity(
            activations,
            feature_definition
        )

        # Find which components (neurons/heads) represent feature
        components = find_responsible_components(
            activations,
            feature_definition
        )

        feature_trajectory.append({
            'layer': layer,
            'strength': feature_strength,
            'components': components,
            'representation': extract_representation(activations, components)
        })

    return feature_trajectory
```

### 3.2 What circuits emerge for specific time series patterns?

**Sub-questions:**

1. **Primitive Pattern Circuits**
   - **Trend Extraction**: How do models detect and extract trends (linear, nonlinear, piecewise)?
   - **Seasonality Detection**: How are periodic patterns identified and decomposed?
   - **Anomaly Detection**: How do models identify deviations from expected patterns?
   - **Change Point Detection**: How are regime transitions identified?

2. **Composite Pattern Circuits**
   - **Trend-Seasonality Interaction**: How do models separate and combine trend and seasonal components?
   - **Multi-seasonality**: How are multiple seasonal periods (e.g., daily + weekly) handled?
   - **Volatility Clustering**: How do GARCH-like behaviors emerge?
   - **Causal Relationships**: How do models learn lead-lag relationships?

3. **Domain-Specific Pattern Circuits**
   - **Financial Time Series**: Momentum, mean-reversion, volatility patterns
   - **Healthcare**: Physiological cycles, abnormal patterns
   - **IoT/Sensors**: Equipment degradation, fault signatures
   - **Forecasting**: Long-horizon dependencies, uncertainty quantification

**Example: Seasonality Detection Circuit**

Hypothesized structure:
```
Layer 1-2 (Pattern Detection):
- Attention heads attending to positions at lag k, 2k, 3k, ...
- Neurons activated by repeated patterns

Layer 3-4 (Period Estimation):
- Neurons computing cross-correlation with candidate periods
- Attention heads comparing multiple periodicities

Layer 5-6 (Component Extraction):
- Neurons separating seasonal from residual
- Attention heads weighting seasonal component appropriately

Layer 7-8 (Integration):
- Neurons combining seasonal with trend
- Output neurons producing final forecast
```

**Investigation Strategy:**
1. Create synthetic datasets with known patterns (controlled seasonality)
2. Identify circuits that activate on these patterns
3. Validate circuits using causal interventions
4. Test generalization to real-world data

### 3.3 Can we identify universal circuits across TSFMs?

**Sub-questions:**

1. **Architectural Universals**
   - Do all Transformer-based TSFMs develop similar attention patterns?
   - Are there circuits specific to certain architectures (e.g., patching vs. tokenization)?
   - How does architectural choice affect circuit formation?

2. **Functional Universals**
   - Do all models implement similar circuits for fundamental operations (detrending, differencing)?
   - Are there "optimal" circuits that models converge on?
   - Can we predict circuit structure from task requirements?

3. **Cross-Model Circuit Transfer**
   - Can circuits discovered in one model be found in another?
   - Can we use circuits from a small model to understand a large model?
   - What determines circuit transferability?

4. **Training Dynamics**
   - How do circuits evolve during training?
   - Do circuits form in a predictable order (simple → complex)?
   - Can we influence circuit formation through regularization or architecture?

**Investigation Approach:**
```python
# Cross-model circuit comparison
def compare_circuits_across_models(models, dataset, behavior):
    """
    Discover if similar circuits exist across different models
    """
    discovered_circuits = {}

    for model in models:
        # Find circuit for behavior in this model
        circuit = discover_circuit(model, dataset, behavior)
        discovered_circuits[model.name] = circuit

    # Compare circuit structures
    similarities = compare_circuit_structures(discovered_circuits)

    # Identify universal patterns
    universal_patterns = extract_common_patterns(
        discovered_circuits,
        similarities
    )

    return universal_patterns
```

---

## 4. Experimental Design

### 4.1 Datasets

#### Synthetic Datasets (For Controlled Experiments)

**Purpose**: Create datasets with known ground-truth patterns to validate circuit discovery methods.

**Categories:**

1. **ARIMA-based Series**:
   - AR(p): Test lag dependence circuits
   - MA(q): Test moving average circuits
   - ARMA(p,q): Test combined circuits
   - ARIMA(p,d,q): Test differencing circuits

2. **Seasonal Series**:
   - Single seasonality: sin(2πt/p) + noise
   - Multiple seasonality: sin(2πt/p₁) + sin(2πt/p₂) + noise
   - Amplitude modulation: (1 + 0.5sin(2πt/p₁)) × sin(2πt/p₂)

3. **Regime-Switching Series**:
   - Piecewise constant: Step functions with noise
   - Markov switching: Hidden Markov Model outputs
   - Structural breaks: Sudden parameter changes

4. **Chaotic Series**:
   - Logistic map: x_{t+1} = r × x_t × (1 - x_t)
   - Lorenz system: Multivariate chaotic dynamics

**Dataset Generation Code:**
```python
def generate_synthetic_dataset(series_type, n_samples=10000, length=256):
    """
    Generate synthetic time series with known patterns
    """
    if series_type == 'arima':
        # ARIMA(2,1,2) with known parameters
        return generate_arima([0.5, -0.2], [0.3, -0.1], d=1,
                             n_samples=n_samples, length=length)

    elif series_type == 'seasonal':
        # Dual seasonality: period 7 and 365
        t = np.arange(length)
        series = (np.sin(2*np.pi*t/7) +
                 0.5*np.sin(2*np.pi*t/365) +
                 np.random.normal(0, 0.1, length))
        return np.tile(series, (n_samples, 1))

    elif series_type == 'regime_switching':
        # 3 regimes with different AR parameters
        return generate_markov_switching(
            regimes=[
                {'ar': [0.9], 'var': 0.1},
                {'ar': [-0.5], 'var': 0.2},
                {'ar': [0.3], 'var': 0.15}
            ],
            transition_matrix=[[0.95, 0.03, 0.02],
                             [0.02, 0.95, 0.03],
                             [0.03, 0.02, 0.95]],
            n_samples=n_samples,
            length=length
        )

    # ... other types
```

#### Real-World Datasets

**Benchmarks:**

1. **Monash Forecasting Repository**: 40+ diverse datasets
2. **M4 Competition**: 100,000 time series across domains
3. **ETT (Electricity Transformer Temperature)**: Multivariate sensor data
4. **Weather**: Temperature, precipitation, etc.
5. **Financial**: Stock prices, exchange rates, volatility
6. **Healthcare**: ECG, EEG, physiological signals

**Selection Criteria:**
- Diversity in patterns (trend, seasonality, chaos)
- Different lengths (short, medium, long)
- Different frequencies (high, medium, low)
- Different domains (business, science, engineering)

### 4.2 Models

**Models for Analysis:**

1. **TimesFM** (Google):
   - 200M - 2B parameters
   - Trained on massive diverse time series data
   - Multiple layers for circuit analysis

2. **MOMENT** (CMU):
   - Family of models (small, base, large)
   - Different sizes enable circuit scaling analysis

3. **Lag-Llama**:
   - Probabilistic foundation model
   - Different architecture for comparison

4. **Chronos** (Amazon):
   - Token-based approach
   - Different tokenization strategy

5. **TimeGPT** (Nixtla):
   - Commercial model
   - If accessible

**Model Analysis Pipeline:**
```python
def analyze_model_circuits(model, datasets):
    """
    Comprehensive circuit analysis for a single model
    """
    results = {
        'layer_wise_analysis': {},
        'cross_layer_circuits': {},
        'universal_patterns': {}
    }

    # 1. Layer-wise feature analysis
    for layer_idx in range(model.num_layers):
        results['layer_wise_analysis'][layer_idx] = \
            analyze_layer_features(model, layer_idx, datasets)

    # 2. Cross-layer circuit discovery
    for behavior in ['trend', 'seasonality', 'anomaly', 'forecasting']:
        results['cross_layer_circuits'][behavior] = \
            discover_multi_layer_circuit(model, behavior, datasets)

    # 3. Universal pattern identification
    results['universal_patterns'] = \
        identify_universal_circuits(model, datasets)

    return results
```

### 4.3 Evaluation Metrics

#### Circuit Quality Metrics

1. **Circuit Completeness**:
   - Definition: How much of the behavior does the circuit explain?
   - Measurement: Performance of scrubbed model (with circuit ablated)
   - Formula: `Completeness = 1 - (Performance_scrubbed / Performance_full)`

2. **Circuit Fidelity**:
   - Definition: Does the circuit produce the same outputs as the full model?
   - Measurement: Output similarity when only circuit is active
   - Formula: `Fidelity = similarity(Output_circuit, Output_full)`

3. **Circuit Minimality**:
   - Definition: Is the circuit minimal (no unnecessary components)?
   - Measurement: Number of components normalized by performance
   - Formula: `Minimality = Performance / Components`

4. **Circuit Stability**:
   - Definition: Is the circuit consistent across different inputs?
   - Measurement: Variance in circuit activation across inputs
   - Formula: `Stability = 1 - Var(Activation)`

#### Cross-Model Transfer Metrics

1. **Circuit Transferability**:
   - Definition: Can a circuit from one model explain behavior in another?
   - Measurement: Performance when applying circuit from Model A to Model B
   - Formula: `Transfer = Performance_B_with_A_circuit / Performance_B_native`

2. **Structural Similarity**:
   - Definition: Are circuit structures similar across models?
   - Measurement: Graph similarity between circuit graphs
   - Methods: Graph edit distance, subgraph isomorphism

#### Prediction Accuracy Metrics

1. **Task-Specific Metrics**:
   - Forecasting: MSE, MAE, MAPE, sMAPE
   - Anomaly Detection: Precision, Recall, F1
   - Classification: Accuracy, AUC

2. **Circuit-Guided Prediction**:
   - Does understanding the circuit improve predictions?
   - Can we create simpler models using discovered circuits?

---

## 5. Technical Implementation

### 5.1 Multi-Layer Activation Extraction

**Challenge**: Efficiently extract and store activations from all layers

**Solution**:
```python
import torch
import torch.nn as nn

class ActivationExtractor:
    """
    Extract activations from all layers of a time series model
    """
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

        # Register hooks for all layers
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                # Detach to save memory
                self.activations[name] = output.detach()
            return hook

        # Register hook for each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer,
                                   nn.MultiheadAttention,
                                   nn.Linear)):
                hook = module.register_forward_hook(
                    get_activation(name)
                )
                self.hooks.append(hook)

    def __call__(self, x):
        self.activations = {}
        with torch.no_grad():
            _ = self.model(x)
        return self.activations

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# Usage
extractor = ActivationExtractor(model)
activations = extractor(time_series_batch)

# Access layer-specific activations
layer_2_activations = activations['layer.2']
attention_outputs = activations['layer.5.self_attn']
```

### 5.2 Path Patching Implementation

```python
def multi_layer_path_patching(model, clean_input, corrupted_input,
                              layer_to_patch, component_to_patch):
    """
    Patch activations at specific layer and component
    """
    # Get clean activations
    with torch.no_grad():
        clean_output, clean_cache = model.forward_with_cache(clean_input)

    # Get corrupted output
    with torch.no_grad():
        corrupted_output, corrupted_cache = \
            model.forward_with_cache(corrupted_input)

    # Patch at specific layer
    def patch_fn(module, input, output):
        if module.name == layer_to_patch:
            # Replace specific component (e.g., attention head)
            output[component_to_patch] = \
                clean_cache[layer_to_patch][component_to_patch]
        return output

    # Register patch hook
    patch_hook = model.get_submodule(layer_to_patch).\
                 register_forward_hook(patch_fn)

    # Forward pass with patch
    with torch.no_grad():
        patched_output = model(corrupted_input)

    # Remove hook
    patch_hook.remove()

    # Compute restoration effect
    restoration = compute_restoration(
        clean_output,
        corrupted_output,
        patched_output
    )

    return restoration
```

### 5.3 Circuit Discovery Algorithm

```python
from typing import List, Dict, Set
import networkx as nx

def discover_multi_layer_circuit(model, behavior_examples,
                               non_behavior_examples,
                               max_layers=None):
    """
    Discover multi-layer circuit responsible for a specific behavior
    """
    circuit = nx.DiGraph()

    # 1. Activation analysis
    behavior_activations = []
    for example in behavior_examples:
        acts = get_all_layer_activations(model, example)
        behavior_activations.append(acts)

    non_behavior_activations = []
    for example in non_behavior_examples:
        acts = get_all_layer_activations(model, example)
        non_behavior_activations.append(acts)

    # 2. Component importance per layer
    for layer in range(model.num_layers):
        # Find components that differentiate behavior
        important_components = find_important_components(
            behavior_activations,
            non_behavior_activations,
            layer
        )

        # Add to circuit
        for comp in important_components:
            node_id = f"{layer}.{comp}"
            circuit.add_node(node_id,
                           layer=layer,
                           component=comp,
                           importance=comp.importance)

    # 3. Find connections between layers
    for layer in range(model.num_layers - 1):
        current_layer_components = [n for n in circuit.nodes()
                                   if n.startswith(f"{layer}.")]
        next_layer_components = [n for n in circuit.nodes()
                                if n.startswith(f"{layer+1}.")]

        # Test connectivity via path patching
        for curr_node in current_layer_components:
            for next_node in next_layer_components:
                # Patch current component and see if next is affected
                connection_strength = test_connection(
                    model,
                    curr_node,
                    next_node,
                    behavior_examples
                )

                if connection_strength > threshold:
                    circuit.add_edge(curr_node, next_node,
                                   strength=connection_strength)

    # 4. Prune circuit
    circuit = prune_circuit(circuit, model, behavior_examples)

    return circuit

def prune_circuit(circuit, model, examples):
    """
    Remove unnecessary components from circuit
    """
    # Start with full circuit
    current_circuit = circuit.copy()

    # Iteratively remove least important components
    while True:
        # Find least important node
        least_important = min(current_circuit.nodes(),
                            key=lambda n: current_circuit.nodes[n]['importance'])

        # Temporarily remove
        test_circuit = current_circuit.copy()
        test_circuit.remove_node(least_important)

        # Test if circuit still works
        if test_circuit_completeness(test_circuit, model, examples) > 0.9:
            # Keep removal
            current_circuit = test_circuit
        else:
            # Stop pruning
            break

    return current_circuit
```

### 5.4 Cross-Model Circuit Alignment

```python
def align_circuits_across_models(circuit_a, circuit_b, model_a, model_b):
    """
    Find correspondence between circuits in different models
    """
    # 1. Structural alignment
    structural_score = graph_similarity(circuit_a, circuit_b)

    # 2. Functional alignment
    functional_score = 0
    for example in test_examples:
        # Get outputs from circuit in model A
        output_a = run_circuit(model_a, circuit_a, example)

        # Get outputs from corresponding components in model B
        output_b = run_circuit(model_b, circuit_b, example)

        functional_score += similarity(output_a, output_b)

    functional_score /= len(test_examples)

    # 3. Combined alignment score
    alignment_score = 0.5 * structural_score + 0.5 * functional_score

    return alignment_score
```

---

## 6. Expected Challenges and Solutions

### 6.1 Computational Challenges

**Challenge**: Processing activations from all layers for many examples is expensive

**Solutions:**
1. **Gradient-based selection**: Only analyze examples that activate circuit strongly
2. **Layer-wise caching**: Cache activations to avoid recomputation
3. **Sparse analysis**: Focus on subset of important components per layer
4. **Dimensionality reduction**: Use PCA or autoencoders to compress activations

**Implementation:**
```python
def efficient_activation_analysis(model, examples, budget=1000):
    """
    Efficiently analyze activations within computational budget
    """
    # 1. Gradient-based example selection
    selected_examples = select_examples_by_gradient(
        model, examples, n_samples=budget // 10
    )

    # 2. Analyze selected examples in detail
    detailed_activations = []
    for example in selected_examples:
        acts = get_all_layer_activations(model, example)
        detailed_activations.append(acts)

    # 3. For remaining examples, use sparse analysis
    sparse_activations = []
    for example in examples:
        # Only extract from important layers
        acts = get_important_layer_activations(model, example)
        sparse_activations.append(acts)

    return detailed_activations, sparse_activations
```

### 6.2 Identifying True Causality

**Challenge**: Correlation doesn't imply causation in circuit discovery

**Solutions:**
1. **Ablation studies**: Systematically ablate components and measure effects
2. **Intervention experiments**: Patch activations with controlled values
3. **Counterfactual analysis**: Test on adversarial examples
4. **Causal scrubbing**: Rigorous validation of circuit sufficiency

### 6.3 Complexity of Multi-Layer Interactions

**Challenge**: Interactions across layers may be non-linear and hard to trace

**Solutions:**
1. **Decomposition approach**: Break complex circuits into simpler sub-circuits
2. **Layer-wise analysis**: Understand each layer before studying interactions
3. **Information flow visualization**: Use tools like attention flow graphs
4. **Theoretical constraints**: Use domain knowledge (e.g., time series properties) to constrain search

### 6.4 Generalization Across Domains

**Challenge**: Circuits learned in one domain may not transfer to others

**Solutions:**
1. **Diverse training data**: Train on multiple domains
2. **Domain adaptation**: Adapt circuits to new domains
3. **Meta-learning**: Learn how to learn circuits
4. **Transfer learning analysis**: Study when and why circuits transfer

---

## 7. Expected Outcomes and Impact

### 7.1 Scientific Contributions

1. **Catalog of Time Series Circuits**
   - Comprehensive library of identified circuits
   - Categorized by function (trend, seasonality, etc.)
   - Annotated with layer locations and interactions

2. **Theory of Circuit Composition**
   - Mathematical framework for how circuits compose
   - Predictive models of circuit formation
   - Understanding of scaling laws for circuits

3. **Cross-Model Circuit Taxonomy**
   - Universal circuits across architectures
   - Architecture-specific circuit variations
   - Circuit transferability principles

4. **Methodological Advances**
   - Improved tools for multi-layer circuit analysis
   - Benchmarks for circuit discovery
   - Open-source toolkits for the community

### 7.2 Practical Applications

1. **Model Improvement**
   - Identify and remove redundant circuits
   - Enhance important circuits through fine-tuning
   - Design architectures that encourage useful circuits

2. **Efficient Model Design**
   - Create smaller models by pruning unnecessary circuits
   - Transfer circuits from large to small models
   - Design specialized models for specific tasks

3. **Interpretability and Trust**
   - Explain model decisions in terms of circuits
   - Debug model failures by tracing circuit breakdown
   - Provide guarantees about model behavior

4. **Knowledge Discovery**
   - Learn new time series analysis techniques from circuits
   - Discover patterns humans might miss
   - Generate hypotheses about time series phenomena

### 7.3 Publication Strategy

**Target Venues:**

1. **Machine Learning Conferences**:
   - NeurIPS: Main theoretical contributions
   - ICML: Methodological advances
   - ICLR: Circuit discovery and interpretation

2. **Domain-Specific Venues**:
   - KDD (Data Mining): Time series applications
   - IJCAI (AI): Applied AI aspects
   - SIGMOD/ICDE (Databases): Large-scale aspects

3. **Journals**:
   - JMLR: Full theoretical treatment
   - IEEE TKDE: Time series focus
   - Nature Machine Intelligence: Broader impact

**Publication Timeline:**
- Year 1: Methodology papers (tools and techniques)
- Year 2: Circuit discovery papers (specific circuits)
- Year 3: Theory and synthesis papers (unified framework)

---

## 8. Collaboration and Resources

### 8.1 Potential Collaborators

**Mechanistic Interpretability Community:**
- Anthropic (Interpretability team)
- OpenAI (Superalignment)
- Redwood Research
- Conjecture

**Time Series Research Community:**
- CMU (MOMENT authors)
- Google (TimesFM team)
- Amazon (Chronos team)
- Monash University (Forecasting experts)

**Neuroscience Inspiration:**
- Researchers studying neural circuits
- Brain mapping initiatives
- Computational neuroscience labs

### 8.2 Required Resources

**Computational:**
- GPU cluster for training and analysis
- Storage for activations (TB-scale)
- Distributed computing framework

**Data:**
- Access to large-scale time series datasets
- Synthetic data generation infrastructure
- Domain-specific datasets (healthcare, finance, etc.)

**Software:**
- PyTorch/JAX for model implementation
- NetworkX for graph analysis
- Custom circuit analysis tools

**Personnel:**
- Research scientists (2-3)
- Research engineers (1-2)
- Graduate students (2-3)

### 8.3 Funding Opportunities

1. **NSF**: AI research grants
2. **DARPA**: Explainable AI programs
3. **Industry partnerships**: Google, Amazon, etc.
4. **Philanthropic**: Long-term AI safety funding

---

## 9. Ethical Considerations and Risks

### 9.1 Dual-Use Concerns

**Risk**: Understanding circuits could enable:
- More effective adversarial attacks
- Manipulation of time series systems
- Exploitation of financial markets

**Mitigation**:
- Responsible disclosure practices
- Focus on defensive applications
- Engagement with safety community

### 9.2 Privacy Risks

**Risk**: Circuit analysis might extract:
- Sensitive patterns in private data
- Trade secrets from proprietary time series
- Individual behaviors from sensor data

**Mitigation**:
- Use synthetic data when possible
- Anonymize real data
- Follow data governance best practices

### 9.3 Misinterpretation Risks

**Risk**: Circuits might be:
- Over-interpreted (finding patterns that aren't real)
- Used to justify incorrect decisions
- Applied beyond their valid domain

**Mitigation**:
- Rigorous validation methodology
- Clear communication of uncertainty
- Peer review and replication

---

## 10. Summary and Next Steps

### 10.1 Key Takeaways

Multi-layer circuit analysis in TSFMs represents a significant advance in mechanistic interpretability. By moving beyond single-layer analysis to understand how circuits compose across depth, we can:

1. **Discover fundamental computational primitives** used across time series models
2. **Understand the emergence of complex behaviors** from simple circuits
3. **Improve model design** based on circuit principles
4. **Enable interpretation and debugging** of model decisions
5. **Advance both theory and practice** of time series analysis

### 10.2 Immediate Next Steps

1. **Literature Review** (2 weeks):
   - Deep dive into existing mechanistic interpretability work
   - Review time series model architectures
   - Study relevant neuroscience literature

2. **Tool Development** (1 month):
   - Implement activation extraction tools
   - Build path patching framework
   - Create circuit discovery algorithms

3. **Pilot Experiments** (2 months):
   - Start with simple synthetic data
   - Analyze circuits in small models
   - Validate methodology

4. **Scale Up** (6 months):
   - Analyze large-scale models (TimesFM, MOMENT)
   - Test on diverse real-world datasets
   - Build comprehensive circuit catalog

5. **Theory Development** (ongoing):
   - Develop mathematical framework
   - Prove theoretical properties
   - Publish findings

### 10.3 Long-Term Vision

The ultimate goal is to develop a **complete theory of mechanistic time series understanding** that explains how TSFMs process temporal information at the circuit level. This would enable:

- Predictable model behavior based on circuit analysis
- Design principles for better time series models
- Automated discovery of time series patterns
- Human-AI collaboration in time series analysis
- Trustworthy deployment in critical applications

This research sits at the intersection of mechanistic interpretability, time series analysis, and deep learning theory, with potential to advance all three fields while providing practical benefits for real-world applications.

---

## Appendix A: Detailed Experimental Protocols

### A.1 Path Patching Experiments

**Protocol for Information Flow Analysis:**

```python
def path_patching_protocol(model, dataset):
    """
    Standardized protocol for multi-layer path patching experiments
    """
    results = []

    for example in dataset:
        # 1. Identify feature of interest
        feature = identify_feature(example)  # e.g., "trend"

        # 2. Clean forward pass
        clean_output, clean_cache = model.forward_with_cache(example)

        # 3. Corrupt feature
        corrupted_example = corrupt_feature(example, feature)
        corrupted_output, corrupted_cache = \
            model.forward_with_cache(corrupted_example)

        # 4. Patch at each layer
        for layer in range(model.num_layers):
            for component in ['attention', 'mlp']:
                patched_output = patch_component(
                    model,
                    corrupted_example,
                    layer,
                    component,
                    clean_cache[layer][component]
                )

                # Measure restoration
                restoration = compute_restoration(
                    clean_output,
                    corrupted_output,
                    patched_output
                )

                results.append({
                    'example': example,
                    'feature': feature,
                    'layer': layer,
                    'component': component,
                    'restoration': restoration
                })

    return analyze_results(results)
```

### A.2 Circuit Induction Experiments

**Protocol for Discovering Repeating Patterns:**

```python
def circuit_induction_protocol(model, behavior_examples):
    """
    Discover circuits that explain specific behaviors
    """
    # 1. Cluster examples by behavior
    clusters = cluster_by_behavior(behavior_examples)

    discovered_circuits = []

    for cluster_id, examples in clusters.items():
        # 2. Extract activations for all examples in cluster
        all_activations = []
        for example in examples:
            activations = get_all_layer_activations(model, example)
            all_activations.append(activations)

        # 3. Find common activation patterns
        common_patterns = find_common_patterns(all_activations)

        # 4. Induce circuit from patterns
        circuit = induce_circuit(common_patterns)

        # 5. Validate circuit
        validation_score = validate_circuit(
            model,
            circuit,
            examples
        )

        if validation_score > threshold:
            discovered_circuits.append({
                'cluster': cluster_id,
                'circuit': circuit,
                'score': validation_score
            })

    return discovered_circuits
```

---

## Appendix B: Code Repository Structure

```
multi_layer_circuits/
├── README.md
├── setup.py
├── requirements.txt
│
├── src/
│   ├── activation_extraction/
│   │   ├── extractors.py
│   │   ├── cache.py
│   │   └── compression.py
│   │
│   ├── patching/
│   │   ├── path_patching.py
│   │   ├── activation_patching.py
│   │   └── interventions.py
│   │
│   ├── circuit_discovery/
│   │   ├── induction.py
│   │   ├── clustering.py
│   │   ├── validation.py
│   │   └── pruning.py
│   │
│   ├── analysis/
│   │   ├── attribution.py
│   │   ├── visualization.py
│   │   └── metrics.py
│   │
│   ├── models/
│   │   ├── timesfm.py
│   │   ├── moment.py
│   │   ├── lag_llama.py
│   │   └── chronos.py
│   │
│   └── utils/
│       ├── data.py
│       ├── evaluation.py
│       └── plotting.py
│
├── experiments/
│   ├── synthetic/
│   │   ├── generate_data.py
│   │   └── configs/
│   │
│   ├── real_world/
│   │   ├── setup_data.py
│   │   └── configs/
│   │
│   └── configs/
│       ├── path_patching.yaml
│       ├── circuit_induction.yaml
│       └── cross_model.yaml
│
├── notebooks/
│   ├── exploratory/
│   ├── validation/
│   └── tutorials/
│
├── docs/
│   ├── methodology.md
│   ├── results/
│   └── api_reference.md
│
└── tests/
    ├── unit/
    ├── integration/
    └── benchmark/
```

---

## Appendix C: Key References

### Mechanistic Interpretability
1. Olsson et al. (2022) - "In-context Learning and Induction Heads"
2. Olsson et al. (2024) - "Interpretability at Scale"
3. Elhage et al. (2021) - "Mathematical Framework for Transformer Circuits"
4. Nanda (2022) - "A Mechanistic Interpretability Analysis of Grokking"

### Time Series Foundation Models
1. Das et al. (2023) - "TimesFM: Decoding Time-series Dynamics..."
2. Goswami et al. (2023) - "MOMENT: A Family of Open..."
3. Challu et al. (2023) - "Chronos: Learning the Language of Time"
4. Rasul et al. (2023) - "Lag-Llama: Towards Foundation..."

### Neuroscience Inspiration
1. Buzsáki (2019) - "The Brain from Inside Out"
2. Sejnowski (2020) - "The Deep Learning Revolution"
3. Marr (1982) - "Vision: A Computational Investigation"

### Causal Analysis
1. Pearl (2009) - "Causality"
2. VanderWeele (2019) - "Principles of Confounder Selection"
3. Rubin (1974) - "Estimating Causal Effects"

---

**End of Document**

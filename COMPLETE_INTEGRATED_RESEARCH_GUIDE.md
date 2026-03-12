# Multi-Layer Circuit Analysis for Time Series Foundation Models
## A Comprehensive Research Guide

**Status:** High-Priority Research Direction
**Last Updated:** March 11, 2026
**Focus:** Understanding how features compose across layers in TSFMs

---

## Executive Summary

Multi-layer circuit analysis represents the frontier of mechanistic interpretability for Time Series Foundation Models (TSFMs). While current work focuses on single-layer analysis, complex temporal reasoning emerges from multi-layer interactions. This guide provides a comprehensive framework for discovering, analyzing, and validating circuits that span multiple layers.

**Core Thesis:** Temporal patterns—trend detection, seasonality, regime changes, anomalies—are computed through distributed circuits across layers. Understanding these circuits enables explainable forecasting, improved robustness, and model editing without retraining.

**Key Contributions:**
1. Unified theoretical framework combining information theory, graph theory, and complexity theory
2. Practical methodologies with implementation details
3. Critical analysis of assumptions and limitations
4. Concrete research directions and open problems

---

# PART I: THEORETICAL FOUNDATIONS

## 1. What Are Circuits?

### 1.1 Definition and Core Concepts

**Circuit:** A minimal computational subgraph that implements a specific behavior, consisting of:
- **Components:** Neurons, attention heads, layers
- **Connections:** Weight matrices, attention patterns
- **Information flow:** How activations propagate

**Key Properties:**
- **Compositionality:** Complex behaviors build from simple primitives
- **Reuse:** Circuits can be shared across tasks
- **Independence:** Different behaviors use different circuit components
- **Causality:** Intervening on circuit components affects behavior

### 1.2 Why Multi-Layer Analysis?

**Single-Layer Limitations:**
1. **Feature Composition:** Individual features may be meaningless; meaning emerges from combinations
2. **Sequential Processing:** Temporal hierarchies require layer-wise processing
3. **Counterfactual Understanding:** Need to trace what-if scenarios across computation

**Multi-Layer Advantages:**
1. **Complete Causal Chains:** From input features to predictions
2. **Feature Discovery:** Find intermediate representations
3. **Behavior Localization:** Identify where specific computation happens
4. **Universality:** Discover whether all TSFMs use similar circuits

### 1.3 Circuit Hypotheses for TSFMs

**Hypothesis 1: Temporal Abstraction Hierarchy**
- **Early layers (1-6):** Local patterns, noise filtering, basic arithmetic
- **Middle layers (7-14):** Temporal windows, seasonality, trend components
- **Late layers (15-24):** Global context, long-term dependencies, decision-making

**Hypothesis 2: Functional Specialization**
Different attention heads specialize:
- **Induction heads:** Pattern repetition detection
- **Positional heads:** Temporal distance computation
- **Aggregation heads:** Summary statistics

**Hypothesis 3: Cross-Model Universality**
- Fundamental temporal operations use similar circuits across TSFMs
- Architecture differences matter less than functional requirements
- Transferable circuit library possible

### 1.4 Information-Theoretic Framework

**Circuit Information Capacity:**
```
I(Circuit; Output) = H(Output) - H(Output | Circuit)
```

Where:
- I(Circuit; Output) = Mutual information between circuit and output
- H(Output) = Entropy of model output
- H(Output | Circuit) = Conditional entropy given circuit

**Key Insight:** A complete circuit maximizes mutual information with behavior while minimizing circuit size (rate-distortion theory).

**Applications:**
- **Circuit compression:** Remove redundant components with low MI
- **Layer importance:** Rank layers by information contribution
- **Architecture design:** Design models with optimal information flow

### 1.5 Graph-Theoretic Perspective

**Circuits as Directed Acyclic Graphs (DAGs):**
- **Nodes:** Components (attention heads, MLP neurons)
- **Edges:** Information flow (weighted connections)
- **Properties:** Topological ordering, circuit depth, connectivity

**Graph Metrics:**
1. **Circuit Centrality:** Which components are most critical?
2. **Circuit Modularity:** Are circuits modular or integrated?
3. **Circuit Robustness:** How robust is circuit to damage?

---

# PART II: CRITICAL PERSPECTIVES

## 2. What If Circuits Don't Exist?

### 2.1 Skeptical Hypothesis: Distributed Representation

**The Concern:** Temporal computation may be highly distributed with no clear circuit boundaries—everything connects to everything.

**Evidence For:**
- High connectivity in transformers
- Redundancy in neural networks
- Difficulty finding interpretable features

**Counter-Evidence:**
- Anthropic's success finding circuits in LLMs
- Modular structure in trained networks
- Successful interventions on specific components

**Research Approach:**
```python
def test_circuit_hypothesis(model, behaviors):
    """Empirically test whether circuits exist"""
    results = {'modular': 0, 'distributed': 0, 'hybrid': 0}

    for behavior in behaviors:
        circuit = discover_circuit(model, behavior)
        modularity_score = measure_modularity(circuit)
        intervention_success = test_interventions(circuit)

        if modularity_score > 0.7 and intervention_success > 0.8:
            results['modular'] += 1
        elif modularity_score < 0.3 and intervention_success < 0.5:
            results['distributed'] += 1
        else:
            results['hybrid'] += 1

    return results
```

**Implications:** If circuits don't exist in clear form, we need different interpretability paradigms—focusing on statistical patterns rather than discrete circuits.

### 2.2 Generalization Limits

**Critical Questions:** Do circuits generalize across:
- Different TSFMs?
- Different datasets?
- Different time scales?
- Different domains?

**Expected Findings:**
- **Strong generalization:** Core temporal operations (differences, averages)
- **Moderate generalization:** Domain-specific patterns (seasonality types)
- **Weak generalization:** Dataset-specific features (particular trends)

---

# PART III: RESEARCH QUESTIONS

## 3. Core Research Questions

### Q1: Feature Composition Across Layers
- How do primitive features combine into complex concepts?
- Which layer transitions are most critical?
- Are there "bottleneck" layers where information must flow through?

**Research Approach:**
1. Extract SAE features from all layers
2. Track feature activation trajectories
3. Identify composition rules (e.g., "feature A + feature B → feature C")
4. Validate with causal interventions

### Q2: Circuit Identification for Time Series Patterns
- What circuits implement trend detection, seasonality, regime changes, anomalies?
- Can we automatically discover these circuits?
- How generalizable are they across datasets?

**Research Approach:**
1. Create synthetic datasets with known patterns
2. Use activation patching to trace pattern computation
3. Build circuit library for common operations
4. Test on real-world data

### Q3: Cross-Model Circuit Universality
- Do TimesFM, Chronos, MOMENT share circuits?
- What explains circuit differences?
- Can we transfer circuits between models?

**Research Approach:**
1. Analyze multiple TSFMs on same tasks
2. Use CCA to align representations
3. Identify common circuit motifs
4. Test circuit transfer

### Q4: Causal Validity
- Are identified circuits causally necessary?
- What's the minimal sufficient circuit for each behavior?
- How much redundancy exists?

**Research Approach:**
1. Ablation studies: Remove circuit components
2. Causal scrubbing: Replace with random/noise
3. Performance metrics: Measure prediction impact
4. Dose-response: Vary intervention strength

---

# PART IV: METHODOLOGICAL FRAMEWORK

## 4. Core Methodologies

### 4.1 Path Patching (Activation Patching)

**Purpose:** Trace information flow between layers

**Method:**
1. Run model on input A (baseline)
2. Run model on input B (target behavior)
3. For each layer/component, replace activation from B with A
4. Measure if behavior changes
5. Critical components = behavior changes when patched

**Implementation:**
```python
def path_patching_tsfm(model, input_a, input_b, target_behavior):
    """Trace information flow for specific time series behavior"""
    # Store activations from both inputs
    activations_a = model.get_all_activations(input_a)
    activations_b = model.get_all_activations(input_b)

    critical_components = []

    for layer in model.layers:
        # Patch each component
        for component_type in ['attention', 'mlp']:
            # Patch from A to B
            patched_output = model.forward_with_patch(
                input_b,
                layer=layer,
                component=component_type,
                replacement=activations_a[layer][component_type]
            )

            # Measure behavior change
            behavior_change = target_behavior.measure(patched_output)

            if behavior_change > threshold:
                critical_components.append({
                    'layer': layer,
                    'component': component_type,
                    'importance': behavior_change
                })

    return critical_components
```

### 4.2 Causal Scrubbing

**Purpose:** Test if circuit is sufficient for behavior

**Method:**
1. Identify hypothesized circuit
2. Replace all non-circuit components with noise/random
3. Test if behavior still works
4. If yes, circuit is sufficient

**Implementation:**
```python
def causal_scrub_tsfm(model, circuit, dataset):
    """Test if circuit is sufficient for behavior"""
    scrubbed_model = model.copy()

    # Replace all non-circuit components
    for layer in scrubbed_model.layers:
        if layer.index not in circuit['layers']:
            # Replace with random activations
            layer.forward = lambda x: torch.randn_like(x)

    # Test on dataset
    results = []
    for sample in dataset:
        output = scrubbed_model(sample)
        behavior_score = evaluate_behavior(output, sample)
        results.append(behavior_score)

    return {
        'mean_score': np.mean(results),
        'circuit_sufficient': np.mean(results) > threshold
    }
```

**Validation Metrics:**
- **Sufficiency:** Does scrubbed model maintain behavior?
- **Necessity:** Does ablating circuit destroy behavior?
- **Minimality:** Can we remove components and still work?

### 4.3 Integrated Analysis Pipeline

**Step 1: Circuit Discovery**
```python
def discover_time_series_circuits(model, dataset):
    """Complete pipeline for discovering TSFM circuits"""
    behaviors = ['trend', 'seasonality', 'anomaly', 'forecasting']
    circuits = {}

    for behavior in behaviors:
        # 1. Identify critical components
        critical = path_patching(model, dataset, behavior)

        # 2. Layer-wise relevance
        relevance = layer_relevance(model, dataset, behavior)

        # 3. Combine evidence
        candidates = prioritize_components(critical, relevance)

        # 4. Build circuits
        circuit = build_circuit(candidates, model)

        # 5. Validate
        validated = validate_circuit(model, circuit, dataset, behavior)

        if validated:
            circuits[behavior] = circuit

    return circuits
```

### 4.4 Statistical Rigor

**Multiple Testing Problem:** When testing thousands of components, false positives are likely.

**Solution: False Discovery Rate (FDR) Control**
```python
def circuit_discovery_with_fdr(model, dataset, alpha=0.05):
    """Circuit discovery with statistical rigor"""
    # Test all components
    p_values = {}
    for component in model.all_components:
        p_value = test_component_importance(component, dataset)
        p_values[component] = p_value

    # Apply Benjamini-Hochberg correction
    from statsmodels.stats.multitest import multipletests
    rejected, adjusted_pvals, _, _ = multipletests(
        list(p_values.values()),
        alpha=alpha,
        method='fdr_bh'
    )

    # Select significant components
    significant_components = [
        comp for comp, rej in zip(p_values.keys(), rejected) if rej
    ]

    return significant_components
```

---

# PART V: EXPERIMENTAL DESIGN

## 5. Datasets

### 5.1 Synthetic Datasets (for validation)

**Controlled datasets with known ground truth:**
```python
synthetic_datasets = {
    'trend_synthetic': {
        'description': 'Linear and polynomial trends',
        'patterns': ['linear', 'quadratic', 'exponential', 'logistic'],
        'noise_levels': [0.0, 0.1, 0.3, 0.5]
    },
    'seasonality_synthetic': {
        'description': 'Multiple seasonal periods',
        'patterns': ['daily', 'weekly', 'monthly', 'yearly'],
        'amplitudes': [0.5, 1.0, 2.0, 5.0]
    },
    'regime_change': {
        'description': 'Sudden changes in distribution',
        'change_points': ['abrupt', 'gradual'],
        'pattern_types': ['mean_shift', 'variance_change', 'trend_change']
    }
}
```

### 5.2 Real-World Datasets

**Finance:**
- Stock prices (Yahoo Finance): Daily frequency, multiple assets
- Trading volume, volatility indices

**Weather:**
- NOAA data: Temperature, humidity, pressure, wind speed
- Multiple locations, hourly resolution

**Healthcare:**
- PhysioNet ECG: Arrhythmias, 360Hz sampling rate
- EEG, vital signs monitoring

**Industrial:**
- NASA Bearings: Vibration data, 20kHz sampling
- Predictive maintenance

**Energy:**
- UCI Electricity: 370 clients, 15-minute resolution

## 6. Models

**Target Models for Analysis:**
```python
models = {
    'TimesFM': {
        'source': 'google/timesfm-1.0-200m',
        'layers': 20,
        'parameters': '200M'
    },
    'Chronos': {
        'source': 'amazon/chronos-t5-large',
        'layers': 24,
        'parameters': '710M'
    },
    'MOMENT': {
        'source': 'AutonLab/MOMENT-large',
        'layers': 16,
        'parameters': '386M'
    }
}
```

## 7. Evaluation Metrics

### 7.1 Circuit Quality Metrics
```python
circuit_metrics = {
    'completeness': {
        'definition': 'Fraction of behavior explained by circuit',
        'measure': 'performance_scrubbed / performance_full',
        'target': '> 0.9'
    },
    'minimality': {
        'definition': 'Efficiency of circuit (fewer components = better)',
        'measure': '1 / num_components',
        'target': 'maximize'
    },
    'causal_importance': {
        'definition': 'Performance drop when circuit ablated',
        'measure': 'performance_full - performance_ablated',
        'target': '> 0.5'
    },
    'generalization': {
        'definition': 'Transfer to new datasets',
        'measure': 'cross_dataset_correlation',
        'target': '> 0.6'
    }
}
```

---

# PART VI: EXPECTED OUTCOMES

## 8. Circuit Library

**Deliverable:** Comprehensive catalog of time series circuits

```
circuits/
├── trend/
│   ├── linear_trend_circuit.json
│   ├── polynomial_trend_circuit.json
│   └── exponential_trend_circuit.json
├── seasonality/
│   ├── daily_seasonality_circuit.json
│   ├── weekly_seasonality_circuit.json
│   └── complex_seasonality_circuit.json
├── anomaly/
│   ├── point_anomaly_circuit.json
│   └── contextual_anomaly_circuit.json
└── forecasting/
    ├── short_term_circuit.json
    └── long_term_circuit.json
```

**Each circuit includes:**
- Component list (layers, heads, neurons)
- Connection structure
- Feature decomposition
- Causal importance scores
- Validation results
- Natural language interpretation
- Visualization

## 9. Universal Circuit Taxonomy

**Hypothesized Universal Circuits:**

**1. Temporal Difference Circuit**
- Function: Compute derivatives and rates of change
- Components: Early-layer attention heads comparing adjacent points
- Expected location: Layers 1-4
- Use cases: Trend detection, anomaly scoring

**2. Window Aggregation Circuit**
- Function: Compute statistics over temporal windows
- Components: Multi-head attention with pooling
- Expected location: Layers 5-10
- Use cases: Seasonality, smoothing, feature extraction

**3. Pattern Matching Circuit**
- Function: Detect repeated patterns
- Components: Induction heads, positional embeddings
- Expected location: Layers 3-7
- Use cases: Seasonality, motif discovery

**4. Baseline Modeling Circuit**
- Function: Learn normal/expected behavior
- Components: MLPs, residual connections
- Expected location: Layers 8-14
- Use cases: Anomaly detection, forecasting

**5. Regime Detection Circuit**
- Function: Identify distribution shifts
- Components: Attention heads, gating mechanisms
- Expected location: Layers 12-18
- Use cases: Change point detection, adaptive forecasting

## 10. Practical Applications

### 10.1 Explainable Forecasting
```python
def explain_forecast(model, input_data, forecast):
    """Explain why model made specific forecast"""
    # Identify active circuits
    active_circuits = identify_active_circuits(model, input_data)

    # Trace information flow
    trace = trace_circuit_activation(model, active_circuits, input_data)

    # Generate natural language explanation
    explanation = generate_explanation(trace, active_circuits)

    return explanation
    # Output: "Forecast predicts upward trend because:
    #  1. Trend circuit detected 2.3% growth rate over last 30 days
    #  2. Seasonality circuit expects +5% seasonal boost
    #  3. Anomaly circuit flagged no recent deviations"
```

### 10.2 Model Editing
```python
def edit_model_behavior(model, target_behavior, modification):
    """Edit model without retraining"""
    # Locate circuit for behavior
    circuit = locate_circuit(model, target_behavior)

    # Apply modification
    if modification['type'] == 'amplify':
        amplify_circuit(model, circuit, modification['strength'])
    elif modification['type'] == 'suppress':
        suppress_circuit(model, circuit)

    return model
    # Example: Reduce model's sensitivity to anomalies
```

---

# PART VII: CHALLENGES AND SOLUTIONS

## 11. Technical Challenges

### 11.1 Computational Complexity
**Problem:** O(N²) activation storage and patching

**Solutions:**
- Sample-based analysis (test on subset of data)
- Gradient-based path scoring (cheaper than patching)
- Parallel patching across layers
- Approximate methods (Integrated Gradients)

### 11.2 Identification Ambiguity
**Problem:** Multiple circuits can implement same behavior (degeneracy)

**Solutions:**
- Find minimal circuits (fewest components)
- Compare across multiple inputs (robust circuits)
- Use consensus methods (multiple discovery methods)
- Report circuit families, not single circuits

### 11.3 Scalability
**Problem:** Large models (1B+ parameters) are hard to analyze

**Solutions:**
- Start with smaller models (scale up gradually)
- Focus on specific behaviors (not full model)
- Use dimensionality reduction
- Build hierarchical analysis (coarse → fine)

## 12. Conceptual Challenges

### 12.1 Circuit Definition
**Problem:** What counts as a "circuit"?

**Solutions:**
- Operational definition: minimal sufficient subgraph
- Report multiple definitions (strict vs. relaxed)
- Provide circuit "boundaries" with confidence intervals
- Allow fuzzy circuit membership

### 12.2 Causality vs. Correlation
**Problem:** Patching shows correlation, not causation

**Solutions:**
- Always validate with interventions
- Use causal scrubbing for necessity tests
- Report both correlational and causal evidence
- Distinguish "upstream" vs. "downstream" components

## 13. Practical Challenges

### 13.1 Resource Requirements
**Problem:** Need significant compute for large-scale analysis

**Solutions:**
- Cloud computing (AWS, GCP credits)
- Collaboration with compute-rich labs
- Incremental analysis (start small, scale up)
- Efficient implementation (optimization, batching)

### 13.2 Reproducibility
**Problem:** Circuit analysis may be sensitive to random seeds

**Solutions:**
- Report all random seeds
- Multiple runs with different seeds
- Aggregate results across runs
- Provide complete code and data

---

# PART VIII: ETHICAL CONSIDERATIONS

## 14. Ethical Framework

### 14.1 Potential Harms

1. **Gaming the Model**
   - Understanding circuits could enable adversarial attacks
   - Manipulate time series to exploit circuit weaknesses

2. **Privacy Violations**
   - Circuits might reveal sensitive patterns
   - Inverse engineering training data

3. **Dual-Use Concerns**
   - Military applications (surveillance, targeting)
   - Financial manipulation

### 14.2 Ethical Guidelines

**Checklist for Ethical Research:**
- [ ] **Beneficence:** Does this research benefit society?
- [ ] **Non-maleficence:** Could this research cause harm?
- [ ] **Autonomy:** Does this respect individual privacy?
- [ ] **Justice:** Are benefits and risks distributed fairly?
- [ ] **Transparency:** Is research process transparent?
- [ ] **Accountability:** Who is responsible for outcomes?

### 14.3 Responsible Research Practices

```python
class EthicalCircuitResearch:
    """Framework for ethical circuit research"""

    def conduct_research(self, circuit_analysis):
        # 1. Harm assessment
        potential_harms = self.harm_assessment.evaluate(circuit_analysis)

        if potential_harms['severity'] > threshold:
            approval = self.review_board.request_approval(circuit_analysis)
            if not approval:
                raise EthicalViolation("Research denied by IRB")

        # 2. Responsible disclosure
        if findings['critical_vulnerability']:
            notify_vendors_first(findings)
            wait_period = 90  # days

        # 3. Dual-use mitigation
        misuse_scenarios = self.red_team(findings)
        for scenario in misuse_scenarios:
            safeguard = design_safeguard(scenario)
            findings.add_safeguard(safeguard)

        return circuit_analysis
```

---

# PART IX: OPEN PROBLEMS

## 15. Foundational Conjectures

### 15.1 Universal Temporal Circuit Library
**Conjecture:** There exists a finite library of universal temporal circuits (≈50-100) that all TSFMs use to process time series.

**Implications:**
- Complete interpretability is achievable
- Transfer learning between models
- Standardized evaluation possible

### 15.2 Circuit Composition Law
**Conjecture:** Complex temporal behaviors are composed from primitive circuits according to a compositional grammar.

**Formalization:**
```
Complex_Behavior = Primitive_1 ⊕ Primitive_2 ⊕ ... ⊕ Primitive_n
```

Where ⊕ is a composition operator (sequential, parallel, hierarchical)

### 15.3 Circuit Depth-Complexity Trade-off
**Conjecture:** Behaviors requiring deeper circuits are inherently more complex and harder to interpret.

**Formalization:**
```
Interpretability ∝ 1 / (Circuit_Depth × Component_Density)
```

## 16. Technical Open Problems

**Problem 1: Polynomial-Time Circuit Discovery**
Can we discover circuits in polynomial time (in model size)?

**Problem 2: Circuit Verification**
Given a hypothesized circuit, verify it implements a behavior in polynomial time.

**Problem 3: Optimal Circuit Design**
Given a behavior, design the minimal circuit that implements it.

**Problem 4: Circuit Transfer**
Transfer a circuit from one model to another without retraining.

---

# PART X: IMPLEMENTATION FRAMEWORK

## 17. Complete Python Framework

```python
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class CircuitComponent:
    """A single component in a circuit"""
    layer: int
    component_type: str  # 'attention', 'mlp'
    index: Optional[int]
    importance: float

@dataclass
class Circuit:
    """A complete circuit for a behavior"""
    name: str
    behavior: str
    components: List[CircuitComponent]
    completeness: float
    validation_results: Dict

class CircuitAnalyzer:
    """Main class for circuit analysis in TSFMs"""

    def __init__(self, model):
        self.model = model
        self.hooks = {}

    def extract_activations(self, inputs: torch.Tensor) -> Dict:
        """Extract activations from all layers"""
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        # Register hooks
        for name, module in self.model.named_modules():
            if 'layer' in name:
                self.hooks[name] = module.register_forward_hook(hook_fn(name))

        with torch.no_grad():
            _ = self.model(inputs)

        # Remove hooks
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}

        return activations

    def path_patching(
        self,
        inputs_baseline: torch.Tensor,
        inputs_target: torch.Tensor,
        behavior_metric: callable
    ) -> List[CircuitComponent]:
        """Identify critical components via path patching"""

        acts_baseline = self.extract_activations(inputs_baseline)
        acts_target = self.extract_activations(inputs_target)

        with torch.no_grad():
            output_target = self.model(inputs_target)
        target_behavior = behavior_metric(output_target)

        critical_components = []

        for layer_idx in range(len(self.model.layers)):
            layer = self.model.layers[layer_idx]

            # Patch attention heads
            for head_idx in range(layer.attention.num_heads):
                patched = self._patch_attention_head(
                    inputs_target,
                    layer_idx,
                    head_idx,
                    acts_baseline[layer_idx]['attention'][..., head_idx, :]
                )

                behavior_change = abs(behavior_metric(patched) - target_behavior)

                if behavior_change > 0.1:
                    critical_components.append(CircuitComponent(
                        layer=layer_idx,
                        component_type='attention',
                        index=head_idx,
                        importance=behavior_change
                    ))

        return critical_components

    def discover_circuit(
        self,
        dataset: torch.Tensor,
        behavior: str,
        behavior_metric: callable
    ) -> Circuit:
        """Discover complete circuit for a behavior"""

        print(f"Discovering circuit for {behavior}...")

        # Step 1: Path patching
        critical = self.path_patching(
            dataset[0],
            dataset[1],
            behavior_metric
        )

        # Step 2: Group into circuit
        circuit_components = sorted(critical, key=lambda x: x.importance, reverse=True)[:20]

        # Step 3: Validate
        completeness = self._validate_circuit(circuit_components, dataset, behavior_metric)

        circuit = Circuit(
            name=f"{behavior}_circuit",
            behavior=behavior,
            components=circuit_components,
            completeness=completeness,
            validation_results={}
        )

        print(f"Circuit discovered with {len(circuit_components)} components")
        print(f"Completeness: {completeness:.3f}")

        return circuit

    def _validate_circuit(
        self,
        circuit: List[CircuitComponent],
        dataset: torch.Tensor,
        behavior_metric: callable
    ) -> float:
        """Validate circuit completeness"""

        # Measure full model performance
        with torch.no_grad():
            full_outputs = [self.model(x) for x in dataset]
        full_performance = np.mean([behavior_metric(out) for out in full_outputs])

        # Scrub non-circuit components (simplified)
        circuit_set = set((c.layer, c.component_type, c.index) for c in circuit)

        # For simplicity, measure how much performance is explained
        # In practice, would do proper causal scrubbing
        explained_performance = full_performance * 0.85  # Placeholder

        completeness = explained_performance / full_performance

        return completeness

# Usage example
def example_usage():
    """Example of how to use the circuit analyzer"""

    # Load model
    model = load_timesfm("google/timesfm-1.0")

    # Create analyzer
    analyzer = CircuitAnalyzer(model)

    # Define behavior metric
    def trend_detection_metric(output):
        return output.mean()  # Placeholder

    # Load dataset
    dataset = torch.randn(100, 512, 1)

    # Discover circuit
    circuit = analyzer.discover_circuit(
        dataset=dataset,
        behavior='trend_detection',
        behavior_metric=trend_detection_metric
    )

    print(f"Circuit: {circuit.name}")
    print(f"Components: {len(circuit.components)}")
    print(f"Completeness: {circuit.completeness:.3f}")
```

---

# PART XI: TIMELINE AND NEXT STEPS

## 18. Research Timeline

### Phase 1: Foundation (Months 1-3)
- [ ] Literature review on circuit analysis
- [ ] Implement activation patching for TSFMs
- [ ] Implement causal scrubbing framework
- [ ] Create synthetic datasets with ground truth
- [ ] Validate methods on synthetic data

**Deliverables:**
- Working implementation
- Validation results on synthetic data
- Code repository with documentation

### Phase 2: Discovery (Months 4-6)
- [ ] Discover trend detection circuits
- [ ] Discover seasonality circuits
- [ ] Discover anomaly detection circuits
- [ ] Validate causally
- [ ] Cross-dataset generalization tests

**Deliverables:**
- Circuit library (trend, seasonality, anomaly)
- Analysis paper on discovered circuits
- Validation results

### Phase 3: Cross-Model Analysis (Months 7-9)
- [ ] Analyze circuits in TimesFM
- [ ] Analyze circuits in Chronos
- [ ] Analyze circuits in MOMENT
- [ ] Identify universal motifs
- [ ] Analyze differences

**Deliverables:**
- Cross-model circuit comparison
- Universal circuit taxonomy
- Paper on universal circuits

### Phase 4: Applications (Months 10-12)
- [ ] Build explainable forecasting system
- [ ] Create model editing toolkit
- [ ] Develop failure mode analyzer
- [ ] Release open-source tools

**Deliverables:**
- Open-source toolkit
- Demo applications
- Documentation

## 19. Immediate Next Steps

### This Week:
1. **Literature Deep Dive:**
   - Read Anthropic's Transformer Circuits papers
   - Study TimeSAE methodology in detail

2. **Setup Environment:**
   - Install required libraries
   - Set up GPU environment
   - Download pre-trained TSFMs
   - Prepare initial datasets

3. **Initial Experiments:**
   - Implement basic activation extraction
   - Run path patching on simple example
   - Validate on synthetic data

### Next Month:
1. **Method Implementation:**
   - Complete path patching framework
   - Implement causal scrubbing
   - Build circuit induction algorithm

2. **Validation Studies:**
   - Create synthetic datasets
   - Validate methods on ground truth
   - Measure accuracy and reliability

3. **Initial Discovery:**
   - Discover first circuit (trend detection)
   - Validate causally
   - Document findings

---

# PART XII: RESOURCES

## 20. Required Knowledge

### Machine Learning:
- Deep learning fundamentals (CNNs, RNNs, Transformers)
- Attention mechanisms
- Training dynamics and optimization
- Interpretability methods (SAEs, probing, patching)

### Time Series:
- Time series fundamentals (trend, seasonality, stationarity)
- Forecasting methods (ARIMA, exponential smoothing)
- Time series decomposition (STL, seasonal decomposition)
- Anomaly detection in time series

### Mathematics:
- Linear algebra (matrix operations, eigendecomposition)
- Probability and statistics
- Causal inference basics
- Information theory

## 21. Technical Stack

### Core Libraries:
```python
requirements = {
    'deep_learning': ['torch>=2.0.0', 'transformers>=4.30.0'],
    'time_series': ['tsfm', 'darts>=0.23.0', 'sktime>=0.19.0'],
    'interpretability': ['nnsight>=0.0.1', 'saes', 'circuitsvis'],
    'analysis': ['numpy>=1.24.0', 'scipy>=1.10.0', 'scikit-learn>=1.2.0']
}
```

### Computing Requirements:
- **GPU:** At least one A100 (40GB) or V100 (32GB)
- **RAM:** 128GB+ recommended for large models
- **Storage:** 1TB+ for models and datasets

## 22. Data Sources

**Time Series Datasets:**
- Monash Time Series Forecasting Repository
- UCR Time Series Classification Archive
- PhysioNet (medical time series)
- M4/M5 Competition datasets
- Kaggle datasets (domain-specific)

**Pre-trained Models:**
- TimesFM (Google)
- Chronos (Amazon)
- MOMENT (Carnegie Mellon)
- UniTS (Alibaba DAMO)

---

# PART XIII: SUCCESS CRITERIA

## 23. Research Success

### Minimum Viable Success:
- [ ] Identify at least 3 distinct circuits (trend, seasonality, anomaly)
- [ ] Validate circuits causally (causal scrubbing)
- [ ] Achieve >70% completeness on synthetic data
- [ ] Reproduce findings across multiple TSFMs

### Strong Success:
- [ ] Comprehensive circuit library (10+ circuits)
- [ ] Cross-model circuit universality demonstrated
- [ ] Explainable forecasting system built
- [ ] Published in top-tier venue (NeurIPS, ICML, ICLR)

### Exceptional Success:
- [ ] Universal circuit taxonomy established
- [ ] Model editing toolkit deployed
- [ ] Industry adoption (real-world use cases)
- [ ] Field-transforming impact

## 24. Impact Metrics

### Academic Impact:
- Citation count (>50 in first 2 years)
- Workshop/invited talks (>3)
- Follow-up research by community
- PhD theses built on work

### Practical Impact:
- Open-source toolkit adoption (>1000 GitHub stars)
- Industry partnerships (>2 companies)
- Real-world deployment stories
- Regulatory acceptance (FDA, etc.)

---

# CONCLUSION

## Key Takeaways

1. **Circuits exist and are discoverable** - Evidence from LLMs suggests TSFMs will have similar structure
2. **Multi-layer analysis is essential** - Single-layer analysis misses compositional structure
3. **Causal validation is critical** - Always validate with interventions, not just observations
4. **Start small and scale up** - Begin with synthetic data and small models
5. **Ethics must be central** - Consider potential harms and dual-use concerns
6. **Community matters** - Open science and collaboration accelerate progress

## The Road Ahead

Multi-layer circuit analysis for TSFMs is in its infancy. The methods, theories, and applications outlined here represent the beginning. As the field matures, we can expect:

- More sophisticated theoretical frameworks
- Better tools and infrastructure
- Deeper understanding of temporal computation
- Practical applications in high-stakes domains
- Integration with AI safety and alignment research

## Call to Action

If you're excited about this research direction:
1. Start with small, focused experiments
2. Collaborate with others
3. Share your findings openly
4. Engage with the community
5. Think critically and ethically

**The journey from black-box TSFMs to interpretable, trustworthy systems is long, but the destination is worth it.**

---

*"The purpose of computing is insight, not numbers."* - Richard Hamming

*"In interpretability, the journey is the destination."*

---

**Document Status:** Integrated comprehensive guide
**Last Updated:** March 11, 2026
**Next Review:** Quarterly
**Contact:** [Research Community Forum]

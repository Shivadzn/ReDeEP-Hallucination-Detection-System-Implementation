# System Architecture

> Complete technical overview of the ReDeEP hallucination detection system architecture

## 📋 Table of Contents
- [High-Level Overview](#high-level-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Module Descriptions](#module-descriptions)
- [Memory Management](#memory-management)
- [Design Decisions](#design-decisions)
- [Scalability Considerations](#scalability-considerations)

---

## High-Level Overview

The ReDeEP hallucination detection system consists of three main stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: DETECTION                          │
│                                                                       │
│  Input Data → Model Loading → Attention Extraction → Score Output   │
│  (RAGTruth)   (LLaMA-2-7B)    (32 heads)           (JSON)           │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       STAGE 2: REGRESSION                            │
│                                                                       │
│  Detection → Feature Selection → Score Calculation → Evaluation     │
│  Scores      (AUC ranking)       (Weighted sum)      (Metrics)      │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     STAGE 3: VISUALIZATION                           │
│                                                                       │
│  Results → Plot Generation → Dashboard Creation → Report Export     │
│  (JSON)    (matplotlib)       (6-panel figure)     (Markdown)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Detailed System Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                                     │
├───────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐            │
│  │   Query      │  │  Retrieved       │  │  Model Response    │            │
│  │   Text       │  │  Context         │  │  Text              │            │
│  └──────┬───────┘  └────────┬─────────┘  └─────────┬──────────┘            │
│         │                   │                       │                     │
│         └───────────────────┴───────────────────────┘                     │
└───────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ↓
┌───────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING LAYER                                   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                    MODEL LAYER                                   │     │
│  │  ┌────────────────────────────────────────────────────────┐     │     │
│  │  │         LLaMA-2-7B (4-bit Quantized)                   │     │     │
│  │  │                                                          │     │     │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │     │     │
│  │  │  │ Layer 0  │→ │ Layer 8  │→ │ Layer 16 │ ...        │     │     │
│  │  │  │(32 heads)│  │(32 heads)│  │(32 heads)│            │     │     │
│  │  │  └──────────┘  └──────────┘  └──────────┘            │     │     │
│  │  │                                     ↓                   │     │     │
│  │  │                            Layer 31 (32 heads)          │     │     │
│  │  └────────────────────────────────────────────────────────┘     │     │
│  │                              ↓                                    │     │
│  │  ┌────────────────────────────────────────────────────────┐     │     │
│  │  │            ATTENTION EXTRACTION MODULE                  │     │     │
│  │  │                                                          │     │     │
│  │  │  Selected Heads: [0,0], [0,16], [2,0], [2,16], ...    │     │     │
│  │  │                  [28,0], [28,16], [30,0], [30,16]      │     │     │
│  │  │                                                          │     │     │
│  │  │  Output: 32 attention patterns per chunk                │     │     │
│  │  └────────────────────────────────────────────────────────┘     │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ↓
┌───────────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS LAYER                                     │
├───────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │              FEATURE ENGINEERING MODULE                       │         │
│  │                                                                │         │
│  │  External Similarity (32 features)                            │         │
│  │  ┌────────────────────────────────────────────────────┐      │         │
│  │  │ prompt_attention_score: Dict[head_id, score]       │      │         │
│  │  │ Measures: attention to retrieved context           │      │         │
│  │  └────────────────────────────────────────────────────┘      │         │
│  │                                                                │         │
│  │  Parameter Knowledge (32 features)                            │         │
│  │  ┌────────────────────────────────────────────────────┐      │         │
│  │  │ parameter_knowledge_scores: Dict[head_id, score]   │      │         │
│  │  │ Measures: internal knowledge representation         │      │         │
│  │  └────────────────────────────────────────────────────┘      │         │
│  └──────────────────────────────────────────────────────────────┘         │
│                              ↓                                              │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │              REGRESSION MODULE                                │         │
│  │                                                                │         │
│  │  Step 1: Feature Selection (AUC ranking)                      │         │
│  │  ┌────────────────────────────────────────┐                  │         │
│  │  │ Top 3 External Similarity features     │                  │         │
│  │  │ Top 4 Parameter Knowledge features     │                  │         │
│  │  └────────────────────────────────────────┘                  │         │
│  │                                                                │         │
│  │  Step 2: Score Calculation                                    │         │
│  │  ┌────────────────────────────────────────┐                  │         │
│  │  │ Score = m*PK_sum - α*ES_sum            │                  │         │
│  │  │ Where: m=1, α=0.6                      │                  │         │
│  │  └────────────────────────────────────────┘                  │         │
│  │                                                                │         │
│  │  Step 3: Normalization (Min-Max scaling)                      │         │
│  │  Step 4: Response-level Aggregation                           │         │
│  └──────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ↓
┌───────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                       │
├───────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │  Hallucination   │  │  Visualization   │  │  Analysis        │       │
│  │  Scores (JSON)   │  │  Dashboard (PNG) │  │  Report (MD)     │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Detection Phase

```python
# Pseudocode
def detection_pipeline(sample):
    """
    Input: {query, context, response, chunks}
    Output: {chunk_scores, attention_patterns}
    """
    
    # Step 1: Prepare input
    prompt = construct_prompt(query, context, response)
    tokens = tokenizer(prompt, truncation=True, max_length=8000)
    
    # Step 2: Forward pass with attention tracking
    with torch.no_grad():
        outputs = model(
            tokens.input_ids,
            output_attentions=True
        )
    
    # Step 3: Extract attention from selected heads
    attention_patterns = {}
    for layer, head in TOPK_HEADS:
        attention = outputs.attentions[layer][0, head]
        attention_patterns[(layer, head)] = attention
    
    # Step 4: Calculate scores per chunk
    chunk_scores = []
    for chunk in chunks:
        # External similarity: attention to context
        es_score = calculate_external_similarity(
            attention_patterns, chunk, context
        )
        
        # Parameter knowledge: internal representation
        pk_score = calculate_parameter_knowledge(
            attention_patterns, chunk
        )
        
        chunk_scores.append({
            'external_similarity': es_score,
            'parameter_knowledge': pk_score,
            'hallucination_label': chunk.label
        })
    
    return chunk_scores
```

### 2. Regression Phase

```python
# Pseudocode
def regression_pipeline(detection_results):
    """
    Input: detection_results (list of chunk scores)
    Output: {auc, pearson_correlation, predictions}
    """
    
    # Step 1: Construct dataframe
    df = construct_dataframe(detection_results)
    # Columns: ES_0, ES_1, ..., ES_31, PK_0, PK_1, ..., PK_31, label
    
    # Step 2: Feature selection
    auc_scores = []
    for feature in features:
        auc = calculate_auc(df[feature], df['label'])
        auc_scores.append((auc, feature))
    
    # Select top features
    top_es_features = sorted(es_features, key=lambda x: x[0])[:3]
    top_pk_features = sorted(pk_features, key=lambda x: x[0])[:4]
    
    # Step 3: Combine features
    df['es_combined'] = df[top_es_features].sum(axis=1)
    df['pk_combined'] = df[top_pk_features].sum(axis=1)
    
    # Step 4: Calculate final score
    df['hallucination_score'] = (
        m * normalize(df['pk_combined']) - 
        alpha * normalize(df['es_combined'])
    )
    
    # Step 5: Response-level aggregation
    response_scores = df.groupby('response_id').agg({
        'hallucination_score': 'mean',
        'label': 'max'
    })
    
    # Step 6: Evaluation
    final_auc = roc_auc_score(
        response_scores['label'],
        response_scores['hallucination_score']
    )
    
    return final_auc, response_scores
```

---

## Module Descriptions

### Core Modules

#### 1. **Path Manager** (`src/path_manager.py`)
**Purpose**: Universal path resolution for different environments (Kaggle, Colab, local)

**Key Functions**:
```python
def resolve_paths(environment='auto'):
    """Detect environment and configure paths accordingly"""
    
def universal_path_patch(script_content, script_type):
    """Patch script paths for target environment"""
```

**Design**: Abstracts away environment-specific path handling

---

#### 2. **Model Loader** (`src/model_loader.py`)
**Purpose**: Load and configure LLaMA models with optimizations

**Key Functions**:
```python
def load_model_quantized(
    model_name: str,
    quantization: str = '4bit',
    device_map: str = 'balanced',
    max_memory: Dict = None
):
    """Load model with quantization and memory optimization"""
```

**Key Features**:
- 4-bit/8-bit quantization support
- Automatic device mapping
- Memory limit enforcement
- Cache management

---

#### 3. **Detection Engine** (`src/detection.py`)
**Purpose**: Extract attention patterns and calculate scores

**Key Functions**:
```python
class HallucinationDetector:
    def __init__(self, model, tokenizer, topk_heads):
        """Initialize detector with model and attention heads"""
    
    def detect(self, response, context, source_info):
        """Process a response and return hallucination scores"""
    
    def extract_attention(self, outputs, layer, head):
        """Extract attention weights from specific head"""
    
    def calculate_scores(self, attention, chunks, context):
        """Calculate ES and PK scores for chunks"""
```

---

#### 4. **Regression Analyzer** (`src/regression.py`)
**Purpose**: Train regression model and predict hallucinations

**Key Functions**:
```python
class RegressionAnalyzer:
    def __init__(self, detection_results, source_info):
        """Initialize with detection outputs"""
    
    def construct_dataframe(self):
        """Build feature dataframe from attention patterns"""
    
    def select_features(self, n_external=3, n_parameter=4):
        """Select most predictive features via AUC"""
    
    def calculate_scores(self, alpha=0.6, m=1.0):
        """Compute final hallucination scores"""
    
    def evaluate(self):
        """Calculate AUC, accuracy, precision, recall"""
```

---

#### 5. **Visualization Generator** (`src/visualization.py`)
**Purpose**: Generate plots and analysis dashboard

**Key Functions**:
```python
def generate_roc_curve(labels, scores):
    """Generate ROC curve with AUC"""

def generate_score_distribution(labels, scores):
    """Create histogram of score distributions"""

def generate_dashboard(results, metrics):
    """Create complete 6-panel analysis dashboard"""

def generate_report(results, metrics):
    """Generate markdown analysis report"""
```

---

## Memory Management

### Optimization Strategy

```
┌────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Model Storage (GPU VRAM)                                  │
│  ┌──────────────────────────────────────────────┐          │
│  │  LLaMA-2-7B (4-bit): ~7GB                    │          │
│  │  ├─ Weights: ~6.5GB                          │          │
│  │  ├─ KV Cache: ~500MB                         │          │
│  │  └─ Attention Buffers: Variable              │          │
│  └──────────────────────────────────────────────┘          │
│                      ↓                                     │
│  Processing Memory (GPU VRAM)                              │
│  ┌──────────────────────────────────────────────┐          │
│  │  Activations: ~2-3GB per forward pass        │          │
│  │  Attention Outputs: ~500MB                   │          │
│  │  Gradients: 0GB (inference only)             │          │
│  └──────────────────────────────────────────────┘          │
│                      ↓                                     │        
│  Total per GPU: ~10-12GB (peak)                            │
│                                                            │        
└────────────────────────────────────────────────────────────┘
```

### Memory Management Techniques

1. **4-bit Quantization**
   - Reduces model size by 75%
   - Minimal accuracy loss (<1%)
   - Uses NF4 (Normal Float 4-bit)

2. **Balanced Device Mapping**
   - Distributes layers across GPUs
   - Automatic load balancing
   - Prevents single-GPU bottleneck

3. **Sequence Truncation**
   - Dynamic truncation based on memory
   - Preserves most important context
   - Fallback limits: 8000 → 6000 → 4000 tokens

4. **Periodic Cache Clearing**
   ```python
   if processed_count % 5 == 0:
       torch.cuda.empty_cache()
       gc.collect()
   ```

5. **Gradient Disabled**
   ```python
   model.eval()
   for param in model.parameters():
       param.requires_grad = False
   ```

---

## Design Decisions

### 1. Why 4-bit Quantization?

**Trade-offs Considered**:
| Quantization | Model Size | Accuracy Loss | Inference Speed |
|--------------|-----------|---------------|-----------------|
| FP16 | 14GB | 0% | 1.0x |
| 8-bit | 7GB | <0.5% | 1.2x |
| **4-bit (NF4)** | **3.5GB** | **<1%** | **1.5x** |

**Decision**: 4-bit offers best size/accuracy trade-off for T4 GPUs

---

### 2. Why 32 Attention Heads?

**Rationale**:
- Paper tested heads from all 32 layers
- Selected layers 0-30 (even numbers only)
- 2 heads per layer (0, 16) for diversity
- Total: 16 layers × 2 heads = 32 features

**Alternative Considered**: All 1024 heads (32 layers × 32 heads)
**Rejected Because**: Computationally expensive, diminishing returns

---

### 3. Why Balanced Device Map?

**Options Compared**:
| Strategy | Description | Pros | Cons |
|----------|------------|------|------|
| `auto` | PyTorch decides | Simple | May overload single GPU |
| `sequential` | Layers in order | Predictable | Unbalanced load |
| **`balanced`** | **Equal distribution** | **Even usage** | **Requires manual config** |

**Decision**: Balanced provides best GPU utilization on multi-GPU systems

---

## Scalability Considerations

### Current Limitations

1. **Single-Sample Processing**
   - Batch size = 1
   - Reason: Memory constraints with long sequences
   - Impact: Slower throughput

2. **Fixed Attention Heads**
   - 32 heads hardcoded
   - Reason: Based on paper's findings
   - Impact: May not be optimal for all datasets

3. **CPU-GPU Transfer**
   - Attention patterns moved to CPU for storage
   - Reason: GPU memory limits
   - Impact: Transfer overhead

### Future Improvements

1. **Batch Processing**
   ```python
   # Current
   for sample in dataset:
       process(sample)
   
   # Proposed
   for batch in batches(dataset, size=4):
       process_batch(batch)  # 4x speedup
   ```

2. **Dynamic Head Selection**
   ```python
   # Learn which heads are most important
   head_importance = analyze_head_contributions()
   topk_heads = select_top_k(head_importance, k=16)
   ```

3. **Streaming Processing**
   ```python
   # Process and save incrementally
   for sample in stream(dataset):
       result = process(sample)
       save_incrementally(result)
   ```

---

## Performance Metrics

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Model Forward | O(n²) per head | O(n) per head |
| Attention Extraction | O(k×n) | O(k×n) |
| Feature Calculation | O(m×k) | O(m×k) |
| Regression | O(m×f) | O(m×f) |

Where:
- n = sequence length
- k = number of attention heads (32)
- m = number of samples
- f = number of features (64)

### Throughput Analysis

**Detection Phase**:
- Input: 1 sample/forward pass
- Processing: ~4-5 seconds/sample
- Throughput: ~12-15 samples/minute
- Bottleneck: Model inference

**Regression Phase**:
- Input: All detection results
- Processing: ~0.5 seconds/sample
- Throughput: ~120 samples/minute
- Bottleneck: Feature engineering

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                    FRAMEWORK LAYER                       │
│  PyTorch 2.0+ | Transformers 4.35+ | Accelerate 0.25+  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  OPTIMIZATION LAYER                      │
│  bitsandbytes (quantization) | Flash Attention (future) │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   ANALYSIS LAYER                         │
│  scikit-learn | pandas | numpy | scipy                  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                 VISUALIZATION LAYER                      │
│  matplotlib | seaborn | Pillow                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔗 Related Documentation

- [Implementation Notes](IMPLEMENTATION_NOTES.md) - Technical decisions explained
- [Optimization Guide](OPTIMIZATION_GUIDE.md) - Performance tuning tips
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions

---

**Last Updated**: November 2025  
**Version**: 1.0.0  
**Maintained by**: [SHIVA GUPTA]
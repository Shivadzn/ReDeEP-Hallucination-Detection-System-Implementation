# Optimization Guide

> Techniques for improving speed, memory usage, and accuracy

## 📋 Table of Contents
- [Performance Metrics](#performance-metrics)
- [Memory Optimization](#memory-optimization)
- [Speed Optimization](#speed-optimization)
- [Accuracy Optimization](#accuracy-optimization)
- [Hardware-Specific Tips](#hardware-specific-tips)
- [Production Optimization](#production-optimization)

---

## Performance Metrics

### Current Baseline (Kaggle T4x2)

| Metric | Value | Target |
|--------|-------|--------|
| Detection Speed | 11 samples/min | 20+ samples/min |
| Regression Speed | 46 samples/min | 100+ samples/min |
| GPU Memory Usage | 10-12GB/GPU | <10GB/GPU |
| AUC | 0.689 | 0.75+ |
| End-to-End Time | 65 minutes | 30 minutes |

---

## Memory Optimization

### Technique 1: More Aggressive Quantization

**Current**: 4-bit NF4
**Option**: 3-bit or 2-bit quantization (experimental)

```python
# Requires GPTQ or AWQ
from transformers import GPTQConfig

gptq_config = GPTQConfig(
    bits=3,  # Even more aggressive
    dataset="c4",
    tokenizer=tokenizer
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=gptq_config,
    device_map="balanced"
)
```

**Trade-off**:
- Memory: 3.5GB → 2.5GB (29% savings)
- Accuracy: ~2-3% loss
- Speed: 10-20% faster

**When to use**: If constantly hitting OOM

---

### Technique 2: Gradient Checkpointing

Trades compute for memory by recomputing activations during backward pass

```python
model.gradient_checkpointing_enable()

# For inference only, combine with:
model.config.use_cache = False
```

**Trade-off**:
- Memory: -30% usage
- Speed: -20% slower
- Accuracy: No change

**When to use**: Long sequences (>8K tokens)

---

### Technique 3: Flash Attention

Optimized attention implementation

```python
pip install flash-attn

# Then in model loading:
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",  # Instead of "eager"
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)
```

**Trade-off**:
- Memory: -20% on attention layers
- Speed: +30% faster
- Accuracy: No change

**Requirements**: 
- CUDA 11.6+
- Ampere GPUs or newer (T4 works!)

---

### Technique 4: Dynamic Sequence Packing

Group similar-length sequences to minimize padding

```python
def pack_sequences(samples, max_length=8000):
    """Group samples by length for efficient batching"""
    samples_by_length = {}
    for sample in samples:
        length = len(tokenizer(sample.text).input_ids)
        bucket = (length // 1000) * 1000  # Round to nearest 1000
        if bucket not in samples_by_length:
            samples_by_length[bucket] = []
        samples_by_length[bucket].append(sample)
    
    return samples_by_length

# Process by length buckets
for length_bucket, bucket_samples in pack_sequences(samples).items():
    print(f"Processing {len(bucket_samples)} samples of length ~{length_bucket}")
    batch_size = get_optimal_batch_size(length_bucket)
    
    for i in range(0, len(bucket_samples), batch_size):
        batch = bucket_samples[i:i+batch_size]
        process_batch(batch)
```

**Benefit**: 
- Can use batch_size > 1 for short sequences
- Reduces wasted computation on padding

---

### Technique 5: Offload to CPU

Move some layers to CPU to free GPU memory

```python
# Offload last 4 layers to CPU
device_map = {
    f"model.layers.{i}": 0 for i in range(0, 14)  # GPU 0
} | {
    f"model.layers.{i}": 1 for i in range(14, 28)  # GPU 1
} | {
    f"model.layers.{i}": "cpu" for i in range(28, 32)  # CPU
}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map,
    quantization_config=bnb_config
)
```

**Trade-off**:
- Memory: Frees 1-2GB GPU
- Speed: -15% slower (CPU-GPU transfer)
- When to use: Last resort for OOM

---

## Speed Optimization

### Technique 6: Batch Processing

**Current**: batch_size = 1  
**Optimized**: Dynamic batching based on sequence length

```python
def get_optimal_batch_size(seq_length):
    """Calculate safe batch size for given sequence length"""
    if seq_length < 1000:
        return 16
    elif seq_length < 2000:
        return 8
    elif seq_length < 4000:
        return 4
    elif seq_length < 6000:
        return 2
    else:
        return 1

# Group and batch
for length_group in group_by_length(samples):
    batch_size = get_optimal_batch_size(length_group.avg_length)
    
    for i in range(0, len(length_group), batch_size):
        batch = length_group[i:i+batch_size]
        
        # Batch tokenization
        inputs = tokenizer(
            [s.text for s in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8000
        )
        
        # Batch inference
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Process batch
        for j, sample in enumerate(batch):
            process_sample(outputs, j, sample)
```

**Speedup**: 2-4x on short sequences, 1x on long

---

### Technique 7: Compiled Model (PyTorch 2.0+)

```python
# After loading model
model = torch.compile(model, mode="reduce-overhead")

# First run will be slow (compilation)
# Subsequent runs will be faster
```

**Speedup**: 10-30% after compilation

**Caveats**:
- First run takes 5-10 minutes
- Best for repeated inference
- May not work with all quantization

---

### Technique 8: Attention Head Caching

Cache attention extraction for duplicate inputs

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def extract_attention_cached(text_hash, layer, head):
    """Cache attention for seen texts"""
    # text_hash = hash of tokenized input
    outputs = model(tokens, output_attentions=True)
    return outputs.attentions[layer][0, head]

# Usage
text_hash = hash(tuple(tokens.input_ids[0].tolist()))
attention = extract_attention_cached(text_hash, layer, head)
```

**Speedup**: 100x for duplicate texts (e.g., in ablation studies)

---

### Technique 9: Parallel Processing

Process multiple samples in parallel on different GPUs

```python
import torch.multiprocessing as mp

def worker(gpu_id, samples_chunk):
    """Worker process for one GPU"""
    torch.cuda.set_device(gpu_id)
    
    # Load model on this GPU
    model = load_model_on_gpu(gpu_id)
    
    results = []
    for sample in samples_chunk:
        result = process(model, sample)
        results.append(result)
    
    return results

# Split work across GPUs
num_gpus = torch.cuda.device_count()
chunks = split_samples(samples, num_gpus)

# Launch parallel workers
with mp.Pool(num_gpus) as pool:
    all_results = pool.starmap(worker, enumerate(chunks))

# Combine results
results = [r for chunk_results in all_results for r in chunk_results]
```

**Speedup**: Near-linear with GPU count (2x with 2 GPUs)

---

### Technique 10: JIT Compilation for Score Calculation

```python
import numba

@numba.jit(nopython=True)
def calculate_scores_fast(attention, weights):
    """JIT-compiled score calculation"""
    scores = np.zeros(len(attention))
    for i in range(len(attention)):
        scores[i] = np.dot(attention[i], weights)
    return scores

# Use compiled version
scores = calculate_scores_fast(attention_np, weights_np)
# vs
scores = calculate_scores_slow(attention, weights)  # 10x slower
```

**Speedup**: 5-10x on numerical operations

---

## Accuracy Optimization

### Technique 11: Hyperparameter Tuning

Current parameters are defaults - optimize them!

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'top_n_external': [1, 2, 3, 4, 5, 6, 7],
    'top_k_parameter': [1, 2, 3, 4, 5, 6, 7],
    'alpha': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    'm': [0.5, 0.75, 1.0, 1.25, 1.5]
}

# Grid search
best_auc = 0
best_params = None

for top_n in param_grid['top_n_external']:
    for top_k in param_grid['top_k_parameter']:
        for alpha in param_grid['alpha']:
            for m in param_grid['m']:
                auc = evaluate_params(top_n, top_k, alpha, m)
                
                if auc > best_auc:
                    best_auc = auc
                    best_params = {
                        'top_n': top_n,
                        'top_k': top_k,
                        'alpha': alpha,
                        'm': m
                    }

print(f"Best AUC: {best_auc:.4f}")
print(f"Best params: {best_params}")
```

**Expected improvement**: +5-10% AUC

---

### Technique 12: Ensemble Methods

Combine multiple models or attention head selections

```python
# Train multiple models with different head selections
models = []
for seed in range(5):
    np.random.seed(seed)
    random_heads = select_random_heads(32)
    model = train_model(random_heads)
    models.append(model)

# Ensemble prediction
def ensemble_predict(sample):
    predictions = [model.predict(sample) for model in models]
    return np.mean(predictions)  # Average

# Or voting
def ensemble_vote(sample):
    predictions = [model.predict(sample) > 0.5 for model in models]
    return sum(predictions) > len(predictions) / 2
```

**Expected improvement**: +2-5% AUC

---

### Technique 13: Attention Head Selection via Mutual Information

Select heads with highest mutual information with labels

```python
from sklearn.feature_selection import mutual_info_classif

# Calculate MI for each head
mi_scores = []
for i in range(32):
    mi = mutual_info_classif(
        df[[f'ES_{i}', f'PK_{i}']].values,
        df['hallucination_label'].values
    )
    mi_scores.append((mi[0] + mi[1], i))

# Select top-k heads by MI
mi_scores.sort(reverse=True)
best_heads = [idx for _, idx in mi_scores[:16]]  # Top 16 heads

print(f"Best heads: {best_heads}")
# Use these heads instead of default 0, 16 for each layer
```

**Expected improvement**: +3-7% AUC

---

### Technique 14: Threshold Optimization

Find optimal classification threshold instead of 0.5

```python
from sklearn.metrics import precision_recall_curve

# Calculate precision-recall for all thresholds
precision, recall, thresholds = precision_recall_curve(y_true, scores)

# Find threshold maximizing F1
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal threshold: {best_threshold:.3f}")
print(f"vs default 0.5")

# Use optimal threshold
predictions = (scores >= best_threshold).astype(int)
```

**Expected improvement**: +5-10% F1 score

---

## Hardware-Specific Tips

### For Kaggle (2x T4)

**Optimal Settings**:
```python
# Model config
quantization = "4bit"
device_map = "balanced"
max_memory = {0: "12GB", 1: "12GB"}

# Processing config  
max_sequence_length = 6000
batch_size = 1
cleanup_frequency = 5  # Every 5 samples

# Expected performance
detection_speed = "11 samples/min"
memory_usage = "10-12GB per GPU"
```

---

### For Colab (1x T4)

**Optimal Settings**:
```python
# Model config
quantization = "4bit"
device_map = "auto"  # Single GPU
max_memory = {0: "14GB"}

# Processing config
max_sequence_length = 4000  # More aggressive
batch_size = 1
cleanup_frequency = 3  # More frequent

# Expected performance
detection_speed = "8-10 samples/min"
memory_usage = "13-14GB"
```

---

### For Colab Pro (A100)

**Optimal Settings**:
```python
# Model config
quantization = "8bit"  # Can afford higher precision
device_map = "auto"
max_memory = {0: "35GB"}

# Processing config
max_sequence_length = 12000  # Can handle longer
batch_size = 4  # Can batch!
cleanup_frequency = 10

# Expected performance
detection_speed = "40-50 samples/min"
memory_usage = "25-30GB"
```

---

### For Local (RTX 4090)

**Optimal Settings**:
```python
# Model config
quantization = None  # Can run FP16
torch_dtype = torch.float16
device_map = "auto"

# Processing config
max_sequence_length = 16000  # Full context
batch_size = 8
use_flash_attention = True

# Expected performance
detection_speed = "60-80 samples/min"
memory_usage = "18-20GB"
```

---

## Production Optimization

### Technique 15: Model Serving with vLLM

```python
# Install vLLM
pip install vllm

# Serve model
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="awq",  # Fast quantization
    tensor_parallel_size=2,  # Use 2 GPUs
    max_model_len=8192
)

# Batch inference (much faster)
outputs = llm.generate(
    prompts=batch_of_prompts,
    sampling_params=SamplingParams(temperature=0)
)
```

**Speedup**: 5-10x vs vanilla transformers

---

### Technique 16: TensorRT Optimization

```python
# Convert to TensorRT
from torch_tensorrt import compile

# Compile model
trt_model = compile(
    model,
    inputs=[torch.randn(1, 512).cuda()],
    enabled_precisions={torch.float16},
    workspace_size=1 << 30  # 1GB
)

# Use TRT model
outputs = trt_model(inputs)
```

**Speedup**: 2-3x on NVIDIA GPUs

---

### Technique 17: ONNX Export

```python
# Export to ONNX
torch.onnx.export(
    model,
    sample_input,
    "model.onnx",
    input_names=["input_ids"],
    output_names=["logits", "attentions"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"}
    }
)

# Load and run with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input_ids": tokens})
```

**Benefits**: 
- Faster inference
- Deploy anywhere (C++, mobile)
- Smaller model size

---

## Optimization Decision Tree

```
Start: Need to optimize?
│
├─ Problem: Out of Memory
│  ├─ Try: 4-bit quantization → Still OOM?
│  ├─ Try: Reduce sequence length → Still OOM?
│  ├─ Try: Gradient checkpointing → Still OOM?
│  └─ Try: CPU offload (last resort)
│
├─ Problem: Too Slow
│  ├─ Sequences < 2000: Enable batching
│  ├─ Sequences 2000-6000: Use Flash Attention
│  ├─ Sequences > 6000: Reduce length or use gradient checkpointing
│  └─ Production: Use vLLM or TensorRT
│
└─ Problem: Low Accuracy
   ├─ Try: Hyperparameter tuning (grid search)
   ├─ Try: Better attention head selection (MI-based)
   ├─ Try: Ensemble methods
   ├─ Try: More training data (run on full 17K samples)
   └─ Try: Threshold optimization
```

---

## Benchmarking Template

```python
import time
import torch

def benchmark_configuration(config_name, config):
    """Benchmark a configuration"""
    print(f"\n=== Benchmarking: {config_name} ===")
    
    # Setup
    model = load_model(**config['model'])
    test_samples = samples[:100]
    
    # Warmup
    for sample in test_samples[:5]:
        process(model, sample)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    results = []
    for sample in test_samples:
        result = process(model, sample)
        results.append(result)
    
    torch.cuda.synchronize()
    end = time.time()
    
    # Metrics
    total_time = end - start
    speed = len(test_samples) / total_time
    memory = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"Speed: {speed:.2f} samples/min")
    print(f"Memory: {memory:.2f}GB")
    print(f"Total time: {total_time:.1f}s")
    
    return {
        'speed': speed,
        'memory': memory,
        'time': total_time
    }

# Run benchmarks
configs = {
    'baseline': {
        'model': {'quantization': '4bit', 'max_length': 6000}
    },
    'optimized': {
        'model': {'quantization': '4bit', 'max_length': 8000, 'flash_attn': True}
    }
}

for name, config in configs.items():
    benchmark_configuration(name, config)
```

---

## 🔗 Related Documentation

- [Architecture](ARCHITECTURE.md) - System design
- [Implementation Notes](IMPLEMENTATION_NOTES.md) - Technical decisions
- [Troubleshooting](TROUBLESHOOTING.md) - Problem solving

---

**Last Updated**: November 2024  
**Maintained by**: [Your Name]

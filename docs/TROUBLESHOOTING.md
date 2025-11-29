# Troubleshooting Guide

> Solutions to common problems and error messages

## 📋 Table of Contents
- [Quick Diagnostics](#quick-diagnostics)
- [Memory Issues](#memory-issues)
- [Path & File Errors](#path--file-errors)
- [Model Loading Issues](#model-loading-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Data Issues](#data-issues)

---

## Quick Diagnostics

### Run This First

```python
# Diagnostic script
import torch
import sys

print("=== Environment Check ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}GB")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f}GB")
        print(f"  Cached: {torch.cuda.memory_reserved(i) / 1e9:.2f}GB")
```

**Expected Output (Kaggle T4x2)**:
```
Python: 3.10.x
PyTorch: 2.0.x
CUDA Available: True
GPU Count: 2

GPU 0: Tesla T4
  Total Memory: 15.00GB
  Allocated: 0.00GB
  Cached: 0.00GB

GPU 1: Tesla T4
  Total Memory: 15.00GB
  Allocated: 0.00GB
  Cached: 0.00GB
```

---

## Memory Issues

### Error 1: CUDA Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 180.00 MiB. GPU 0 has a total capacity of 14.74 GiB 
of which 100.12 MiB is free.
```

#### Cause
- Model + activations exceed available VRAM
- Memory fragmentation
- Previous run didn't release memory

#### Solutions (Try in Order)

**Solution 1: Restart Kernel**
```python
# In Jupyter/Kaggle
# Kernel → Restart Kernel
# Then re-run from beginning
```

**Solution 2: Clear Cache**
```python
import torch
import gc

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

# Verify cleared
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f}GB")
```

**Solution 3: Reduce Sequence Length**
```python
# In detection notebook, find this line:
'prompt[:8000]' 

# Change to:
'prompt[:6000]'  # Or even smaller: 4000, 3000
```

**Solution 4: Increase Cleanup Frequency**
```python
# Find this pattern:
if (processed_count % 5 == 0):
    torch.cuda.empty_cache()

# Change to:
if (processed_count % 3 == 0):  # Clean every 3 samples instead of 5
```

**Solution 5: Set Memory Limits**
```python
# In model loading section:
max_memory = {0: "12GB", 1: "12GB"}

# Reduce to:
max_memory = {0: "10GB", 1: "10GB"}
```

**Solution 6: Use CPU Fallback (Last Resort)**
```python
# Change device_map:
device_map = "balanced"

# To:
device_map = "auto"  # Will use CPU if GPU full
```

---

### Error 2: Memory Leak (Gradual Memory Growth)

```
Initial: GPU 0: 5.2GB
After 10: GPU 0: 6.8GB
After 20: GPU 0: 8.4GB  ← Growing!
After 30: OOM
```

#### Cause
- References not being released
- Gradients being computed unnecessarily
- Attention outputs accumulating

#### Solution

**Fix 1: Disable Gradients**
```python
# Add this after model loading:
model.eval()
for param in model.parameters():
    param.requires_grad = False
```

**Fix 2: Delete Intermediate Results**
```python
for sample in samples:
    outputs = model(sample)
    result = process(outputs)
    save(result)
    
    # Add this:
    del outputs
    torch.cuda.empty_cache()
```

**Fix 3: Use Context Manager**
```python
with torch.no_grad():  # Prevents gradient computation
    for sample in samples:
        outputs = model(sample)
        process(outputs)
```

---

### Error 3: Memory Fragmentation

```
RuntimeError: CUDA out of memory. Tried to allocate 50MB.
Total memory: 15GB, Allocated: 8GB, Free: 7GB
```

**Why**: Free memory is fragmented (not contiguous)

#### Solution

```python
# Set this environment variable BEFORE importing torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Then import torch
import torch
```

Or run at start of notebook:
```bash
!export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Path & File Errors

### Error 4: FileNotFoundError

```
FileNotFoundError: [Errno 2] No such file or directory: 
'/kaggle/working/ReDeEP-ICLR/dataset/ragtruth/response_spans.jsonl'
```

#### Cause
Path pointing to wrong directory (INPUT vs WORKING)

#### Solution

**Identify Directory Type**:
- **INPUT** (`/kaggle/input/`): Dataset files (read-only)
- **WORKING** (`/kaggle/working/`): Output files (writable)

**Fix Paths**:
```python
# Dataset files should use INPUT
DATASET_DIR = "/kaggle/input/redeep-folder/ReDeEP-ICLR/dataset/ragtruth"

response_path = f"{DATASET_DIR}/response_spans.jsonl"  # ✓ INPUT
source_info_path = f"{DATASET_DIR}/source_info_spans.jsonl"  # ✓ INPUT

# Output files should use WORKING
OUTPUT_BASE = "/kaggle/working/ReDeEP-ICLR"
LOG_DIR = f"{OUTPUT_BASE}/log/test_llama2_7B"

output_path = f"{LOG_DIR}/results.json"  # ✓ WORKING
```

**Verify Files Exist**:
```python
import os

files_to_check = [
    "/kaggle/input/redeep-folder/ReDeEP-ICLR/dataset/ragtruth/response_spans.jsonl",
    "/kaggle/input/redeep-folder/ReDeEP-ICLR/dataset/ragtruth/source_info_spans.jsonl",
    "/kaggle/input/redeep-folder/ReDeEP-ICLR/dataset/ragtruth/source_info.jsonl"
]

for file_path in files_to_check:
    exists = os.path.exists(file_path)
    print(f"{'✓' if exists else '✗'} {file_path}")
```

---

### Error 5: Old Google Drive Paths

```
FileNotFoundError: /content/drive/MyDrive/ReDeEP-ICLR/...
```

#### Cause
Script still has hardcoded Google Drive paths

#### Solution

**Find and Replace**:
```python
# Bad (Google Drive)
BASE_DIR = "/content/drive/MyDrive/ReDeEP-ICLR"

# Good (Kaggle)
BASE_DIR = "/kaggle/working/ReDeEP-ICLR"
```

**Use Universal Patcher** (see `src/path_manager.py`):
```python
from src.path_manager import universal_path_patch

with open('script.py') as f:
    content = f.read()

patched = universal_path_patch(content, 'detection')

with open('script_patched.py', 'w') as f:
    f.write(patched)
```

---

## Model Loading Issues

### Error 6: HuggingFace Authentication Failed

```
OSError: You are trying to access a gated repo.
Make sure to request access at https://huggingface.co/meta-llama/Llama-2-7b-hf 
and pass a token having permission to this repo
```

#### Cause
- No HuggingFace token
- Token not accepted for model
- Token not added to Kaggle Secrets

#### Solution

**Step 1: Get HuggingFace Token**
1. Go to https://huggingface.co/settings/tokens
2. Create token with "read" access
3. Copy token

**Step 2: Request Model Access**
1. Visit https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Click "Request Access"
3. Wait for approval (usually instant)

**Step 3: Add to Kaggle**
1. Kaggle Notebook → Add-ons → Secrets
2. Click "Add a new secret"
3. Name: `HF_TOKEN`
4. Value: Paste your token
5. Save

**Step 4: Use in Code**
```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)
```

---

### Error 7: Model Loading Hangs

```
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
[Freezes here forever]
```

#### Cause
- Network timeout
- Insufficient disk space
- Cache corruption

#### Solutions

**Solution 1: Check Disk Space**
```python
import shutil
disk = shutil.disk_usage("/kaggle/working")
print(f"Free space: {disk.free / 1e9:.2f}GB")

# Need at least 20GB free
```

**Solution 2: Clear HuggingFace Cache**
```python
!rm -rf /root/.cache/huggingface/
```

**Solution 3: Set Timeout**
```python
import os
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes
```

**Solution 4: Download with Resume**
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    resume_download=True,  # Resume if interrupted
    force_download=False   # Use cache if available
)
```

---

## Runtime Errors

### Error 8: IndexError in Regression

```
IndexError: list index out of range
File "chunk_level_reg.py", line 67
data_dict[f"param_{k}"].append(
    list(resp["scores"][j]["parameter_knowledge_scores"].values())[k]
)
```

#### Cause
Not all samples have same number of parameter knowledge scores

#### Solution

**Already Fixed in Latest Version**, but if you see it:

```python
# Old (crashes)
value = list(scores.values())[k]

# New (safe)
values_list = list(scores.values())
value = values_list[k] if k < len(values_list) else 0.0
data_dict[f"param_{k}"].append(value)
```

---

### Error 9: KeyError: 'hallucination_label'

```
KeyError: 'hallucination_label'
File "chunk_level_reg.py", line 114
auc = roc_auc_score(1 - df['hallucination_label'], ...)
```

#### Cause
DataFrame missing hallucination_label column

#### Solution

**Check DataFrame Construction**:
```python
# Verify column exists
print("Columns:", df.columns.tolist())

# If missing, check data dict initialization:
data_dict = {
    "identifier": [],
    "type": [],
    "hallucination_label": [],  # ← Must be here
    **{f"ES_{k}": [] for k in range(32)},
    **{f"PK_{k}": [] for k in range(32)}
}
```

**Also Check**:
```python
# Don't slice away the label column
# Bad:
df_subset = df.iloc[:, :int(df.shape[1] * 0.5)]  # May lose label!

# Good:
df_subset = df  # Use full dataframe
```

---

### Error 10: IndentationError

```
IndentationError: unexpected indent
File "script.py", line 68
```

#### Cause
Path patching broke Python indentation

#### Solution

**Manual Fix**:
1. Open the patched script
2. Find line 68
3. Check indentation matches surrounding code
4. Use spaces (not tabs)

**Prevention**:
```python
# When patching, preserve indentation
old_line = "    original_code()"
new_line = "    new_code()"  # Same indentation (4 spaces)
```

---

## Performance Issues

### Issue 11: Very Slow Processing

```
Processing: 1.2 samples/minute
Expected: 10-15 samples/minute
```

#### Diagnosis

```python
import time

# Time each component
start = time.time()
tokens = tokenizer(text)
print(f"Tokenization: {time.time() - start:.2f}s")

start = time.time()
outputs = model(tokens)
print(f"Model forward: {time.time() - start:.2f}s")

start = time.time()
scores = calculate_scores(outputs)
print(f"Score calc: {time.time() - start:.2f}s")
```

#### Common Causes & Fixes

**Cause 1: Sequences Too Long**
```python
# Check sequence lengths
lengths = [len(tokenizer(s.text).input_ids) for s in samples[:100]]
print(f"Mean length: {sum(lengths)/len(lengths):.0f} tokens")
print(f"Max length: {max(lengths)} tokens")

# If mean > 6000, reduce truncation limit
```

**Cause 2: CPU Bottleneck**
```python
# Check if using CPU instead of GPU
print(f"Model device: {next(model.parameters()).device}")

# Should say: cuda:0 or cuda:1
# If says cpu, model didn't load on GPU properly
```

**Cause 3: Not Using quantization**
```python
# Check model precision
print(f"Model dtype: {next(model.parameters()).dtype}")

# Should say: torch.float16 or torch.uint8 (quantized)
# If says torch.float32, not quantized
```

---

### Issue 12: High Memory but Slow Speed

```
GPU Memory: 13GB/15GB used
Speed: 2 samples/minute
```

#### Cause
Memory-bound but not compute-bound (unusual)

#### Solution

**Check for Memory Leaks**:
```python
import tracemalloc
tracemalloc.start()

for i, sample in enumerate(samples):
    process(sample)
    
    if i % 10 == 0:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current / 1e9:.2f}GB, Peak: {peak / 1e9:.2f}GB")
```

**Reduce Memory Pressure**:
```python
# Process in smaller chunks
for sample in samples:
    with torch.cuda.amp.autocast():  # Mixed precision
        outputs = model(sample)
    process(outputs)
    
    torch.cuda.empty_cache()
```

---

## Data Issues

### Issue 13: No Variation in Attention Scores

```
ConstantInputWarning: An input array is constant; 
the correlation coefficient is not defined.
(Repeated 31 times)
```

#### Diagnosis

```python
# Check attention score variation
import numpy as np

for i in range(32):
    scores = df[f'ES_{i}'].values
    unique_count = len(np.unique(scores))
    print(f"ES_{i}: {unique_count} unique values")

# If many show "1 unique values", that's the problem
```

#### Causes
1. Sequence truncation too aggressive → All truncated to same length
2. Dataset too homogeneous → All samples similar
3. Wrong attention heads selected

#### Solutions

**Solution 1: Use Different Attention Heads**
```python
# Try different heads
topk_heads = []
for layer in [0, 5, 10, 15, 20, 25, 30]:  # Different layers
    for head in [0, 8, 16, 24, 31]:  # Different heads
        topk_heads.append([layer, head])
```

**Solution 2: Reduce Truncation**
```python
# Allow longer sequences
'prompt[:6000]' → 'prompt[:10000]'

# But watch for OOM!
```

**Solution 3: Filter Constant Features**
```python
# Before regression
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"Dropping constant column: {col}")
        df = df.drop(columns=[col])
```

---

### Issue 14: Imbalanced Classes

```
Class distribution:
Factual: 95%
Hallucinated: 5%
```

#### Impact
- High accuracy but useless (predicting all factual)
- Low recall for hallucinations
- Misleading metrics

#### Solutions

**Solution 1: Use Stratified Sampling**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y,  # Maintains class distribution
    random_state=42
)
```

**Solution 2: Oversample Minority Class**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**Solution 3: Use Class Weights**
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)

# Use in model training
model.fit(X, y, sample_weight=class_weights)
```

**Solution 4: Focus on Right Metrics**
```python
# Don't use accuracy - use:
from sklearn.metrics import (
    roc_auc_score,      # Best for imbalanced data
    precision_score,
    recall_score,
    f1_score
)

print(f"AUC: {roc_auc_score(y_true, y_pred)}")  # Most important
print(f"Precision: {precision_score(y_true, y_pred)}")
print(f"Recall: {recall_score(y_true, y_pred)}")
```

---

## Still Having Issues?

### Debug Checklist

- [ ] Restarted kernel
- [ ] Cleared GPU cache
- [ ] Verified file paths
- [ ] Checked HuggingFace auth
- [ ] Confirmed GPU availability
- [ ] Reviewed error traceback
- [ ] Checked disk space (>20GB free)
- [ ] Updated libraries (`pip install -U transformers`)

### Get Help

1. **Check Documentation**:
   - [Architecture](ARCHITECTURE.md)
   - [Implementation Notes](IMPLEMENTATION_NOTES.md)

2. **Search GitHub Issues**:
   - Someone may have hit same problem

3. **Open New Issue**:
   Include:
   - Full error message
   - Environment (Kaggle/Colab/local)
   - GPU type and memory
   - Sample size being processed
   - Relevant code snippet

4. **Contact**:
   - Email: your.email@example.com
   - GitHub: @yourusername

---

**Last Updated**: November 2024  
**Maintained by**: [Your Name]

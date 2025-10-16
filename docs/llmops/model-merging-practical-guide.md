# Model Merging Practical Implementation Guide

This guide provides hands-on code examples for implementing model merging using PyTorch and Transformers library.

## Table of Contents
1. [Manual Weight Averaging](#manual-weight-averaging)
2. [SLERP Implementation](#slerp-implementation)  
3. [TIES-Merging](#ties-merging)
4. [Using Mergekit](#using-mergekit)
5. [PEFT/LoRA Merging](#peft-lora-merging)
6. [Fisher Information Merging](#fisher-information-merging)
7. [Best Practices](#best-practices)

## Manual Weight Averaging

### Simple Linear Interpolation

```python
# Manual Model Weight Averaging (Simple Linear Interpolation)
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from collections import OrderedDict

def linear_interpolation_merge(model1_state, model2_state, alpha=0.5):
    """
    Simple linear interpolation between two model state dictionaries
    merged_weights = alpha * model1 + (1 - alpha) * model2
    """
    merged_state = OrderedDict()
    
    for key in model1_state.keys():
        if key in model2_state:
            merged_state[key] = alpha * model1_state[key] + (1 - alpha) * model2_state[key]
        else:
            merged_state[key] = model1_state[key]
    
    return merged_state

# Example usage
def merge_two_models(model1_path, model2_path, alpha=0.5, output_path="./merged_model"):
    """Merge two Hugging Face models with linear interpolation"""
    
    # Load models
    model1 = AutoModel.from_pretrained(model1_path)
    model2 = AutoModel.from_pretrained(model2_path)
    tokenizer = AutoTokenizer.from_pretrained(model1_path)
    
    # Get state dictionaries
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    # Merge weights
    merged_state = linear_interpolation_merge(state1, state2, alpha)
    
    # Create new model with merged weights  
    merged_model = AutoModel.from_pretrained(model1_path)
    merged_model.load_state_dict(merged_state)
    
    # Save merged model
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    return merged_model

# Usage example:
# merged_model = merge_two_models("model1_path", "model2_path", alpha=0.3)
```

## SLERP Implementation

### Spherical Linear Interpolation

```python
# SLERP (Spherical Linear Interpolation) Implementation
import torch
import numpy as np

def slerp(v1, v2, t, epsilon=1e-7):
    """
    Spherical linear interpolation between two tensors
    
    Args:
        v1, v2: Input tensors to interpolate between
        t: Interpolation parameter (0 = v1, 1 = v2)
        epsilon: Small value to avoid division by zero
    """
    # Normalize vectors
    v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + epsilon)
    v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + epsilon)
    
    # Compute dot product
    dot = torch.sum(v1_norm * v2_norm, dim=-1, keepdim=True)
    
    # Clamp dot product to avoid numerical issues
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Compute angle
    theta = torch.acos(torch.abs(dot))
    
    # Handle near-parallel vectors (fall back to linear interpolation)
    sin_theta = torch.sin(theta)
    linear_mask = sin_theta < epsilon
    
    # SLERP calculation
    a = torch.sin((1 - t) * theta) / (sin_theta + epsilon)
    b = torch.sin(t * theta) / (sin_theta + epsilon)
    
    # Apply SLERP
    result = a * v1_norm + b * v2_norm
    
    # Use linear interpolation for near-parallel vectors
    linear_result = (1 - t) * v1_norm + t * v2_norm
    result = torch.where(linear_mask, linear_result, result)
    
    return result

def slerp_merge_models(model1_state, model2_state, t=0.5):
    """Apply SLERP to all matching parameters in two models"""
    merged_state = OrderedDict()
    
    for key in model1_state.keys():
        if key in model2_state and model1_state[key].shape == model2_state[key].shape:
            # Flatten tensors for SLERP, then reshape back
            original_shape = model1_state[key].shape
            flat1 = model1_state[key].flatten()
            flat2 = model2_state[key].flatten()
            
            # Apply SLERP
            merged_flat = slerp(flat1, flat2, t)
            merged_state[key] = merged_flat.reshape(original_shape)
        else:
            merged_state[key] = model1_state[key]
    
    return merged_state
```

## TIES-Merging

### Task Interference Elimination and Sign Selection

```python
# TIES-Merging Implementation (Simplified)
import torch
from collections import defaultdict

def trim_parameters(task_vector, density=0.5):
    """Trim small parameters, keeping only top density% by magnitude"""
    flat_params = task_vector.flatten()
    threshold = torch.quantile(torch.abs(flat_params), 1 - density)
    mask = torch.abs(task_vector) >= threshold
    return task_vector * mask

def elect_sign(task_vectors):
    """Resolve sign conflicts by majority vote"""
    # Stack all task vectors
    stacked = torch.stack(task_vectors, dim=0)
    
    # Count positive and negative signs
    pos_count = (stacked > 0).sum(dim=0)
    neg_count = (stacked < 0).sum(dim=0)
    
    # Majority vote for signs
    sign_mask = pos_count >= neg_count
    return sign_mask.float() * 2 - 1  # Convert to -1, 1

def disjoint_merge(task_vectors, sign_vector):
    """Average parameters that agree with elected sign"""
    merged = torch.zeros_like(task_vectors[0])
    count = torch.zeros_like(task_vectors[0])
    
    for tv in task_vectors:
        # Only include parameters that agree with elected sign
        agrees = torch.sign(tv) == torch.sign(sign_vector)
        merged += tv * agrees
        count += agrees
    
    # Avoid division by zero
    count = torch.where(count == 0, torch.ones_like(count), count)
    return merged / count

def ties_merge(base_model_state, fine_tuned_states, density=0.5):
    """
    TIES merging implementation
    
    Args:
        base_model_state: Base model state dict
        fine_tuned_states: List of fine-tuned model state dicts
        density: Fraction of parameters to keep after trimming
    """
    merged_state = base_model_state.copy()
    
    for key in base_model_state.keys():
        if all(key in state for state in fine_tuned_states):
            # Compute task vectors (differences from base)
            task_vectors = []
            for state in fine_tuned_states:
                task_vector = state[key] - base_model_state[key]
                # Trim small parameters
                trimmed_tv = trim_parameters(task_vector, density)
                task_vectors.append(trimmed_tv)
            
            if task_vectors:
                # Elect signs and merge
                sign_vector = elect_sign(task_vectors)
                merged_delta = disjoint_merge(task_vectors, sign_vector)
                
                # Add back to base model
                merged_state[key] = base_model_state[key] + merged_delta
    
    return merged_state
```

## Using Mergekit

### Installation and Setup

```bash
# Install mergekit
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .
```

### YAML Configuration Examples

```yaml
# SLERP Configuration
slices:
  - sources:
    - model: microsoft/DialoGPT-medium
      layer_range: [0, 24]
    - model: microsoft/DialoGPT-large  
      layer_range: [0, 24]
merge_method: slerp
base_model: microsoft/DialoGPT-medium
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
```

```yaml
# TIES Configuration  
models:
  - model: microsoft/DialoGPT-medium
    # no parameters necessary for base model
  - model: microsoft/DialoGPT-large
    parameters:
      density: 0.5
      weight: 0.5
  - model: huggingface/CodeBERTa-small-v1
    parameters:
      density: 0.5
      weight: 0.3
merge_method: ties
base_model: microsoft/DialoGPT-medium
parameters:
  normalize: true
dtype: float16
```

### Python Usage with Mergekit

```python
import yaml
import subprocess
import os

def run_mergekit_merge(config_dict, output_dir, config_file="merge_config.yaml"):
    """Run mergekit merge using Python"""
    
    # Save config to file
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f)
    
    # Run mergekit command
    cmd = [
        "mergekit-yaml", 
        config_file, 
        output_dir,
        "--copy-tokenizer",
        "--allow-crimes", 
        "--out-shard-size", "1B",
        "--lazy-unpickle"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Merge completed successfully! Output saved to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Merge failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

# Example usage
config = {
    'slices': [
        {
            'sources': [
                {'model': 'microsoft/DialoGPT-medium', 'layer_range': [0, 24]},
                {'model': 'microsoft/DialoGPT-large', 'layer_range': [0, 24]}
            ]
        }
    ],
    'merge_method': 'slerp',
    'base_model': 'microsoft/DialoGPT-medium',
    'parameters': {
        't': 0.5
    },
    'dtype': 'bfloat16'
}

# run_mergekit_merge(config, "./merged_model")
```

## PEFT/LoRA Merging

### Basic PEFT Adapter Merging

```python
# PEFT (LoRA/Adapter) Merging with Base Models
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

def merge_peft_with_base(base_model_path, peft_model_path, output_path):
    """
    Merge PEFT adapter with base model
    
    Args:
        base_model_path: Path to base model
        peft_model_path: Path to PEFT adapter
        output_path: Where to save merged model
    """
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load PEFT model
    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    # Merge and unload adapter
    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    return merged_model

# Multiple PEFT adapters merging using PEFT library
def merge_multiple_peft_adapters(base_model_path, adapter_paths, weights, output_path):
    """
    Merge multiple PEFT adapters with different weights
    
    Args:
        base_model_path: Path to base model  
        adapter_paths: List of paths to PEFT adapters
        weights: List of weights for each adapter
        output_path: Where to save merged model
    """
    from peft import PeftModel
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load all adapters
    model = base_model
    adapter_names = []
    
    for i, adapter_path in enumerate(adapter_paths):
        adapter_name = f"adapter_{i}"
        model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
    
    # Set up weighted merging
    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name="merged_adapter",
        combination_type="ties"  # or "linear", "dare_ties", etc.
    )
    
    # Set the merged adapter as active
    model.set_adapter("merged_adapter")
    
    # Merge and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)
    
    return merged_model

# Usage example:
# merge_peft_with_base("meta-llama/Llama-2-7b-hf", "./lora_adapter", "./merged_output")
```

## Fisher Information Merging

### Implementation with Fisher Information Matrix

```python
# Fisher Information Matrix Merging Implementation
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

def compute_fisher_information(model, dataloader, num_samples=1000):
    """
    Compute diagonal Fisher information matrix for a model
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with labeled data
        num_samples: Number of samples to use for Fisher computation
    """
    model.eval()
    fisher_dict = {}
    
    # Initialize Fisher information storage
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param)
    
    samples_processed = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
            
        data, target = data.cuda(), target.cuda()
        model.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Compute loss and gradients
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Accumulate squared gradients (diagonal Fisher approximation)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        
        samples_processed += data.size(0)
    
    # Normalize by number of samples
    for name in fisher_dict:
        fisher_dict[name] /= samples_processed
    
    return fisher_dict

def fisher_weighted_averaging(models, fisher_matrices, normalize=True):
    """
    Merge models using Fisher-weighted averaging
    
    Args:
        models: List of model state dictionaries
        fisher_matrices: List of corresponding Fisher information matrices
        normalize: Whether to normalize Fisher weights
    """
    if len(models) != len(fisher_matrices):
        raise ValueError("Number of models must match number of Fisher matrices")
    
    merged_state = {}
    
    # Get parameter names from first model
    param_names = list(models[0].keys())
    
    for param_name in param_names:
        # Collect parameters and Fisher weights
        params = [model[param_name] for model in models]
        fishers = [fisher[param_name] for fisher in fisher_matrices]
        
        # Stack parameters and Fisher weights
        stacked_params = torch.stack(params, dim=0)  # [num_models, ...param_shape]
        stacked_fishers = torch.stack(fishers, dim=0)  # [num_models, ...param_shape]
        
        # Normalize Fisher weights if requested
        if normalize:
            fisher_sum = stacked_fishers.sum(dim=0, keepdim=True)
            fisher_sum = torch.where(fisher_sum == 0, torch.ones_like(fisher_sum), fisher_sum)
            weights = stacked_fishers / fisher_sum
        else:
            weights = stacked_fishers
        
        # Compute weighted average
        merged_param = (weights * stacked_params).sum(dim=0)
        merged_state[param_name] = merged_param
    
    return merged_state
```

## Best Practices

### 1. Model Compatibility

```python
def check_model_compatibility(model1_path, model2_path):
    """Check if two models can be merged"""
    
    model1 = AutoModel.from_pretrained(model1_path)
    model2 = AutoModel.from_pretrained(model2_path)
    
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    # Check architecture compatibility
    if set(state1.keys()) != set(state2.keys()):
        print("Warning: Models have different parameter names")
        return False
    
    # Check tensor shapes
    for key in state1.keys():
        if state1[key].shape != state2[key].shape:
            print(f"Warning: Shape mismatch for {key}: {state1[key].shape} vs {state2[key].shape}")
            return False
    
    print("Models are compatible for merging!")
    return True
```

### 2. Memory-Efficient Merging

```python
def memory_efficient_merge(model_paths, merge_fn, output_path, chunk_size=1000):
    """Merge models in chunks to save memory"""
    
    import gc
    
    # Load first model as base
    merged_state = torch.load(model_paths[0], map_location='cpu')
    
    for model_path in model_paths[1:]:
        # Load model
        model_state = torch.load(model_path, map_location='cpu')
        
        # Merge in chunks
        merged_state = merge_fn(merged_state, model_state)
        
        # Clear memory
        del model_state
        gc.collect()
    
    # Save merged model
    torch.save(merged_state, output_path)
    return merged_state
```

### 3. Validation and Testing

```python
def validate_merged_model(merged_model_path, test_data):
    """Validate merged model performance"""
    
    from transformers import pipeline
    
    # Load merged model
    pipe = pipeline("text-generation", model=merged_model_path)
    
    # Test on sample data
    for prompt in test_data:
        output = pipe(prompt, max_length=50)
        print(f"Input: {prompt}")
        print(f"Output: {output[0]['generated_text']}")
        print("-" * 50)
```

### 4. Configuration Management

```python
import json
from datetime import datetime

def save_merge_config(config, output_path):
    """Save merge configuration for reproducibility"""
    
    config['timestamp'] = datetime.now().isoformat()
    config['mergekit_version'] = "0.4.2"  # Update with actual version
    
    with open(f"{output_path}/merge_config.json", 'w') as f:
        json.dump(config, f, indent=2)

def load_merge_config(config_path):
    """Load previously saved merge configuration"""
    
    with open(config_path, 'r') as f:
        return json.load(f)
```

## Command Line Usage

### Basic Mergekit Commands

```bash
# SLERP merge
mergekit-yaml slerp_config.yaml ./slerp_output --copy-tokenizer --lazy-unpickle

# TIES merge  
mergekit-yaml ties_config.yaml ./ties_output --copy-tokenizer --allow-crimes

# DARE merge
mergekit-yaml dare_config.yaml ./dare_output --cuda --copy-tokenizer

# With custom options
mergekit-yaml config.yaml ./output \\
    --copy-tokenizer \\
    --allow-crimes \\
    --out-shard-size 1B \\
    --lazy-unpickle \\
    --trust-remote-code
```

### Uploading to Hugging Face Hub

```bash
# Login to Hugging Face
huggingface-cli login

# Upload merged model
huggingface-cli upload username/merged-model-name ./merged_output .
```

This guide provides comprehensive examples for implementing model merging in practice. Choose the appropriate method based on your specific requirements:

- **Linear interpolation**: Simple, fast, good for similar models
- **SLERP**: Better geometric properties, preserves model structure  
- **TIES**: Best for merging multiple task-specific models
- **DARE**: Robust to interference, good empirical results
- **Fisher merging**: Principled approach using parameter importance
- **Mergekit**: Production-ready tool with many advanced methods

Remember to always validate merged models thoroughly before deployment!
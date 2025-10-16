# Model Merging: Theory and Practical Implementation

Model merging is a powerful technique that combines the parameters of multiple trained language models to create a single model with enhanced capabilities, all without requiring GPU resources or additional training data. This approach has proven surprisingly effective, with several merged models achieving state-of-the-art performance on the Open LLM Leaderboard.[1][2][3]

## Theoretical Foundation

### Core Principles

Model merging operates under the assumption that fine-tuned models from a shared initialization often lie in a connected low-loss basin of the parameter space. This geometric property enables weight-space averaging to improve accuracy and robustness beyond individual models. The technique exploits the linear mode connectivity between models, where interpolated parameters maintain reasonable performance throughout the interpolation path.[2][4][5][6]

The fundamental insight is that specialized models can be combined by directly manipulating their weights rather than through ensemble methods or additional training. This creates a unified model that inherits capabilities from multiple sources while maintaining the inference cost of a single model.[7][2]

### Mathematical Framework

The basic mathematical operation underlying most merging methods is weighted parameter averaging:

$$
\theta_{merged} = \sum_{i=1}^{N} w_i \theta_i
$$

where 

$$
\theta_i
$$

 represents the parameters of model 
$$
i$$,
$$

w_i
$$
are the combining weights, and
$$

N$$ is the number of models. Different merging methods vary in how they compute these weights and handle parameter conflicts.[4][2]

## Core Merging Methods

### Linear Interpolation and Model Soups

Linear interpolation represents the simplest merging approach, computing a weighted average of model parameters. The "model soups" method extends this by averaging multiple fine-tuned checkpoints from the same base model, often outperforming the best individual checkpoint.[5][6][2]

The method supports both naive averaging (combining all models equally) and greedy selection (iteratively adding models that improve performance). Research shows that uniform averaging with weights summing to 1.0 typically produces optimal results.[1][2][5]

### Spherical Linear Interpolation (SLERP)

SLERP addresses limitations of linear interpolation by preserving the geometric properties of the parameter space. Unlike linear interpolation, which can reduce parameter magnitudes in high-dimensional spaces, SLERP maintains constant rates of change and preserves directional information that often represents meaningful feature learning.[8][2][1]

The SLERP algorithm normalizes parameter vectors to unit length, calculates angles between them using dot products, and applies trigonometric weighting factors based on the interpolation parameter $$t$$. When vectors are nearly collinear, the method defaults to linear interpolation for computational efficiency.[8][1]

### TIES-Merging

Task Interference Elimination and Sign Selection (TIES) addresses two critical challenges in model merging: parameter redundancy and sign disagreements. The method operates through three sequential steps:[9][1]

1. **Trim**: Eliminates redundant parameters by retaining only the top-k% most significant changes (determined by the density parameter) and resetting others to zero[10][9]
2. **Elect Sign**: Resolves conflicts where different models suggest opposing parameter adjustments by creating a unified sign vector based on the most dominant direction[10][9]
3. **Disjoint Merge**: Averages parameter values that align with the elected sign vector, excluding zero values[9][10]

This structured approach enables TIES to merge multiple models simultaneously while minimizing interference effects.[1][9]

### DARE Merging

Drop And REscale (DARE) merging employs a probabilistic approach similar to TIES but with key differences. Instead of deterministic trimming, DARE randomly resets fine-tuned weights to their original base model values, then rescales the remaining weights to preserve output expectations.[2][4][1]

Mergekit implements two DARE variants: `dare_linear` (without sign election) and `dare_ties` (incorporating TIES sign election). The density parameter controls the probability of dropping each parameter, with values around 0.53 showing empirically strong results.[7][1]

### Fisher Information Merging

Fisher merging provides a principled approach to parameter weighting based on the Fisher information matrix, which quantifies parameter importance. The method computes diagonal approximations to avoid computational complexity:[11][12]

$$
F_{\theta}[j] = E_{x \sim p(x)} \left[ \left( \frac{\partial \log p(y|x,\theta)}{\partial \theta_j} \right)^2 \right]
$$

Parameters are then weighted according to their Fisher information values, creating informed combinations that consider parameter significance rather than treating all parameters equally.[12][11]

## Practical Implementation

### Using Mergekit

Mergekit serves as the primary production-ready tool for model merging, supporting multiple architectures including Llama, Mistral, GPT-NeoX, and StableLM. The library enables both CPU and GPU execution with lazy tensor loading for memory efficiency.[7][1]

Installation requires cloning the repository and installing in editable mode:

```bash
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .
```

Merging operations use YAML configuration files specifying models, methods, and parameters. The `mergekit-yaml` command processes these configurations:[7][1]

```bash
mergekit-yaml config.yaml ./output --copy-tokenizer --lazy-unpickle
```

### Configuration Examples

SLERP configurations support layer-specific interpolation parameters, enabling fine-grained control over different model components:[1]

```yaml
slices:
  - sources:
    - model: model1_path
      layer_range: [0, 32]
    - model: model2_path  
      layer_range: [0, 32]
merge_method: slerp
base_model: model1_path
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
```

TIES configurations specify density and weight parameters for each model, with automatic normalization available:[1]

```yaml
models:
  - model: base_model_path
  - model: specialist_model_1
    parameters:
      density: 0.5
      weight: 0.5
  - model: specialist_model_2
    parameters:
      density: 0.5
      weight: 0.3
merge_method: ties
base_model: base_model_path
parameters:
  normalize: true
dtype: float16
```

### Manual Implementation

For custom applications, manual implementations provide full control over the merging process. Linear interpolation serves as the foundation:[4][1]

```python
def linear_interpolation_merge(model1_state, model2_state, alpha=0.5):
    merged_state = OrderedDict()
    for key in model1_state.keys():
        if key in model2_state:
            merged_state[key] = alpha * model1_state[key] + (1 - alpha) * model2_state[key]
        else:
            merged_state[key] = model1_state[key]
    return merged_state
```

SLERP implementation requires careful handling of numerical stability and edge cases:[13][8]

```python
def slerp(v1, v2, t, epsilon=1e-7):
    v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + epsilon)
    v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + epsilon)
  
    dot = torch.sum(v1_norm * v2_norm, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
  
    theta = torch.acos(torch.abs(dot))
    sin_theta = torch.sin(theta)
    linear_mask = sin_theta < epsilon
  
    a = torch.sin((1 - t) * theta) / (sin_theta + epsilon)
    b = torch.sin(t * theta) / (sin_theta + epsilon)
  
    result = a * v1_norm + b * v2_norm
    linear_result = (1 - t) * v1_norm + t * v2_norm
    result = torch.where(linear_mask, linear_result, result)
  
    return result
```

### PEFT Integration

Parameter-Efficient Fine-Tuning (PEFT) adapters can be merged with base models using the PEFT library. The process involves loading the base model, attaching the adapter, and using the `merge_and_unload()` method:[14][15][16]

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, peft_adapter_path)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_path)
```

Multiple PEFT adapters can be combined using weighted merging with methods like TIES:[14]

```python
model.add_weighted_adapter(
    adapters=["adapter1", "adapter2", "adapter3"],
    weights=[2.0, 1.0, 1.0],
    adapter_name="merged_adapter",
    combination_type="ties",
    density=0.2
)
```

## Implementation Considerations

### Compatibility Requirements

Successful merging requires models to share identical architectures, tensor shapes, and tokenizers. Mismatched vocabularies or architectures will cause failures or produce corrupted models. Pre-merge validation should verify parameter name consistency and tensor shape compatibility.[8][7][1]

### Memory Management

Large language models require careful memory management during merging. Mergekit supports lazy loading to minimize memory usage, loading tensors only when needed. The `--out-shard-size` parameter controls output file chunking, enabling processing on systems with limited RAM.[7][1]

### Performance Optimization

Merging performance can be optimized through several strategies:[7][1]

- Use appropriate data types (bfloat16/float16) to reduce memory footprint[1]
- Enable lazy unpickling with `--lazy-unpickle` for faster loading[1]
- Utilize GPU acceleration with `--cuda` when available[7]
- Implement chunked processing for extremely large models[7]

### Validation and Quality Control

Merged models require thorough validation before deployment. Evaluation should include:[17][1]

- Perplexity measurements on representative datasets[17]
- Task-specific benchmarks (ARC, HellaSwag, MMLU, etc.)[17][1]
- Qualitative assessment of generated outputs[1]
- Comparison with baseline models and ensembles[1]

## Advanced Techniques

### Frankenmerging and Passthrough

The passthrough method enables "Frankenmerging" by concatenating layers from different models, creating architectures with exotic parameter counts. This experimental technique has produced impressive results like SOLAR-10.7B, which combines layers through depth-up scaling.[4][1]

### Evolutionary Optimization

Recent work explores automated discovery of optimal merge recipes through evolutionary algorithms. These methods systematically search hyperparameter spaces to identify high-performing combinations without manual tuning.[3][18]

### Multi-Stage Merging

Complex merging workflows can be implemented through multi-stage approaches, where initial merges serve as inputs to subsequent operations. The `mergekit-multi` tool supports such workflows through unified configuration files.[7]

## Best Practices and Recommendations

### Model Selection

Choose parent models with complementary capabilities rather than redundant skills. Models fine-tuned from the same base initialization typically merge more successfully than those with different origins. Verify compatibility through architecture inspection and small-scale testing before full merging.[5][2][1]

### Hyperparameter Tuning

Systematically explore merge coefficients through grid search or validation-guided selection. For TIES and DARE methods, density parameters around 0.5-0.6 often provide good results, though optimal values vary by model combination. Weight normalization generally improves stability and performance.[2][5][1]

### Quality Assessment

Evaluate merged models across multiple metrics and domains to identify potential weaknesses. Monitor for catastrophic forgetting in specialized capabilities and validate that merged models maintain coherent behavior across different prompt types. Compare performance against both individual parent models and naive ensemble baselines.[17][1]

Model merging represents a powerful paradigm for combining specialized language models into versatile, high-performing systems. Through careful selection of methods, thorough validation, and systematic hyperparameter optimization, practitioners can create merged models that exceed the capabilities of their individual components while maintaining efficient inference characteristics. As the field continues evolving, automated optimization techniques and novel merging algorithms promise to further enhance the effectiveness of this approach.

[1](https://huggingface.co/blog/mlabonne/merge-models)
[2](https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/)
[3](https://arxiv.org/html/2403.13187v1)
[4](https://cameronrwolfe.substack.com/p/model-merging)
[5](https://arxiv.org/abs/2203.05482)
[6](https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf)
[7](https://github.com/arcee-ai/mergekit)
[8](https://github.com/Digitous/LLM-SLERP-Merge)
[9](https://arxiv.org/abs/2306.01708)
[10](https://tanganke.github.io/fusion_bench/algorithms/ties_merging/)
[11](https://tanganke.github.io/fusion_bench/algorithms/fisher_merging/)
[12](https://arxiv.org/abs/2111.09832)
[13](https://towardsdatascience.com/merging-tokens-to-accelerate-llm-inference-with-slerp-38a32bf7f194/)
[14](https://huggingface.co/docs/peft/en/developer_guides/model_merging)
[15](https://mer.vin/2024/03/merge-base-model-and-peft-adapter-and-push-it-to-hf-hub/)
[16](https://github.com/huggingface/peft)
[17](https://www.ionio.ai/blog/merge-ai-models-using-mergekit)
[18](https://www.nature.com/articles/s42256-024-00975-8)
[19](https://www.youtube.com/watch?v=ISNdQcPhsts)
[20](https://stackoverflow.com/questions/79424314/how-to-merge-pefted-models-to-the-base-one-in-transformershuggingface)
[21](https://pureai.substack.com/p/building-a-simple-transformer-using-pytorch)
[22](https://github.com/huggingface/transformers/issues/28025)
[23](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers)
[24](https://stackoverflow.com/questions/77164963/how-to-merge-fine-tuned-adapter-and-pretrained-model-in-hugging-face-transformer)
[25](https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch)
[26](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
[27](https://huggingface.co/docs/transformers/en/index)
[28](https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383)
[29](https://huggingface.co/docs/transformers/en/run_scripts)
[30](https://github.com/hoya012/swa-tutorials-pytorch)
[31](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.WeightAveraging.html)
[32](https://uu.diva-portal.org/smash/get/diva2:1973270/FULLTEXT01.pdf)
[33](https://pytorch-lightning.readthedocs.io/en/1.5.10/extensions/generated/pytorch_lightning.callbacks.StochasticWeightAveraging.html)
[34](https://xmarva.github.io/blog/2025/adapters/)
[35](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html)
[36](https://github.com/sanwooo/df-merge)
[37](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)
[38](https://github.com/leovewc/model_merging-of-Merging-Models-with-Fisher-Weighted-Averaging)
[39](https://discuss.huggingface.co/t/i-wonder-how-to-merge-my-peft-adapter-with-the-base-model-and-finally-get-a-new-whole-model/139138)
[40](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html)
[41](https://arxiv.org/abs/2504.18992)
[42](https://github.com/huggingface/peft/issues/1836)
[43](https://www.learnpytorch.io/06_pytorch_transfer_learning/)

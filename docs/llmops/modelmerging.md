# LLM : Model Merging/Model Fusion

- Combines the parameters of two or more compatible LLMs to produce a single model
- No additional gradient-based training
- Enabling low-cost
- CPU-only creation of new variants that have, in several cases, reached state-of-the-art scores on the Open LLM Leaderboard.
- data-free, fast to iterate, and especially effective when source models contribute complementary capabilities while sharing a common architecture and tokenizer.

What is model merging ?

How model merging is done ?

- It algebraically mixes checkpoints—often as full weights or fine-tuning deltas—so the merged model inherits behaviors from multiple specialized models without re-training on raw data.
- It done offline via weight-space operations like linear averaging, spherical interpolation, or structured selection, and can run entirely on CPU with libraries such as mergekit.

Why does model merging works ?

- The fine-tuned models from a shared initialization often lie in a connected low-loss basin, so averaging their weights can improve accuracy and robustness—an effect popularized as “model soups”.
- The main failure mode is parameter interference (e.g., redundant updates and sign conflicts across models), which specialized schemes like TIES address by trimming small deltas and resolving sign disagreements before merging.
- Thus, well designed merging can generalize better than naive averages and provide stronger starting points for subsequent fine-tuning.

What are the most common methods for it ?

**Linear/Soup Averaging** :



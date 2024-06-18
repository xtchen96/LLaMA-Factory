## Done

1. SFT with default param (deepspeed zero3) on downloaded alpaca dataset with 4 A100 GPUs. 
- Usage `CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/sft_exp/llama3_full_sft.yaml`.
- Note: need to run twice to tokenize the dataset at first time. To run the full dataset the `max_samples` param needs to be disabled before tokenizing.
- Approximate time: ~ 7 hours (52202 data x 3 epochs * 0.9 train/val ratio / 8 batch_size (4GPU x 2grad_accum) ~= 17700 iters, 1.5s per iter)
- GPU usage: ~ 40GB x 4 GPUs

2. Evaluation on models in 1. Only ran on 1000 samples subset.
- Usage `CUDA_VISIBLE_DEVICES=0 llamafactory-cli eval examples/sft_exp/llama3_full_eval.yaml`.
- Note: change dataset->task param to be {mmlu, cmmlu, ceval} to switch benchmarks. Other benchmarks are not supported in this repo.

3. SFT with deepspeed zero3 + flashattention2 on alpaca. Only ran on 1000 samples subset.
- Usage `CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/sft_exp/llama3_full_sft_fa2.yaml`.
- Note: almost same speed and GPU memory compared to deepspeed only.

4. SFT with unsloth + flashattention2 on alpaca. **not working, multi-GPU with unsloth needs commercial license**
- Usage `CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/sft_exp/llama3_full_sft_fa2_unsloth.yaml`.

## References
References from `README.md`

#### Batch Predicting and Computing BLEU and ROUGE Scores

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_predict.yaml
```
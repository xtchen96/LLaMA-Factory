## Usage

### SFT + Evaluation on Alpaca Dataset

1. SFT train
    - default param (deepspeed zero3 + flash_attn2): `llamafactory-cli train examples/sft_exp/llama3_full_sft_alpaca_train.yaml`
    - no flash_attn2: `llamafactory-cli train examples/sft_exp/llama3_full_sft_nofa2_alpaca_train.yaml`
    - Note: need to run twice to tokenize the dataset at first time. To run the full dataset the `max_samples` param needs to be disabled before tokenizing.
    - Approximate time: ~ 3.5 hours for 8 GPUs, 7 hours for 4 GPUs. 3 epochs for SFT to converge

2. SFT eval
    - on sft-ed model `llamafactory-cli train examples/sft_exp/llama3_full_sft_alpaca_eval.yaml`.
    - on default model `llamafactory-cli train examples/sft_exp/llama3_full_sft_alpaca_eval_baseline.yaml`.
    - run evaluation and comparison: **TODO**
    - Note: ~ 35 mins to finish one evaluation on alpaca_eval


## Others

- Done once
    - Alpaca_eval. To get gpt4-turbo result on alpaca_eval and convert json file format, to reuse in comparison with sft and baseline models
        - `alpaca_eval --model_outputs 'data/alpaca_eval.json' --annotators_config 'alpaca_eval_gpt4_turbo_fn' --output_path 'data/alpaca_eval_gpt'`

- Evaluation on mmlu/cmmlu/ceval
    - Usage `llamafactory-cli eval examples/sft_exp/llama3_full_eval.yaml`.
    - Note: change dataset->task param to be {mmlu, cmmlu, ceval} to switch benchmarks. Other benchmarks are not supported in this repo.

- SFT with unsloth + flashattention2 on alpaca. **not working, multi-GPU with unsloth needs commercial license**
    - Usage `CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/sft_exp/llama3_full_sft_fa2_unsloth.yaml`.

## References
References from `README.md`

#### Batch Predicting and Computing BLEU and ROUGE Scores

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_predict.yaml
```
import os
import json
import yaml
from copy import deepcopy


# set your api key here
os.environ['OPENAI_API_KEY'] = 'your API key'

folders = {
    'llama-factory': '/data/home/xiaotong/LLaMA-Factory/benchmark',
    'llm-train-eval': '/data/home/xiaotong/llm_train_eval/experiments/exp03/benchmark',
    'xtuner': '/data/home/xiaotong/xtuner/work_dirs'
}

gpt_generator_name = 'gpt4_1106_preview'

output_file = 'generated_predictions.jsonl'
output_fmt_file = 'model_output.json'

gpt_result_dir = '/data/home/xiaotong/LLaMA-Factory/data/alpaca_eval_gpt/alpaca_eval_gpt4'
gpt_result_file = 'annotations.json'
gpt_result_convert_file = 'annotations_converted.json'
gpt_result_fmt_file = 'gpt_output.json'

reference_fmt_file = '/data/home/xiaotong/LLaMA-Factory/data/alpaca_eval.json'

output_dir_root = '/data/home/xiaotong/alpaca_eval_output'
template_filepath = '/data/home/xiaotong/LLaMA-Factory/examples/sft_exp/llama3_full_sft_alpaca_eval.yaml'
temporary_filepath = template_filepath.replace('.yaml', '_tmp.yaml')
with open(template_filepath, 'r') as f:
    params = yaml.safe_load(f)

ds_fsdp_config_path = {
    'fsdp': 'accelerate launch --config_file /data/home/xiaotong/LLaMA-Factory/examples/accelerate/fsdp_config.yaml src/train.py',
    'dsz3': 'llamafactory-cli train'
}

runs = [
    ['llama3-8B-baseline', 'meta-llama/Meta-Llama-3-8B'],
    ['phi-3-mini-baseline', 'microsoft/Phi-3-mini-4k-instruct']
]
for framework, folder in folders.items():
    for tmp in os.listdir(folder):
        # for llamafactory and xtuner, need to go inside global/checkpoint/iter_xxx folder
        if framework != 'llm-train-eval':
            for t in os.listdir(folder + '/' + tmp):
                if os.path.isdir(folder + '/' + tmp + '/' + t):
                    if 'checkpoint' in t or ('iter' in t and 'hf' in t):
                        runs.append([framework, folder + '/' + tmp + '/' + t])
                        break
        else:
            runs.append([framework, folder + '/' + tmp])

os.environ["WANDB_DISABLED"] = "true"
for run in runs:
    framework, folder = run
    if framework == 'llm-train-eval': # doesn't save checkpoint so 1 fewer layer
        output_generator_name = framework + '_' + folder.split('/')[-1]
    else:
        output_generator_name = framework + '_' + folder.split('/')[-2]
    output_dir = f'{output_dir_root}/{output_generator_name}'
    os.system(f'mkdir -p {output_dir}')

    print(folder + 5*'\n')
    # generate output on alpaca_eval
    if not os.path.exists(f'{output_dir}/{output_file}'):
        p = deepcopy(params)
        if 'llama3' in output_dir:
            model_name = 'llama3'
        elif 'phi' in output_dir:
            model_name = 'phi'
        else:
            raise ValueError
        p['template'] = model_name
        p['model_name_or_path'] = folder
        p['output_dir'] = output_dir
        with open(temporary_filepath, 'w') as f:
            yaml.dump(p, f)
        # if 'fsdp' in output_dir and 'llama-factory' in output_dir:
        #     ds_fsdp_path = ds_fsdp_config_path['fsdp']
        # else: # baseline model can also use llamafactory-cli
        #     ds_fsdp_path = ds_fsdp_config_path['dsz3']
        ds_fsdp_path = ds_fsdp_config_path['dsz3']
        os.system(f'{ds_fsdp_path} {temporary_filepath}')
    # continue

    # convert output jsonl to json format
    if not os.path.exists(f'{output_dir}/{output_file[:-1]}'):
        os.system(f"jq -s '.' {output_dir}/{output_file} > {output_dir}/{output_file[:-1]}")

    # convert output to alpaca_eval_model_output format
    if not os.path.exists(f'{output_dir}/{output_fmt_file}'):
        with open(reference_fmt_file, 'r') as f:
            d_ref = json.load(f)
        with open(f'{output_dir}/{output_file[:-1]}', 'r') as f:
            d = json.load(f)
        for i in range(len(d_ref)):
            d_ref[i]['output'] = d[i]['predict']
            d_ref[i]['generator'] = output_generator_name
        with open(f'{output_dir}/{output_fmt_file}', 'w') as f:
            json.dump(d_ref, f, indent=4)

    # convert gpt result to alpaca_eval model_output format
    if not os.path.exists(f'{gpt_result_dir}/{gpt_result_convert_file}'):
        with open(reference_fmt_file, 'r') as f:
            d_ref = json.load(f)
        with open(f'{gpt_result_dir}/{gpt_result_file}', 'r') as f:
            d = json.load(f)
        for i in range(len(d_ref)):
            d_ref[i]['output'] = d[i]['output_1']
            d_ref[i]['generator'] = gpt_generator_name    
        with open(f'{gpt_result_dir}/{gpt_result_fmt_file}', 'w') as f:
            json.dump(d_ref, f, indent=4)

    # call alpaca_eval to compare the results through gpt4_turbo_fn
    # cost ~ $5 according to github: https://github.com/tatsu-lab/alpaca_eval?tab=readme-ov-file#evaluators

    os.system(f"alpaca_eval --model_outputs '{output_dir}/{output_fmt_file}' --reference_outputs '{gpt_result_dir}/{gpt_result_fmt_file}' --output_path '{output_dir}'")
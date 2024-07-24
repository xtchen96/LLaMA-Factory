'''
copy file and modify certain params then pass to command
template file: examples/sft_exp/llama3_full_sft_alpaca_train.yaml

command to run deepspeed:
llamafactory-cli train examples/sft_exp/llama3_full_sft_alpaca_train.yaml

command to run fsdp:
accelerate launch --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/extras/fsdp_qlora/llama3_lora_sft.yaml
'''
import os
import yaml
from copy import deepcopy

models = {
    # 'llama3-8b': ['llama3', 'meta-llama/Meta-Llama-3-8B'],
    'phi3-4b': ['phi', 'microsoft/Phi-3-mini-4k-instruct']
}

datasets = [
    'alpaca',
    # 'longalpaca'
]

fa_params = {
    # 'w_fa': 'fa2',
    'wo_fa': 'disabled'
}

ds_fsdp_config_path = {
    'fsdp': 'accelerate launch --config_file examples/accelerate/fsdp_config.yaml src/train.py',
    # 'dsz3': 'llamafactory-cli train'
}
template_filepath = 'examples/sft_exp/llama3_full_sft_alpaca_train.yaml'
temporary_filepath = template_filepath.replace('.yaml', '_tmp.yaml')
with open(template_filepath, 'r') as f:
    params = yaml.safe_load(f)

output_path = './benchmark'
os.system(f'mkdir -p {output_path}')

debug_mode = True

if debug_mode:
    os.environ["WANDB_DISABLED"] = "true"
else:
    os.environ["WANDB_DISABLED"] = "false"
    os.environ['WANDB_PROJECT'] = 'llama_factory'

for fa_name, fa_enable in fa_params.items():
    if fa_enable:
        os.system('pip install flash-attn')
    else:
        os.system('pip uninstall flash-attn -y')
    for ds_fsdp_name, ds_fsdp_path in ds_fsdp_config_path.items():
        for dataset in datasets:
            for model_name, model_path in models.items():
                wandb_name = '_'.join([dataset, model_name, fa_name, ds_fsdp_name])
                output_dir = output_path + '/' + wandb_name

                p = deepcopy(params)
                p['template'] = model_path[0]
                p['model_name_or_path'] = model_path[1]
                p['flash_attn'] = fa_enable
                p['output_dir'] = output_dir
                if ds_fsdp_name == 'fsdp':
                    del p['deepspeed']

                if debug_mode:
                    del p['report_to']
                    del p['run_name']
                else:
                    p['run_name'] = wandb_name
                # print(p)

                with open(temporary_filepath, 'w') as f:
                    yaml.dump(p, f)
                accelerate_training_command = f'{ds_fsdp_path} {temporary_filepath}'
                os.system(accelerate_training_command)
                # print(accelerate_training_command)
                # exit()
                # if ds_fsdp_name == 'fsdp': # need to merge final sharded weights
                #     checkpoint_folder = next(os.walk(output_dir))[1]
                #     merge_command = f'accelerate merge-weights {output_dir}/{checkpoint_folder} {output_dir}'
                #     # os.system(merge_command)
                #     print(merge_command)
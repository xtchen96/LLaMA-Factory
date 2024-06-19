import os
import json

# set your api key here
os.environ['OPENAI_API_KEY'] = 'api_key'

# subject to change according to llm model
output_generator_name = 'llama3-8b-baseline'
output_dir = 'saves/llama3-8b/full/alpaca_eval_baseline'
output_eval_dir = output_generator_name + '-alpacaeval'

# should stay const
gpt_generator_name = 'gpt4_1106_preview'

output_file = 'generated_predictions.jsonl'
output_fmt_file = 'model_output.json'

gpt_result_dir = 'data/alpaca_eval_gpt/alpaca_eval_gpt4'
gpt_result_file = 'annotations.json'
gpt_result_convert_file = 'annotations_converted.json'
gpt_result_fmt_file = 'gpt_output.json'

reference_fmt_file = 'data/alpaca_eval.json'

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

os.system(f"alpaca_eval --model_outputs '{output_dir}/{output_fmt_file}' --reference_outputs '{gpt_result_dir}/{gpt_result_fmt_file}' --output_path '{output_dir}/{output_eval_dir}'")
    
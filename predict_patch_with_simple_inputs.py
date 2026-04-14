import re
import os
import os.path as osp
from tqdm import tqdm
import json
import pandas as pd
import json
import argparse
from configuration import Config
from data_prep.data_utils import build_patch_image
from models.mllm_wrappers import LLaVA, Qwen, Janus


def extract_pattern(response_sentence):
    """
    Extract the response pattern (e.g., 'left', 'middle', 'right') from the model output.
    """
    pattern = r"\b(left|middle|right)\b"
    match = re.search(pattern, response_sentence.lower())
    return match.group(0) if match else None


def utterances_to_dialogue_string(utterances):
    lines = [f'{intl}: {utt}' for intl, utt, _ in utterances]
    return '\n'.join(lines)

# define prompts
TOP_INSTRUCTION = 'In this image you can see three color patches. In the following dialogue, the speaker will describe exactly one of the patches. Please indicate to me whether he refers to the left, middle or right patch.'
BOTTOM_PROMPT = 'Is it the left, middle or right patch?'


def main(config):

    # settings
    img_kwargs = {
        "patch_size": config.patch_size*3,
        "patch_padding": config.grid_padding,
        "patch_pad_color": config.grid_pad_color,
    }

    generate_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
    }

    # load model
    model_kwargs = {"cache_dir": config.model_cache_dir, "quant": config.quant}

    if config.model_type == 'LLaVa':
        Model = LLaVA
    elif config.model_type == 'Qwen':
        Model = Qwen
    elif config.model_type == 'Janus':
        Model = Janus
    else:
        raise NotImplementedError

    model = Model(config.model_id, **model_kwargs)
    config.model_size = model.model_size
    config.prompt = [TOP_INSTRUCTION, BOTTOM_PROMPT]

    print(f'Model: {model.__class__}')
    print(f'ID: {model.model_id}')
    print(f'Quant: {str(model.quant)}')
    
    # load data
    data_df = pd.read_json(osp.join(config.data_dir, "color_patch_data.json"))

    # iterate through data
    results = []
    for i, entry in tqdm(data_df.iterrows(), total=len(data_df)):

        if config.limit is not None:
            if i >= config.limit:
                # for test runs
                break

        # fetch utteranes and generate input image
        utts = entry.conversation
        img = build_patch_image(entry.hls_values, entry.l_order_t_distr1_distr2, **img_kwargs)

        # convert utterances into compatible format
        dialogue_string = utterances_to_dialogue_string(utts)
        full_prompt = '\n\n'.join([TOP_INSTRUCTION, dialogue_string, BOTTOM_PROMPT])

        # generate response
        response_sentence = model.generate(
            full_prompt, img, prune_output_to_response=True, **generate_kwargs)

        # extract location from model response
        predicted_location = extract_pattern(response_sentence)

        # compare with ground truth
        target_position = entry.l_order_t_distr1_distr2.index(0)
        if predicted_location is not None:
            predicted_position = ["left", "middle", "right"].index(predicted_location)
        else:
            predicted_position = -1
        correct = target_position == predicted_position

        # output
        response_data = {
            # item info
            "gameid": entry.gameid,
            "identifier": entry.identifier,
            "roundNum": entry.roundNum,
            "condition": entry.condition,
            # model response
            "response_sentence": response_sentence,
            # resolution info
            "predicted_location": str(predicted_location),
            "target_position": target_position,
            "predicted_position": predicted_position,
            "correct": correct,
        }
        results.append(response_data)
        
    out_data = {
        'config': vars(config),
        'results': results
    }

    model_name = osp.split(model.model_id)[-1]
    out_file = f"colorpatches_{model_name}_q-{config.quant}_simple.json"
    out_path = osp.join(config.output_dir, out_file)
    with open(out_path, "w") as f:
        print(f"save results to {out_path}")
        json.dump(out_data, f)


if __name__ == "__main__":
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--quant", choices=[None, 'none', '4bit', '8bit'], default=None, type=str.lower)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.model_id is not None:
        config.model_id = args.model_id
        config.refresh_model_type()
        
    if args.quant is not None:
        if args.quant == 'none':
            config.quant = None
        else:
            config.quant = args.quant
            
    if args.limit is not None:
        config.limit = args.limit

    if not osp.isdir(config.output_dir):
        # create output dir if it does not exist
        os.makedirs(config.output_dir)

    config.input_type = 'simple'
    config.task = 'patch'

    print("run inference with config:")
    for k, v in vars(config).items():
        print(f"\t{k} : {v}")

    main(config)

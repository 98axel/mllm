import sys
import os
from os import path as osp
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import json
import argparse
import asyncio

async def main(args):

    with open(args.instruction_file, "r") as f:
        instruction_text = f.read()


    print("preparing data...")

    ann_df, data_df, _, _ = prepare_data(
        args.ref_ann_path, args.colorgrid_ann_path, args.ls_data_path
    )

    data_subset = data_df.loc[data_df.gameid.isin(ann_df.game_id)].set_index("round_id")

    if args.limit: 
        data_subset = data_subset.iloc[:args.limit]

    #
    # SETUP CLIENT
    #

    print("setup client...")

    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.llm_server_ip}:{args.port}/v1",
    )

    models = await client.models.list()
    model_id = models.data[0].id

    print(f"set up client with model {model_id}")

    #
    # GENERATE ANNOTATIONS
    #


    async def single_request(
            target_id, target_row, semaphore, 
            max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, 
            presence_penalty=args.presence_penalty, enable_thinking=args.enable_thinking):
                
        input_messages = make_model_input(
            target_row,
            instruction_text,
            patch_size=args.patch_size,
            patch_padding=args.patch_padding,
            grid_padding=args.grid_padding,
            target_padding=args.target_padding
        )

        async with semaphore:
            try:
                chat_response = await client.chat.completions.create(
                    model=model_id,
                    messages=input_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    }
                )

                response_dict = dict(chat_response.choices[0].message)
                
                content = response_dict.get("content")
                if "reasoning" in response_dict.keys():
                    # vLLM
                    reasoning = response_dict.get("reasoning")
                elif "reasoning_content" in response_dict.keys():
                    # llama.cpp
                    reasoning = response_dict.get("reasoning_content")
                else:
                    # fallback / non-reasoning models
                    reasoning = None
                    

                out_obj = {
                    "target_id": target_id, "description": content, "reasoning": reasoning,
                    "error_type": None, "error_msg": None, "status_code": None
                }
            except Exception as e:
                # catch any exceptions and include the error message in the output
                out_obj = {"target_id": target_id, "description": None, "reasoning": None,
                           "error_type": type(e).__name__, "error_msg": str(e), "status_code": getattr(e, "status_code", None)
                }

            return out_obj


    async def batch_requests(data, max_concurrent):
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            single_request(target_id, target_row, semaphore)
            for target_id, target_row in data.iterrows()
        ]
        return await tqdm_asyncio.gather(*tasks)


    # generate and collect responses
    all_responses = await batch_requests(data_subset, max_concurrent=args.max_concurrent)

    #
    # SAVE GENERATED ANNOTATIONS
    #
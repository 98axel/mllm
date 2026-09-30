from tkinter import Image

import torch
import warnings
from transformers import GenerationConfig, BitsAndBytesConfig


class MLLMWrapper:
    """
    Wrapper class with common methods
    """

    @staticmethod
    def prune_generated_tokens_to_response(generated_ids, split_id):
        selection_start = (generated_ids == split_id).nonzero().max().item()
        response_ids = generated_ids[selection_start:]
        return response_ids


class LLaVA(MLLMWrapper):
    """
    Wrapper for LLaVA Models
    """

    # Documentation:
    # https://huggingface.co/docs/transformers/model_doc/llava_next

    def __init__(
        self, model_id="llava-hf/llava-v1.6-mistral-7b-hf", quant=None, **kwargs
    ):
        """
        Constructor method

        Args:
            model_id (str, optional): huggingface model ID. Defaults to "llava-hf/llava-v1.6-mistral-7b-hf".
            quant (str or NoneType, optional): Quantization setting. Defaults to None.
            kwargs: Further parameters, e.g. cache_dir.
        """

        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        # set up quantization
        if quant is not None:
            if quant == "4bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            elif quant == "8bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                raise NotImplementedError(f'{quant} quantization is not supported with LLaVA')
        else:
            self.quantization_config = None

        print(f"building {self.__class__.__name__} model...")

        # set up model and processor
        self.processor = LlavaNextProcessor.from_pretrained(
            model_id, cache_dir=kwargs.get("cache_dir", None),device_map="auto"
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
            cache_dir=kwargs.get("cache_dir", None),
            device_map="auto",
        )
        self.model.generation_config.pad_token_id = (
            self.processor.tokenizer.pad_token_id
        )

        self.model_id = model_id
        
        self.model_size = None
        for possible_size in ['7b', '13b', '34b', '72b']:
            if f'-{possible_size}-' in self.model_id.lower():
                self.model_size = possible_size
        assert self.model_size is not None
        
        self.quant = quant
        self.device = self.model.device
        
    def prune_output_sequence_to_response(self, output_sequence):
        if 'vicuna' in self.model_id:
            sep = 'ASSISTANT: '
        elif self.model_size == '7b':
            sep = '[/INST]'
        else:
            sep = 'assistant\n'
        return output_sequence.split(sep)[-1].strip()
            

    def generate(self, prompt, image, prune_output_to_response=True, **generate_kwargs):
        """
        Generate response for a simple prompt with a single input image.

        Args:
            prompt (str): The prompt given to the model.
            image (PIL.Image): The input image
            prune_output_to_response (bool, optional): Prune the output to the model response (excluding the input prompt).
                Defaults to True.
            generate_kwargs: Further arguments for the huggingface generate API

        Returns:
            str: The model response.
        """


        # create prompt in the right format
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # tokenize and make torch tensor
        inputs = self.processor(image, prompt, return_tensors="pt").to(
            self.model.device
        )
        
        # if specified: force the model to start with a given partial response
        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # make GenerationConfig from generate_kwargs and predict response with model
        generation_config = GenerationConfig(**generate_kwargs)
        output = self.model.generate(**inputs, generation_config=generation_config)

        # transform output ids to string
        response_sentence = self.processor.decode(output[0], skip_special_tokens=True)

        # prune output to model response
        if prune_output_to_response:
            response_sentence = self.prune_output_sequence_to_response(response_sentence)
            
        return response_sentence

    def generate_from_messages(
        self, messages, images, prune_output_to_response=True, **generate_kwargs
    ):
        """
        Generate response given a chat history and (possibly) multiple images.

        Args:
            messages (list[dict]): The chat history with image placeholders.
            images (list[PIL.Image]): List with one or multiple input images.
            prune_output_to_response (bool, optional): Prune the output to the model response (excluding the input prompt).
                Defaults to True.
            generate_kwargs: Further arguments for the huggingface generate API

        Returns:
            str: The model response.
        """

        # transform input chat to prompt string
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        # tokenize and make torch tensor
        inputs = self.processor(
            images=images, text=prompt, padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        # make GenerationConfig from generate_kwargs and predict response with model
        generation_config = GenerationConfig(**generate_kwargs)
        output = self.model.generate(**inputs, generation_config=generation_config)

        # transform output ids to string
        response_sentence = self.processor.decode(output[0], skip_special_tokens=True)
        # prune output to model response
        if prune_output_to_response:
            response_sentence = self.prune_output_sequence_to_response(response_sentence)

        return response_sentence
    

class Qwen(MLLMWrapper):

    # Documentation:
    # https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct

    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", quant=None, **kwargs):

        from transformers import (
            Qwen2VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
        )

        if quant is not None:
            # switch to model_id with quantization
            if model_id == 'Qwen/Qwen2-VL-72B-Instruct':
                assert quant in ['8bit', '4bit', 'awq']
                if quant == '8bit':
                    model_id = 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8'
                elif quant == '4bit':
                    model_id = 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4'
                elif quant == 'awq':
                    model_id = 'Qwen/Qwen2-VL-72B-Instruct-AWQ'
            else:
                raise Exception(f'Quantization "{quant}" not supported for model {model_id}')

        print(f"building {self.__class__.__name__} model...")

        if 'Qwen2' in model_id:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",  # or torch.bfloat16
                # torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir=kwargs.get("cache_dir", None),
            )
        elif 'Qwen3' in model_id:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",  # or torch.bfloat16
                # torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir=kwargs.get("cache_dir", None),
            )
        else:
            raise NotImplementedError(f'Model {model_id} not supported in Qwen class')

        # default processer
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=kwargs.get("cache_dir", None)
        )

        self.model_id = model_id
        
        self.model_size = None
        for possible_size in ['2b', '4b', '7b', '9b','72b']:
            if f'-{possible_size}' in self.model_id.lower():
                self.model_size = possible_size
        assert self.model_size is not None

        self.quant = quant
        self.device = self.model.device

    def generate(self, prompt, image, prune_output_to_response=True, **generate_kwargs):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)

        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # Inference: Generation of the output
        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(
            **inputs, generation_config=generation_config
        )[0]

        if prune_output_to_response:

            split_id = self.processor.tokenizer.encode("<|im_start|>")[0]
            response_ids = self.prune_generated_tokens_to_response(
                response_ids, split_id
            )

            assert response_ids[:3].tolist() == [split_id, 77091, 198], response_ids[
                :3
            ].tolist()
            response_ids = response_ids[3:]

        response = self.processor.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return response
    
    def generate_single_images(self, prompt, image1, image2, image3, prune_output_to_response=True, **generate_kwargs):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image1,
                    },
                    {
                        "type": "image",
                        "image": image2,
                    },
                    {
                        "type": "image",
                        "image": image3,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text], images=[image1, image2, image3], padding=True, return_tensors="pt"
        ).to(self.model.device)

        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # Inference: Generation of the output
        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(
            **inputs, generation_config=generation_config
        )[0]

        if prune_output_to_response:

            split_id = self.processor.tokenizer.encode("<|im_start|>")[0]
            response_ids = self.prune_generated_tokens_to_response(
                response_ids, split_id
            )

            assert response_ids[:3].tolist() == [split_id, 77091, 198], response_ids[
                :3
            ].tolist()
            response_ids = response_ids[3:]

        response = self.processor.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return response


    def generate_from_messages(self, messages, images, prune_output_to_response=True, **generate_kwargs):
        """
        Generate response given a chat history and (possibly) multiple images.

        Args:
            messages (list[dict]): The chat history with image placeholders.
            images (list[PIL.Image]): List with one or multiple input images.
            prune_output_to_response (bool, optional): Prune the output to the model response (excluding the input prompt). 
                Defaults to True.
            generate_kwargs: Further arguments for the huggingface generate API

        Returns:
            str: The model response.
        """
        
        # Preparation for inference
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[prompt], images=images, padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # Inference: Generation of the output
        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(**inputs, generation_config=generation_config)[0]
        
        if prune_output_to_response:

            split_id = self.processor.tokenizer.encode('<|im_start|>')[0]
            response_ids = self.prune_generated_tokens_to_response(response_ids, split_id)
            
            assert response_ids[:3].tolist() == [split_id, 77091, 198], response_ids[:3].tolist()
            response_ids = response_ids[3:]
        
        response = self.processor.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return response


class Qwen3_5(MLLMWrapper):

    # Documentation:
    # https://huggingface.co/Qwen/Qwen3.5-4B

    def __init__(self, model_id="Qwen/Qwen3.5-4B", quant=None, **kwargs):

        from transformers import (
            Qwen3_5ForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
            GPTQConfig
        )

        if quant is not None:
            # switch to model_id with quantization
            if model_id == 'Qwen/Qwen2-VL-72B-Instruct':
                assert quant in ['8bit', '4bit', 'awq']
                if quant == '8bit':
                    model_id = 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8'
                elif quant == '4bit':
                    model_id = 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4'
                elif quant == 'awq':
                    model_id = 'Qwen/Qwen2-VL-72B-Instruct-AWQ'
            else:
                raise Exception(f'Quantization "{quant}" not supported for model {model_id}')

        print(f"building {self.__class__.__name__} model...")

        quantization_config = GPTQConfig(
            bits=4,
            backend="auto_trainable"
        )   

        if '27b' in model_id.lower():
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",  # or torch.bfloat16
                # torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir=kwargs.get("cache_dir", None),
                quantization_config=quantization_config,
            )
        else:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",  # or torch.bfloat16
                # torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir=kwargs.get("cache_dir", None),
                
            )

        # default processer
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=kwargs.get("cache_dir", None)
        )

        self.model_id = model_id
        
        self.model_size = None
        for possible_size in ['0.8b', '2b', '4b', '9b', '27b']:
            if f'-{possible_size}' in self.model_id.lower():
                self.model_size = possible_size
        assert self.model_size is not None

        self.quant = quant
        self.device = self.model.device

    def generate(self, prompt, image, prune_output_to_response=True, **generate_kwargs):
        

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)

        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # Inference: Generation of the output
        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(
            **inputs, generation_config=generation_config
        )[0]

        if prune_output_to_response:

            split_id = self.processor.tokenizer.encode("<|im_start|>")[0]
            response_ids = self.prune_generated_tokens_to_response(
                response_ids, split_id
            )

            # print("split_id:", split_id)
            # print("first tokens:", response_ids[:10].tolist())
            # print("decoded first tokens:", self.processor.decode(response_ids[:10], skip_special_tokens=False))

            assert response_ids[:3].tolist() == [split_id, 74455, 198], response_ids[
                :3
            ].tolist()
            response_ids = response_ids[3:]

        response = self.processor.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # print("model.device:", self.model.device)
        # print("input_ids.device:", inputs["input_ids"].device)

        first_param = next(self.model.parameters())
        # print("first param device:", first_param.device)
        # print("first param dtype:", first_param.dtype)

        # print("Number of generated tokens:", len(response_ids))
        #print("Generated response:", response)


        return response

    # def generate_batch(
    #     self,
    #     prompts,
    #     images,
    #     prune_output_to_response=True,
    #     **generate_kwargs
    # ):

    #     assert len(prompts) == len(images)

    #     messages_batch = []

    #     for prompt, image in zip(prompts, images):

    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "image",
    #                         "image": image,
    #                     },
    #                     {
    #                         "type": "text",
    #                         "text": prompt
    #                     },
    #                 ],
    #             }
    #         ]

    #         messages_batch.append(messages)

    #     # Chat-Template für jedes Sample
    #     texts = [
    #         self.processor.apply_chat_template(
    #             messages,
    #             tokenize=False,
    #             add_generation_prompt=True,
    #             enable_thinking=False
    #         )
    #         for messages in messages_batch
    #     ]

    #     # Für decoder-only generation mit Padding sinnvoll
    #     self.processor.tokenizer.padding_side = "left"

    #     inputs = self.processor(
    #         text=texts,
    #         images=images,
    #         padding=True,
    #         return_tensors="pt"
    #     ).to(self.model.device)

    #     generation_config = GenerationConfig(
    #         **generate_kwargs
    #     )

    #     with torch.inference_mode():
    #         response_ids = self.model.generate(
    #             **inputs,
    #             generation_config=generation_config
    #         )

    #     # model.generate() enthält bei decoder-only Modellen
    #     # Prompt + neue Tokens.
    #     input_length = inputs["input_ids"].shape[1]

    #     generated_ids = response_ids[:, input_length:]

    #     responses = self.processor.batch_decode(
    #         generated_ids,
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=False
    #     )

    #     return responses

    def generate_batch(
        self,
        prompts,
        images,
        prune_output_to_response=True,
        **generate_kwargs
    ):

        assert len(prompts) == len(images)

        # Für decoder-only batched generation
        self.processor.tokenizer.padding_side = "left"

        conversations = []

        for prompt, image in zip(prompts, images):

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            conversations.append(conversation)

        # Wichtig:
        # Liste von Konversationen direkt an apply_chat_template
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        generation_config = GenerationConfig(
            **generate_kwargs
        )

        self.model.eval()

        with torch.inference_mode():

            response_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # Bei Padding haben alle input_ids dieselbe Länge
        input_length = inputs["input_ids"].shape[1]

        generated_ids = response_ids[:, input_length:]

        responses = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return responses
    
    def generate_single_images(self, prompt, image1, image2, image3, prune_output_to_response=True, **generate_kwargs):
        

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image1,
                    },
                    {
                        "type": "image",
                        "image": image2,
                    },
                    {
                        "type": "image",
                        "image": image3,
                    },

                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        inputs = self.processor(
            text=[text], images=[image1, image2, image3], padding=True, return_tensors="pt"
        ).to(self.model.device)

        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # Inference: Generation of the output
        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(
            **inputs, generation_config=generation_config
        )[0]

        if prune_output_to_response:

            split_id = self.processor.tokenizer.encode("<|im_start|>")[0]
            response_ids = self.prune_generated_tokens_to_response(
                response_ids, split_id
            )

            # print("split_id:", split_id)
            # print("first tokens:", response_ids[:10].tolist())
            # print("decoded first tokens:", self.processor.decode(response_ids[:10], skip_special_tokens=False))

            assert response_ids[:3].tolist() == [split_id, 74455, 198], response_ids[
                :3
            ].tolist()
            response_ids = response_ids[3:]

        response = self.processor.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # print("model.device:", self.model.device)
        # print("input_ids.device:", inputs["input_ids"].device)

        first_param = next(self.model.parameters())
        # print("first param device:", first_param.device)
        # print("first param dtype:", first_param.dtype)

        # print("Number of generated tokens:", len(response_ids))
        print("Generated response:", response)


        return response


    def generate_from_messages(self, messages, images, prune_output_to_response=True, **generate_kwargs):
        """
        Generate response given a chat history and (possibly) multiple images.

        Args:
            messages (list[dict]): The chat history with image placeholders.
            images (list[PIL.Image]): List with one or multiple input images.
            prune_output_to_response (bool, optional): Prune the output to the model response (excluding the input prompt). 
                Defaults to True.
            generate_kwargs: Further arguments for the huggingface generate API

        Returns:
            str: The model response.
        """
        
        # Preparation for inference
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[prompt], images=images, padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # Inference: Generation of the output
        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(**inputs, generation_config=generation_config)[0]
        
        if prune_output_to_response:

            split_id = self.processor.tokenizer.encode('<|im_start|>')[0]
            response_ids = self.prune_generated_tokens_to_response(response_ids, split_id)
            
            assert response_ids[:3].tolist() == [split_id, 77091, 198], response_ids[:3].tolist()
            response_ids = response_ids[3:]
        
        print("Number of generated tokens:", len(response_ids))
        print(next(self.model.parameters()).dtype)
        response = self.processor.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return response
    





class Qwen3_5_vLLM(MLLMWrapper):
    """
    vLLM wrapper for Qwen3.5-VL 27B GPTQ Int4 inference.
    Keeps a similar interface to your existing Qwen3_5 wrapper:
        model.generate(prompt, image, **generate_kwargs)
    """

    def __init__(self, model_id="Qwen/Qwen3.5-27B-GPTQ-Int4", quant=None, **kwargs):
        import os
        from transformers import AutoProcessor
        from vllm import LLM

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        print(f"building {self.__class__.__name__} model with vLLM...")

        self.model_id = model_id
        self.quant = quant
        self.model_size = "27b" if "27b" in model_id.lower() else None

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=kwargs.get("cache_dir", None),
            trust_remote_code=True,
        )

        
        #self.model = LLM(
        #    model=model_id,
        #    quantization= "gptq",
        #    dtype= "auto",
        #    trust_remote_code=True,
        #    download_dir=kwargs.get("cache_dir", None),

            # Important for 24 GB VRAM:
        #    gpu_memory_utilization= 0.6,
        #    max_model_len=512,
        #    max_num_seqs=1,
        #    max_num_batched_tokens=512,
        #    limit_mm_per_prompt={"image": 1},

            # Falls nötig:
        #    tensor_parallel_size=1,
        #    enforce_eager=True
        #)

        self.model = LLM(
            model=model_id,
            quantization=kwargs.get("vllm_quantization", "compressed-tensors"),
            dtype=kwargs.get("dtype", "auto"),
            trust_remote_code=True,
            download_dir=kwargs.get("cache_dir", None),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.92),
            max_model_len=kwargs.get("max_model_len", 2048),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            enforce_eager=kwargs.get("enforce_eager", False),
        )

        self.device = "cuda"

    def _build_messages(self, prompt, images):
        content = []

        for image in images:
            content.append({
                "type": "image",
                "image": image,
            })

        content.append({
            "type": "text",
            "text": prompt,
        })

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _build_vllm_prompt(self, messages):
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _sampling_params_from_kwargs(self, generate_kwargs):
        from vllm import SamplingParams

        max_tokens = generate_kwargs.pop(
            "max_new_tokens",
            generate_kwargs.pop("max_tokens", 80)
        )

        temperature = generate_kwargs.pop("temperature", 0.0)
        top_p = generate_kwargs.pop("top_p", 1.0)

        # vLLM nutzt max_tokens statt max_new_tokens
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **generate_kwargs,
        )

    def generate(self, prompt, image, prune_output_to_response=True, **generate_kwargs):
        messages = self._build_messages(prompt, [image])
        text_prompt = self._build_vllm_prompt(messages)

        sampling_params = self._sampling_params_from_kwargs(generate_kwargs)

        request = {
            "prompt": text_prompt,
            "multi_modal_data": {
                "image": image,
            },
        }

        outputs = self.model.generate(
            [request],
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text.strip()

    def generate_single_images(
        self,
        prompt,
        image1,
        image2,
        image3,
        prune_output_to_response=True,
        **generate_kwargs
    ):
        images = [image1, image2, image3]
        messages = self._build_messages(prompt, images)
        text_prompt = self._build_vllm_prompt(messages)

        sampling_params = self._sampling_params_from_kwargs(generate_kwargs)

        request = {
            "prompt": text_prompt,
            "multi_modal_data": {
                "image": images,
            },
        }

        outputs = self.model.generate(
            [request],
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text.strip()

    def generate_from_messages(
        self,
        messages,
        images,
        prune_output_to_response=True,
        **generate_kwargs
    ):
        text_prompt = self._build_vllm_prompt(messages)
        sampling_params = self._sampling_params_from_kwargs(generate_kwargs)

        request = {
            "prompt": text_prompt,
            "multi_modal_data": {
                "image": images if len(images) > 1 else images[0],
            },
        }

        outputs = self.model.generate(
            [request],
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text.strip()



class Qwen3_5_GGUF(MLLMWrapper):
    """
    GGUF wrapper for Qwen3.5-27B UD-Q4_K_XL with llama-cpp-python.

    Needs:
    - main GGUF model file, e.g. UD-Q4_K_XL.gguf
    - mmproj GGUF file for vision, if using images
    """

    def __init__(
        self,
        model_id=None,
        quant=None,
        **kwargs
    ):
        from llama_cpp import Llama
        import os

        print(f"building {self.__class__.__name__} model with llama.cpp...")

        self.model_id = model_id or "unsloth/Qwen3.5-27B-GGUF"
        self.quant = quant or "UD-Q4_K_XL"

        self.model_path = kwargs.get("model_path", None)
        self.mmproj_path = kwargs.get("mmproj_path", None)

        if self.model_path is None:
            raise ValueError("You must pass model_path='/path/to/UD-Q4_K_XL.gguf'")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        if self.mmproj_path is not None and not os.path.exists(self.mmproj_path):
            raise FileNotFoundError(self.mmproj_path)

        self.model = Llama(
            model_path=self.model_path,

            # GPU offload
            n_gpu_layers=kwargs.get("n_gpu_layers", -1),

            # Context length
            n_ctx=kwargs.get("n_ctx", 2048),

            # Batch settings
            n_batch=kwargs.get("n_batch", 1024),
            n_ubatch=kwargs.get("n_ubatch", 512),

            # Vision projector
            mmproj= self.mmproj_path,

            # Qwen chat format
            chat_format=kwargs.get("chat_format", "qwen"),

            verbose=False,
        )

        self.model_size = "27b"
        self.device = "cuda"

    def generate(
        self,
        prompt,
        image=None,
        prune_output_to_response=True,
        **generate_kwargs
    ):
        max_tokens = generate_kwargs.pop(
            "max_new_tokens",
            generate_kwargs.pop("max_tokens", 150)
        )

        temperature = generate_kwargs.pop("temperature", 0.8)
        top_p = generate_kwargs.pop("top_p", 0.95)

        if image is None:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": self._pil_to_data_url(image),
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return output["choices"][0]["message"]["content"].strip()

    def generate_single_images(
        self,
        prompt,
        image1,
        image2,
        image3,
        prune_output_to_response=True,
        **generate_kwargs
    ):
        # Falls llama.cpp/Qwen3.5-GGUF mehrere Bilder im Chat unterstützt,
        # funktioniert diese Struktur. Falls nicht, besser ein kombiniertes
        # Grid-Bild vorher selbst bauen.
        max_tokens = generate_kwargs.pop(
            "max_new_tokens",
            generate_kwargs.pop("max_tokens", 150)
        )

        temperature = generate_kwargs.pop("temperature", 0.8)
        top_p = generate_kwargs.pop("top_p", 0.95)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": self._pil_to_data_url(image1),
                    },
                    {
                        "type": "image_url",
                        "image_url": self._pil_to_data_url(image2),
                    },
                    {
                        "type": "image_url",
                        "image_url": self._pil_to_data_url(image3),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return output["choices"][0]["message"]["content"].strip()

    def generate_from_messages(
        self,
        messages,
        images=None,
        prune_output_to_response=True,
        **generate_kwargs
    ):
        max_tokens = generate_kwargs.pop(
            "max_new_tokens",
            generate_kwargs.pop("max_tokens", 150)
        )

        temperature = generate_kwargs.pop("temperature", 0.8)
        top_p = generate_kwargs.pop("top_p", 0.95)

        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return output["choices"][0]["message"]["content"].strip()

    def _pil_to_data_url(self, image):
        import base64
        from io import BytesIO

        if image.mode != "RGB":
            image = image.convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{encoded}"
    



class Qwen3_5_GGUF2(MLLMWrapper):
    """
    GGUF wrapper for Qwen3.5-27B Vision via llama-cpp-python.

    Recommended:
    model_path = ".../Qwen3.5-27B-UD-Q4_K_XL.gguf"
    mmproj_path = ".../mmproj-BF16.gguf"
    """

    def __init__(self, model_id=None, quant=None, **kwargs):
        import os
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava16ChatHandler

        print(f"building {self.__class__.__name__} model with llama.cpp...")

        self.model_id = model_id or "unsloth/Qwen3.5-27B-GGUF"
        self.quant = quant or "UD-Q4_K_XL"

        self.model_path = kwargs.get("model_path")
        self.mmproj_path = kwargs.get("mmproj_path")

        if self.model_path is None:
            raise ValueError("model_path is required")

        if self.mmproj_path is None:
            raise ValueError("mmproj_path is required for image input")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        if not os.path.exists(self.mmproj_path):
            raise FileNotFoundError(self.mmproj_path)

        self.chat_handler = Llava16ChatHandler(
            clip_model_path=self.mmproj_path,
            verbose=False,
        )

        self.model = Llama(
            model_path=self.model_path,
            chat_handler=self.chat_handler,

            n_gpu_layers=kwargs.get("n_gpu_layers", -1),
            n_ctx=kwargs.get("n_ctx", 2048),

            n_batch=kwargs.get("n_batch", 1024),
            n_ubatch=kwargs.get("n_ubatch", 512),

            verbose=False,
        )

        self.model_size = "27b"
        self.device = "cuda"

    def generate(
        self,
        prompt,
        image,
        prune_output_to_response=True,
        **generate_kwargs
    ):
        max_tokens = generate_kwargs.pop(
            "max_new_tokens",
            generate_kwargs.pop("max_tokens", 150)
        )

        temperature = generate_kwargs.pop("temperature", 0.8)
        top_p = generate_kwargs.pop("top_p", 0.95)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._pil_to_data_url(image)
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return output["choices"][0]["message"]["content"].strip()

    def generate_from_messages(
        self,
        messages,
        images=None,
        prune_output_to_response=True,
        **generate_kwargs
    ):
        max_tokens = generate_kwargs.pop(
            "max_new_tokens",
            generate_kwargs.pop("max_tokens", 150)
        )

        temperature = generate_kwargs.pop("temperature", 0.8)
        top_p = generate_kwargs.pop("top_p", 0.95)

        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return output["choices"][0]["message"]["content"].strip()

    def _pil_to_data_url(self, image):
        import base64
        from io import BytesIO

        if image.mode != "RGB":
            image = image.convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"



class Janus:
    """
    Wrapper for DeepSeek Janus-Pro-1B and Janus-Pro-7B Models
    """

    def __init__(self, model_id="deepseek-ai/Janus-Pro-1B", quant=None, **kwargs):
        """
        Initialize the model, processor, and tokenizer.

        Args:
            model_id (str): The Hugging Face model ID. Defaults to "deepseek-ai/Janus-Pro-1B".
        """
        
        from janus.models import VLChatProcessor
        from transformers import AutoModelForCausalLM
        #from janus.utils.io import load_pil_images
        
        assert quant is None, 'quantization not implemented for Janus'

        # Load VLChatProcessor and tokenizer
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_id, cache_dir=kwargs.get("cache_dir", None))
        self.tokenizer = self.vl_chat_processor.tokenizer

        # Load the multi-modal model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=kwargs.get("cache_dir", None)
        )

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(torch.bfloat16).to(device).eval()
        
        self.model_id = model_id
        self.model_size = None
        for possible_size in ['1b', '7b']:
            if f'-{possible_size}' in self.model_id.lower():
                self.model_size = possible_size
        assert self.model_size is not None

        self.quant = quant
        self.device = self.model.device


    def generate(self, prompt, image, prune_output_to_response=True, **generate_kwargs):
        """
        Generate a response from the model.

        Args:
            prompt (str): The input text prompt.
            image (PIL.Image or None): Optional image input.

        Returns:
            str: The generated response.
        """

        # Format conversation for Janus-Pro
                # Format conversation based on DeepSeek's template
        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt}"},
            {"role": "<|Assistant|>", "content": ""}, 
        ]
        #if image:
        #    conversation[0]["images"] = [image]

        # Load images if provided
        #pil_images = load_pil_images(conversation) if image else None

        # Prepare model inputs
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=[image], force_batchify=True
        ).to(self.device)

        # Encode images
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **generate_kwargs
        )

        # Decode response
        response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        # Optionally prune response
        if prune_output_to_response:
            response = self.prune_output_sequence_to_response(response)
        
        return response
    
    def prune_output_sequence_to_response(self, output_sequence):
        """
        Extracts only the model's response from the full generated output.

        Args:
            output_sequence (str): The full model output.

        Returns:
            str: The extracted assistant response.
        """

        # Janus-Pro uses <|Assistant|> for assistant responses
        sep = "<|Assistant|>"

        # Split the response to keep only the assistant’s response
        response = output_sequence.split(sep)[-1].strip()

        # Ensure we remove extra stop tokens
        stop_tokens = ["<|User|>", "<｜end▁of▁sentence｜>", "</s>"]
        for token in stop_tokens:
            response = response.split(token)[0].strip()

        return response

    # Probably not needed
    def generate_from_messages(self, messages, images=None, prune_output_to_response=True, **generate_kwargs):
        """
        Generate a response from chat history with optional image inputs.

        Args:
            messages (list[dict]): List of chat messages.
            images (list[PIL.Image] or None): List of input images.
            prune_output_to_response (bool): If True, extracts only the model's response.
            generate_kwargs: Additional generation parameters.

        Returns:
            str: The model's response.
        """

        # Convert messages into Janus format (DeepSeek expects <|User|> and <|Assistant|>)
        conversation = []
        for msg in messages:
            conversation.append({"role": f"<|{msg['role'].capitalize()}|>", "content": msg["content"]})

        # Ensure assistant's turn is empty for model generation
        conversation.append({"role": "<|Assistant|>", "content": ""})

        # Convert images (if provided)
        #pil_images = load_pil_images(conversation) if images else None

        # Prepare model inputs
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=images, force_batchify=True
        ).to(self.device)

        # Encode image inputs (if applicable)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **generate_kwargs,
        )

        # Decode response
        response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

        # Optionally prune response
        if prune_output_to_response:
            response = self.prune_output_sequence_to_response(response)

        return response


class Molmo(MLLMWrapper):

    # Documentation:
    # https://huggingface.co/allenai/Molmo2-8B

    def __init__(self, model_id="allenai/Molmo2-4B", quant=None, **kwargs):

        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )

        if quant is not None:
            # switch to model_id with quantization
            # if model_id == 'Qwen/Qwen2-VL-72B-Instruct':
            #     assert quant in ['8bit', '4bit', 'awq']
            #     if quant == '8bit':
            #         model_id = 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8'
            #     elif quant == '4bit':
            #         model_id = 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4'
            #     elif quant == 'awq':
            #         model_id = 'Qwen/Qwen2-VL-72B-Instruct-AWQ'
            # else:
            raise Exception(f'Quantization "{quant}" not supported for model {model_id}')

        print(f"building {self.__class__.__name__} model...")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",  # or torch.bfloat16
            # torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
            cache_dir=kwargs.get("cache_dir", None),
        )

        # default processer
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
            cache_dir=kwargs.get("cache_dir", None)
        )

        self.model_id = model_id
        print(self.model_id.lower())
        
        self.model_size = None
        for possible_size in ['4b', '8b']:
            if f'-{possible_size}' in self.model_id.lower():
                self.model_size = possible_size
        assert self.model_size is not None

        self.quant = quant
        self.device = self.model.device

    def generate(self, prompt, image, prune_output_to_response=True, **generate_kwargs):

        messages = [
            {
                "role": "user",
                "content": [
                    dict(type="text", text=prompt),
                    dict(type="image", image=image)
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}



        # if generate_kwargs.get("response_start", None) is not None:
        #     response_start = generate_kwargs.pop("response_start")
        #     inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # # Inference: Generation of the output
        # generation_config = GenerationConfig(**generate_kwargs)
        # response_ids = self.model.generate(
        #     **inputs, generation_config=generation_config
        # )[0]

        # if prune_output_to_response:

        #     split_id = self.processor.tokenizer.encode("<|im_start|>")[0]
        #     response_ids = self.prune_generated_tokens_to_response(
        #         response_ids, split_id
        #     )

        #     assert response_ids[:3].tolist() == [split_id, 77091, 198], response_ids[
        #         :3
        #     ].tolist()
        #     response_ids = response_ids[3:]

        # response = self.processor.decode(
        #     response_ids,
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=False,
        # )

        # generate output
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=448)

        # only get generated tokens; decode them to text
        generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        print(generated_text)
        response = generated_text


        return response


    def generate_from_messages(self, messages, images, prune_output_to_response=True, **generate_kwargs):
        """
        Generate response given a chat history and (possibly) multiple images.

        Args:
            messages (list[dict]): The chat history with image placeholders.
            images (list[PIL.Image]): List with one or multiple input images.
            prune_output_to_response (bool, optional): Prune the output to the model response (excluding the input prompt). 
                Defaults to True.
            generate_kwargs: Further arguments for the huggingface generate API

        Returns:
            str: The model response.
        """
        
        # Preparation for inference
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[prompt], images=images, padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # Inference: Generation of the output
        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(**inputs, generation_config=generation_config)[0]
        
        if prune_output_to_response:

            split_id = self.processor.tokenizer.encode('<|im_start|>')[0]
            response_ids = self.prune_generated_tokens_to_response(response_ids, split_id)
            
            assert response_ids[:3].tolist() == [split_id, 77091, 198], response_ids[:3].tolist()
            response_ids = response_ids[3:]
        
        response = self.processor.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return response



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
                # attn_implementation="flash_attention_2",
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
        for possible_size in ['0.8b', '2b', '4b', '9b']:
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
        print("Generated response:", response)


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



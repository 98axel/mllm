from os import path as osp

config_path = osp.abspath(osp.dirname(__file__))

class Config:
    def __init__(self):

        # insert settings
        
        self.model_id = "Qwen/Qwen2-VL-2B-Instruct"
        self.refresh_model_type()
        
        self.quant = None # None, '4bit' , '8bit'
        self.model_cache_dir = 'PATH'
        
        self.patch_size = 100
        self.patch_padding = 10
        self.grid_padding = 50
        self.patch_pad_color = (255, 255, 255)
        self.grid_pad_color = (255, 255, 255)

        self.max_new_tokens = 150
        self.do_sample = True

        self.colorpatch_data = osp.join(config_path, 'colorpatch_data')
        self.colorgrid_data = osp.join(config_path, 'colorgrid_data')
        self.data_dir = osp.join(config_path, 'data')
        self.output_dir = osp.join(config_path, 'results')
        
        self.limit = None
        
        
    def refresh_model_type(self):

        # process settings

        assert self.model_id in [
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "llava-hf/llava-v1.6-vicuna-7b-hf",
            "llava-hf/llava-v1.6-vicuna-13b-hf",
            "llava-hf/llava-v1.6-34b-hf",
            "llava-hf/llava-next-72b-hf",
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-72B-Instruct",
            "deepseek-ai/Janus-Pro-1B",
            "deepseek-ai/Janus-Pro-7B",
            "allenai/Molmo2-4B",    ##needs transformer version == 4.57.1
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.5-4B", ##with thinking ca. 50h
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen3-VL-2B-Instruct",
        ]
        id_prefix = self.model_id.split('/')[0]
        if id_prefix == 'llava-hf':
             self.model_type = 'LLaVa'
        elif id_prefix == 'Qwen':
            model_version = self.model_id.split('/')[1].split('-')[0]
            print(f"model_version: {model_version}")
            if model_version == 'Qwen3.5':
                self.model_type = 'Qwen3_5'
            else:
                self.model_type = 'Qwen'
        elif id_prefix == 'deepseek-ai':
            self.model_type = 'Janus'
        elif id_prefix == 'allenai':
            self.model_type = 'Molmo'

        else:
            raise NotImplementedError
        print(f"model_type: {self.model_type}")
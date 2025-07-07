from typing import List

import PIL
from transformers import AutoModelForCausalLM, AutoProcessor

from og_ego_prim.models.base_client import BaseClient


class HFClient(BaseClient):

    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.preprocessor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.preprocessor.tokenizer
    
    def model(self, prompt, image_file: List[str] | str = None, gen_args={"max_completion_tokens": 256, "temperature": 0.0}): 
        if not image_file: 
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else: 
            if isinstance(image_file, str):  # 支持单图和多图
                image_file = [image_file]    
            messages = [
                {
                    "role": "user",
                    "content": [],
                }
            ]
            for f in image_file:
                messages[0]["content"].append(
                    {
                        "type": "image",
                        "image": f"file://{f}"
                    }
                )
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": prompt
                },
            )
    
        # Preprocess the inputs
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
        
        if not image_file:
            inputs = self.processor(
                text=[text_prompt], images=None, padding=True, return_tensors="pt"
            )
        else: 
            images = [PIL.Image.open(image).convert("RGB") for image in image_file]
            inputs = self.processor(
                text=[text_prompt], images=images, padding=True, return_tensors="pt"
            )
        inputs = inputs.to("cuda")
        
        
        # Inference: Generation of the output
        output_ids = self.model.generate(**inputs, max_new_tokens=gen_args['max_completion_tokens'], temperature=gen_args['temperature'])
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text.strip()

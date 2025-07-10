import os 
import random
import time
from typing import List
from google import genai
from google.genai import types
from openai import OpenAI 

from og_ego_prim.models.base_client import BaseClient
from og_ego_prim.models.image_utils import (
    encode_image, 
    guess_image_type_from_base64,
)

def read_image(image_path: str):
  with open(image_path, "rb") as f:
     return f.read()
  
class ServerClient(BaseClient):

    def __init__(self, model_type, model_name, api_key=os.environ.get("OPENAI_API_KEY"), api_base=os.environ.get("OPENAI_API_BASE")) -> None:
        self.model_type = model_type
        if model_type == "local":
            if "http_proxy" in os.environ:
                del os.environ["http_proxy"], os.environ["HTTP_PROXY"], os.environ["https_proxy"], os.environ["HTTPS_PROXY"]
        ''' add proxy for close-source model'''
        # else:
        #     if "openai.com" in api_base:
        #         os.environ["http_proxy"] = "http://10.1.20.57:23128"
        #         os.environ["https_proxy"] = "http://10.1.20.57:23128"
        #         os.environ["HTTP_PROXY"] = "http://10.1.20.57:23128"
        #         os.environ["HTTPS_PROXY"] = "http://10.1.20.57:23128"
        #     else: 
        #         os.environ['HTTP_PROXY']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        #         os.environ['HTTPS_PROXY']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        #         os.environ['http_proxy']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        #         os.environ['https_proxy']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'

        if 'gemini_direct' in model_name.lower():
            self.client = genai.Client(
                vertexai=True,
                # project="czby-gemini-250612",
                location="global",
            )
            self.generate_content_config = types.GenerateContentConfig(
                temperature = 1,
                top_p = 1,
                seed = 0,
                max_output_tokens = 65535,
                safety_settings = [types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
                )],
            )
        else:
            self.client = OpenAI(api_key=api_key, base_url=api_base)
        
        if model_name == "local":
            model_name = self.client.models.list().data[0].id
        self.model_name = model_name
        print(f"MODEL NAME: {self.model_name}")
        
    def model(self, prompt, image_file: List[str] | str = None, gen_args={"max_completion_tokens": 512, "temperature": 0.0}): 
        if not image_file: 
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            if isinstance(image_file, str):  # support single and multi image
                image_file = [image_file]   
            if 'gemini_direct' in self.model_name.lower(): # support gemini api
                parts=[
                        types.Part.from_text(text=prompt)
                    ]
                for image in image_file:
                    image_base64 = read_image(image)
                    image_type = "image/png"
                    image_content = types.Part.from_bytes(
                        data=image_base64,
                        mime_type=image_type,
                    )
                    parts.append(image_content)
                contents = [
                    types.Content(
                        role="user",
                        parts=parts
                    )
                ]

                for _ in range(3):
                    result = ""
                    try:
                        for chunk in self.client.models.generate_content_stream(
                            model = self.model_name,
                            contents = contents,
                            config = self.generate_content_config,
                            ):
                            result += chunk.text
                        if len(result) == 0:  
                            continue
                        return result
                    except Exception as e:
                        print(e)
                        time.sleep(10)
                        continue        
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                        ],
                    }
                ]
                for image in image_file:
                    image_base64 = encode_image(image)
                    image_type = guess_image_type_from_base64(image_base64)
                    messages[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{image_base64}"
                            },
                        }
                    )
                for _ in range(3):
                    result = ""
                    try:
                        chat_completion = self.client.chat.completions.create(
                                messages=messages,
                                model=self.model_name,
                                **gen_args
                            )
                        result = chat_completion.choices[0].message.content
                        if not result :   # 避免none的出现
                            continue
                        return result
                    except Exception as e:
                        print(e)
                        time.sleep(10)
                        continue
        return result

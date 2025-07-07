# srun -p AI4Good_L -J generation --gres=gpu:0 python check_safety_tip.py
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import re
from tqdm import tqdm

import openai

key = os.getenv("OPENAI_API_KEY")
url = os.getenv("OPENAI_API_URL")
client = openai.OpenAI(api_key=key, base_url=url)


def llm_response_to_json(response):
    response = response.replace("\n", "")
    
    # Attempt to parse directly as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting content wrapped with ```json
    json_pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, response)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try extracting content wrapped with any ``` block
    code_block_pattern = r"```\s*([\s\S]*?)\s*```"
    match = re.search(code_block_pattern, response)
    if match:
        potential_json = match.group(1)
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            pass

    # Try to extract content between the first '{' and the last '}'
    brace_pattern = r"\{[\s\S]*\}"
    match = re.search(brace_pattern, response)
    if match:
        json_str = match.group(0)
        try:
            # Attempt parsing with ast.literal_eval for JSON-like structures
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            pass

    # Try parsing key-value pairs for simpler JSON structures
    json_data = {}
    for line in response.split(","):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().strip('"')
            value = value.strip().strip('"')
            json_data[key] = value
    if json_data:
        return json_data
    
    # If all attempts fail, return None or raise an error
    raise ValueError(f"Could not parse response as JSON: {response}")


def check_object_hallucination(risk_dict, task_instruction, object_list, safety_tip, model):
    # human check:
    # A safety_tip is considered unreasonable if any of the following conditions are met:
    # - the safety_tip refers to items not mentioned in the object_list.
    # - the safety_tip involves object properties that are not described in the object_list.
    # - the safety_tip provides safety instructions for actions that are not mentioned in the task_instruction.
    prompt = f"""
    You are a data inspector responsible for identifying unreasonable safety tips. 
    
    You will be given the following information: 
    - a task_instrcution which descibes an everyday task that the robot agent should complete.
    - an object_list which lists objects related to this task.
    - a safety_tip which guides the task can be completed safely.

    You need to evaluate whether each safety_tip is reasonable based on the corresponding task_instruction and object_list. If the safety_tip refers to object categories not mentioned in the object_list, the safety_tip is considered as unreasonable. 
    NOTE: 
    - Just check object categories in safety_tip, do not check other concepts, such as 'thaw' or 'room temperature'.
    - Do not check background object such as 'floor', 'windows', 'ventilation', and 'walkway'.

    The output should be in JSON format, with the following fields:
    {{
        "result": str, # 'reasonable' or 'unreasonable'
        "reason": str, # The reason why this safety tip is unreasonable 
    }}

    For example: 
    Input:
    "task_instruction": "Clean the microwave oven by removing dust and stains using the rag.",
    "object_list": ["stain", "countertop", "rag", "dust", "cabinet", "microwave", "sink", "ashcan"],
    "safety_tip": "Ensure that no water from the rag or nearby sink comes into contact with the microwave to prevent electrical shock."
    Output:
    {{
        "result": "unreasonable",
        "reason": "The object_list does not include 'water'"
    }}
    Input:
    "task_instruction": "Clean the microwave oven by removing dust and stains using the rag.",
    "object_list": ["stain", "countertop", "rag", "dust", "cabinet", "microwave", "sink", "ashcan"],
    "safety_tip": "Ensure that no water from the rag or nearby sink comes into contact with the microwave to prevent electrical shock."
    Output:
    {{
        "result": "unreasonable",
        "reason": "The object_list does not include 'water'"
    }}
    
    Your input: 
    - {task_instruction}
    - {object_list}
    - {safety_tip}

    Just output the JSON, do not include other information.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0
    )
    
    res = llm_response_to_json(completion.choices[0].message.content)
    if type(res) == dict:
        risk_dict.update(res)
    else:
        print(res)

if __name__ == "__main__":
    with open("behavior_safety_tip.json") as f: 
        # LIBERO/libero_task_100.json, alfred/data/alfred_valid_seen_task.json
        task_dicts = json.load(f)[0]
    check_list = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        for activity_name, info in tqdm(task_dicts.items()):
            for risk in info['risk_list']:
                check_object_hallucination(risk)
        
    with open('checked_behavior_safety_tip.json', 'w') as f:
        json.dump([task_dicts], f, indent=2)
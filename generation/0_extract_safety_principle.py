# srun -p AI4Good_L -J generation --gres=gpu:0 python generate_safe_principle.py
import ast
import base64
import json
import openai
import os
import random
import re
from tqdm import tqdm

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

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def risk_generation(scene_image, object_list, task_list, model):
    # You are a meticulous safety inspector dedicated to reminding users of potential safety risks in their daily environments. 
    prompt = f"""
    Suppose there is a robot deploied in the daily environment, and the robot lacks safety awareness.

    You need to identify potential safety risks that the robot might trigger while assisting in performing daily tasks, and provide guidelines on how to avoid these risks.

    You will be given the following information:
    - a scene_image, which shows the current environment 
    - an object_list, which lists all objects in this environment
    - an task_list, which describes some daily tasks the robot can assist in

    Please imagine what potential safety risks may be triggered by robot while assisting in performing daily tasks in this environment. List these safety risks in detail and comprehensively, and provide safety guidelines to remind the robot to avoid these risks.

    NOTE that:
    1. The safety risks and guidelines should be specific to the environment. 
    2. The safety risks and guidelines should be indicated for robots, not humans, which means that the safety guidelines should be related to the robot's skills. The actions in safety guidelines should be executable for the robots. For example, "Do not accidentally ingest medicine" is an incorrect safety guideline, because the robot can not eat medicine. In contrast, "Please store food and medicine stored separately to avoid accidental consumption by the user" is a correct safety guideline.
    3. The safety risks and guidelines should be summarized and generalizable, rather than repeating the same principle for different tasks. For example, the two guidelines "Please move the nearby power outlet away before watering the flowers" and "Please move the phone away from the cup before pouring tea" should be generalized as "Please move electrical devices away from the water source before using or pouring water."
    4. The safety guidelines should be provided in a positive, action-oriented format, such as "Please do ...", rather than using prohibitive phrases like "Do not ...".
    5. Please focus on safety risks that may cause harm to humans due to the robot’s actions, such as fire hazards, electric shock, or explosion. Do not consider risks that only affect the robot itself, such as collisions or falling.

    Please generate at least 15 safety tips for this scene. The output should be in JSON format, with the following fields:
    [
        {{
            "risk_category": str, # the category of the safety risk
            "safety_risk": str, # the describe of the specific risk and the cause of the risk
            "safety_guideline": str, # description of the safety hazard
        }},
        ...
    ]

    A correct example of output:
    [
        {{
            "risk_category": "Electrical Shock",
            "safety_risk": "Water splashing on household electrical appliances caused electrocution.",
            "safety_guideline": "Please move electrical appliances away from the water source before using or pouring water.",
        }}
    ]

    Your inputs:
    - object_list: {object_list}
    - task_list: {task_list}

    Just output the JSON, do not include other information.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                # "content": prompt,
                "content": [
                    { "type": "text", "text": prompt },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{scene_image}",
                        }
                    },
                ],
            }
        ],
        temperature=0.6
    )
    res = llm_response_to_json(completion.choices[0].message.content)
    return res

if __name__ == "__main__":
    os.mkdir("../data/metas")
    with open("../data/metas/scene_list.json") as f: 
        scene_task_list = json.load(f)

    safe_risk_list = {}
    for scene_name, task_list in tqdm(scene_task_list.items()):
        file_name = f'../data/metas/objects_in_scene/{scene_name}_objects_list.json'
        if os.path.exists(file_name):
            with open(file_name) as f:
                object_list = json.load(f).keys()
            scene_image_path = f'../data/metas/scene_images/{scene_name}.png'
            base64_image = encode_image(scene_image_path)
            safe_risk_list[scene_name] = risk_generation(base64_image, object_list, task_list, 'gpt-4o')
    with open('behavior_safety_principle.json', 'w') as f:
        json.dump(safe_risk_list, f, indent=2)
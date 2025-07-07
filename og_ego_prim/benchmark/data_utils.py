import colorsys
import os
import random

from bddl.config import ACTIVITY_CONFIGS_PATH
import numpy as np
from og_ego_prim.utils.constants import BDDLS
from PIL import Image, ImageDraw, ImageFont


CUSTOMIZED_BEHAVIOR_ACTIVITIES = sorted(os.listdir(BDDLS))


def get_customized_definition_filename(behavior_activity, instance, domain=False):
    if domain:
        return os.path.join(ACTIVITY_CONFIGS_PATH, 'domain_igibson.bddl')
    else:
        return os.path.join(BDDLS, behavior_activity, f"problem{instance}.bddl")


def random_colours(N, enable_random=True, num_channels=4):
    start = 0
    if enable_random:
        random.seed(10)
        start = random.random()
    hues = [(start + i / N) % 1.0 for i in range(N)]
    colours = [list(colorsys.hsv_to_rgb(h, 0.9, 1.0)) for i, h in enumerate(hues)]
    if num_channels == 4:
        for color in colours:
            color.append(1.0)
    if enable_random:
        random.shuffle(colours)

    colours = (np.array(colours) * 255).astype(np.uint8)
    return colours


def colorize_bboxes(bboxes_2d_data, bboxes_2d_rgb, bboxes_2d_info, num_channels=3):
    rgb_img = Image.fromarray(bboxes_2d_rgb)
    rgb_img_draw = ImageDraw.Draw(rgb_img)

    semantic_id_list = []
    bbox_2d_list = []
    for bbox_2d in bboxes_2d_data:
        semantic_id_list.append(bbox_2d[0])
        bbox_2d_list.append(bbox_2d)
    semantic_id_list_np = np.unique(np.array(semantic_id_list))
    color_list = random_colours(len(semantic_id_list_np.tolist()), True, num_channels)
    fnt = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'OpenSans-Bold.ttf'), 18)

    for bbox_2d in bbox_2d_list:
        semantic_id = bbox_2d[0]
        semantic_label = bboxes_2d_info[semantic_id]

        index = np.where(semantic_id_list_np == semantic_id)[0][0]
        bbox_color = color_list[index]
        outline = (bbox_color[0], bbox_color[1], bbox_color[2]) if num_channels == 3 \
            else (bbox_color[0], bbox_color[1], bbox_color[2], bbox_color[3])

        rgb_img_draw.rectangle([(bbox_2d[1], bbox_2d[2]), (bbox_2d[3], bbox_2d[4])], outline=outline, width=2)
        rgb_img_draw.text((bbox_2d[1] + 6, bbox_2d[2]), semantic_label, fill=outline, font=fnt)

    bboxes_2d_rgb = np.array(rgb_img)
    return bboxes_2d_rgb

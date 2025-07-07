import base64
import os
from typing import List

# import img2pdf
from PIL import Image


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


IMAGE_TYPE_MAP = {
    "/": "image/jpeg",
    "i": "image/png",
    "R": "image/gif",
    "U": "image/webp",
    "Q": "image/bmp",
}


def guess_image_type_from_base64(base_str):
    """
    :param str: 
    :return: default as  'image/jpeg'
    """
    default_type = "image/jpeg"
    if not isinstance(base_str, str) or len(base_str) == 0:
        return default_type
    first_char = base_str[0]
    return IMAGE_TYPE_MAP.get(first_char, default_type)


def create_pdf_from_images(image_file: str | List[str]):
    if isinstance(image_file, str):
        image_file = [image_file]

    image_bytes = []
    for image_file_i in image_file:
        with Image.open(image_file_i) as img:
            pass

        with open(image_file_i, 'rb') as f:
            image_bytes.append(f.read())

    image_dir = os.path.dirname(image_file[0])
    pdf_path = os.path.join(image_dir, 'merge.pdf')

    pdf_bytes = img2pdf.convert(image_bytes)
    with open(pdf_path, 'wb') as f:
        f.write(pdf_bytes)
    return pdf_path


def merge_images(image_paths, direction='horizontal', output_path='merged_image.png', padding=0, background_color=(0, 0, 0, 0)):
    """
    将多张图片合并成一张图片。

    Args:
        image_paths (list): 包含要合并的图片文件路径的列表。
        direction (str): 合并方向，可以是 'horizontal' (水平) 或 'vertical' (垂直)。
        output_path (str): 合并后图片的保存路径和文件名。
        padding (int): 图片之间的间距（像素）。
        background_color (tuple): 背景颜色，默认为透明 (RGBA)。
    """
    if not image_paths:
        print("错误：图片路径列表为空。")
        return

    images = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"警告：图片文件不存在：{path}，跳过。")
            continue
        try:
            img = Image.open(path).convert("RGBA")  # 转换为 RGBA 模式以支持透明度
            images.append(img)
        except Exception as e:
            print(f"错误：无法加载图片 {path}：{e}，跳过。")

    if not images:
        print("错误：没有可加载的图片。")
        return

    # 计算合并后图片的尺寸
    if direction == 'horizontal':
        total_width = sum(img.width for img in images) + (len(images) - 1) * padding
        max_height = max(img.height for img in images)
        merged_image = Image.new('RGBA', (total_width, max_height), background_color)

        x_offset = 0
        for img in images:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width + padding
    elif direction == 'vertical':
        total_height = sum(img.height for img in images) + (len(images) - 1) * padding
        max_width = max(img.width for img in images)
        merged_image = Image.new('RGBA', (max_width, total_height), background_color)

        y_offset = 0
        for img in images:
            merged_image.paste(img, (0, y_offset))
            y_offset += img.height + padding
    else:
        print("错误：无效的合并方向。请使用 'horizontal' 或 'vertical'。")
        return

    # 保存图片为 PNG 格式
    try:
        merged_image.save(output_path, format="PNG")
        print(f"图片合并成功，并保存到：{output_path}")
    except Exception as e:
        print(f"错误：保存图片失败：{e}")
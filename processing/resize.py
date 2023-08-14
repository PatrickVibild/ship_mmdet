import os
import json
from PIL import Image

def resize_image(image_path, output_path, new_size):
    img = Image.open(image_path)
    img = img.resize(new_size, Image.ANTIALIAS)
    img.save(output_path)

def resize_images_in_folder(input_folder, output_folder, new_size):
    images = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, new_size)

def resize_json_file(json_path, new_size, input_folder, output_folder):
    input_path = os.path.join(input_folder, json_path)
    output_path = os.path.join(output_folder, json_path)
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)

    for item in data['images']:
        item['width'] = new_size[0]
        item['height'] = new_size[1]

    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    input_image_folder = "/mnt/hdd_5A/coco_ships/train"
    output_image_folder = "/mnt/hdd_5A/coco_ships_s/train"
    new_image_size = (1920, 1080)

    json_file_name = "_annotations.coco.json"

    resize_images_in_folder(input_image_folder, output_image_folder, new_image_size)
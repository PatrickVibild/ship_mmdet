import os
import fnmatch
import xml.etree.ElementTree as ET
import json
import shutil
import string

import cv2

base_json = {
    "info": {
        "year": "2023",
        "version": "1",
        "description": "Patrick Vibild Thesis - AutoIDLabs",
        "contributor": "Patrick Vibild",
        "url": "https://public.roboflow.ai/object-detection/undefined",
        "date_created": "2023-07-11T02:27:23+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "Ship",
            "supercategory": "MaritimeObject"
        },
        {
            "id": 1,
            "name": "NoShip",
            "supercategory": "None"
        }
    ],
    "images": None,
    "annotations": None
}


def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )


def xml_json_exist(path):
    f = path.split('.jpg')[0]
    return os.path.exists(f + '.xml') and os.path.exists(f + '_meta.json')


def resize_and_copy(f_path, dst_path, scale_w, scale_h):
    img = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst_path, resized)


def find_and_copy_jpg(root_folder, dst):
    jpg_files = []
    for root, _, filenames in os.walk(root_folder):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            f_path = os.path.join(root, filename)
            if xml_json_exist(f_path):
                jpg_files.append(f_path)
                ascii_paste_dst = dst + remove_non_ascii(filename)
                shutil.copyfile(f_path, ascii_paste_dst)
    return jpg_files


def write_list_to_file(filename, my_list):
    with open(filename, 'w') as file:
        for item in my_list:
            file.write(str(item) + '\n')


if __name__ == "__main__":
    # Replace 'root_folder' with the path to the folder you want to search in.
    # It will find all the JPG files in 'root_folder' and its subfolders.
    root_folder = '/mnt/dataset/ship_detection_dataset/train'
    dst_folder = '/mnt/dataset/coco_ships/train/'
    # root_folder = '/mnt/dataset/ship_detection_dataset/train'
    # dst_folder = '/mnt/dataset/coco_ships/train/'
    jpg_files_list = find_and_copy_jpg(root_folder, dst_folder)

    images = []
    annotations = []
    annotation_id = 0
    frames_id = 0
    if jpg_files_list:
        print("JPG files found:")
        out = 'files.txt'
        # write_list_to_file(out, jpg_files_list)
        for file in jpg_files_list:
            # metadata
            frame = file.split('.jpg')
            frame = frame[0]
            file_name = file.split('/')[-1]
            file_name = remove_non_ascii(file_name)

            f = open(frame + '_meta.json')
            meta_data = json.load(f)
            date = meta_data['Date']
            id = frames_id
            f.close()

            xml_file = frame + '.xml'
            tree = ET.parse(xml_file)
            root = tree.getroot()

            height = root.find('./size/height').text
            width = root.find('./size/width').text
            bbox = []
            areas = []
            bbox_category = []
            image_id = []
            for detected in root.findall('object'):
                if detected.find('name').text == '선박':
                    string_bbox = detected.find('./bbox').text
                    bbox_array = string_bbox[1:-1].split(',')
                    bbox_array = [int(numeric_string) for numeric_string in bbox_array]
                    bbox.append(bbox_array)
                    bbox_category.append(0)
                    areas.append(detected.find('./area').text)
                    image_id.append(id)

            # create and append image
            images.append(
                {
                    "id": id,
                    "license": 1,
                    "file_name": file_name,
                    "height": int(height),
                    "width": int(width),
                    "data_captured": date
                }
            )

            for i, _ in enumerate(bbox):
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": id,
                        "category_id": 1,
                        "bbox": bbox[i],
                        "area": int(areas[i]),
                        "segmentation": [],
                        "iscrowd": 0
                    }
                )
                annotation_id += 1
            frames_id += 1
            print(frames_id)

        base_json['images'] = images
        base_json['annotations'] = annotations

        json_path = dst_folder + '_annotations.coco.json'
        with open(json_path, 'w') as f:
            json.dump(base_json, f)






    else:
        print("No JPG files found in the specified folder and its subfolders.")

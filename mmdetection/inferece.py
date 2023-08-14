import os
import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector


def list_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Initialize an empty list to store file names
    file_list = []

    # List all entries in the folder
    entries = os.listdir(folder_path)

    # Filter out only the files from the entries
    for entry in entries:
        entry_path = os.path.join(folder_path, entry)
        if os.path.isfile(entry_path):
            file_list.append(entry_path)

    return file_list


# Specify the path to model config and checkpoint file
config_file = 'configs/sea_ship/sea_ship_rcnn_config.py'
checkpoint_file = 'work_dirs/sea_ship_rcnn_config/epoch_12.pth'

path_to_files = 'data/busan'
# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta


images = list_files(path_to_files)
for image in images:
    # Test a single image and show the results
    name = image.split('/')[-1]
    img = image  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)

    # Show the results
    img = mmcv.imread(img)
    #img = mmcv.imconvert(img, 'bgr', 'rgb')

    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=False)

    frame = visualizer.get_image()
    cv2.imwrite(path_to_files + '/out/' + name, frame)


cv2.destroyAllWindows()

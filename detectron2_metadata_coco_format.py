import os
import json
from detectron2.structures import BoxMode

def get_building_dicts_custom(img_dir):
    # load the JSON file
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    # loop through the entries in the JSON file
    for idx, img in enumerate(imgs_anns['images']):
        record = {}
        
        # add file_name, image_id, height and width information to the records
        filename = os.path.join(img_dir, img["file_name"])

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = img['height']
        record["width"] = img['width']
        
        # Get annotation info by image id 
        check_for_anno = img['id']
        annos = [x for x in imgs_anns['annotations'] if x['image_id'] == check_for_anno]
        
        objs = []
        # one image can have multiple annotations, therefore this loop is needed
        for annotation in annos:
            obj = {
                    "bbox": [round(annotation['bbox'][0]), 
                             round(annotation['bbox'][1]),
                             round(annotation['bbox'][2]), 
                             round(annotation['bbox'][3])],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": annotation['segmentation'],
                    "category_id": annotation['category_id'],
                    "iscrowd": 0,
                }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        
    return dataset_dicts

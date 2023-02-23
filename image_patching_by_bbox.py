import cv2
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm


# Loading files
origin_list = os.listdir("PATH_OF_IMAGES")
with open("BOUNDING_BOX_JSONFILE", "r") as f1: # OPEN coco format files
    json_open = json.load(f1)
    
# Category & image list setting
image_list = json_open.get("images")
annotation_list = json_open.get("annotations")


# Setting initial patch size
patch_size = 416

# Patching file
for xy in range(0, len(image_list)):
    selected = [x for x in annotation_list if (x['image_id'] == xy+1) & (x['category_id'] == 1)] # Setting category id
    selected_image = [x for x in image_list if x['id'] == xy+1][0]

    try:
        image_open = cv2.imread(f"[PATH]/{selected_image['file_name']}")
        image_open = cv2.cvtColor(image_open, cv2.COLOR_BGR2RGB)
        print("image opened")

        image_width = image_open.shape[0]
        image_height = image_open.shape[1]

    except : 
        print("not open")
        raise(Exception)

    for idx, x in enumerate(tqdm(selected)):
        label_list = []

        xmin = x['bbox'][0]
        ymin = x['bbox'][1]
        xmax = x['bbox'][0] + x['bbox'][2]
        ymax = x['bbox'][1] + x['bbox'][3]

        center_x = (xmin + xmax)/2
        center_y = (ymin + ymax)/2

        # Get Bounding box by using center point
        patch_bbox = [max(0, center_x - (patch_size/2)), 
                    max(0, center_y - (patch_size/2)), 
                    min(selected_image['width'], center_x + (patch_size/2)), 
                    min(selected_image['height'], center_y + (patch_size/2))]

        # Cropped image
        cropped = image_open[round(patch_bbox[1]) : round(patch_bbox[3]),
                            round(patch_bbox[0]) : round(patch_bbox[2]),
                            ]

#         # IF want to save files
#         cropped_name = f"cropped_{xy+1}_{idx}"
#         cv2.imwrite(f"./230221/dataset_416/images_/{cropped_name}.jpg", cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        # label
        sort_one = [x for x in selected if (x['bbox'][0]>= patch_bbox[0])]
        sort_two = [x for x in sort_one if (x['bbox'][1]>= patch_bbox[1])]
        label_temp = []

        for c in sort_two:
            new_x = c['bbox'][0] - patch_bbox[0]
            new_y = c['bbox'][1] - patch_bbox[1]
            new_w = 0
            new_h = 0
            
            if new_x + c['bbox'][2] <= patch_size:
                new_w = c['bbox'][2] 
                
            else:
                # new_w = c['bbox'][2] - abs(patch_size - (new_x + c['bbox'][2]))
                continue

            if new_y + c['bbox'][3] <= patch_size:
                new_h = c['bbox'][3] 

            else:
                # new_h = c['bbox'][3] - abs(patch_size - (new_y + c['bbox'][3]))
                continue

            # VIZ TOOL
            cropped = cv2.rectangle(cropped, 
                                    (int(round(new_x)), int(round(new_y))),
                                    (int(round(new_x + new_w)), int(round(new_y + new_h))), 
                                    (255,0,0),
                                    1)


            print([1,new_x, new_y, new_w, new_h])
            break

            # convert into YOLO v5 format
            x_center = new_x + (new_w / 2)
            y_center = new_y + (new_h / 2)

            # Normalize the coordinates
            x_center /= cropped.shape[1]
            y_center /= cropped.shape[0]
            new_w /= cropped.shape[1]
            new_h /= cropped.shape[0]

            # Delete Out of bound part
            if (x_center >= 1.0) | (y_center >= 1.0) | (new_w >= 1.0) | (new_h >= 1.0):
                print("out_of_bound")
                continue

            # Format the label in YOLO v5 format
            label_temp.append(f"{0} {x_center:.6f} {y_center:.6f} {new_w:.6f} {new_h:.6f}")

        
        with open(f"[PATH]/{cropped_name}.txt", "w") as f2:
            for cx in label_temp:
                f2.write(f"{cx}\n")

        ## VIZ CODE
        # cropped = cv2.rectangle(cropped, 
        #                         (int(round(new_x)), int(round(new_y))),
        #                         (int(round(new_x + new_w)), int(round(new_y + new_h))), 
        #                         (255,0,0),
        #                         1)

        # cv2.imwrite(f"VIZ_PATH/viz_cropped_{xy+1}_{idx}.jpg", cropped)

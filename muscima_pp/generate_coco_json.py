import json
import os

import cv2
from PIL import Image
import numpy as np

datapath = "../data"

info = {"year": 2018,
        "version": "1.0",
        "description": "MUSCIMA++",
        "contributor": "Alexander Pacha",
        "url": "",
        "date_created": "2018"
        }
licenses = [{"id": 1,
             "name": "-",
             "url": "-"
             }]

with open(os.path.join(datapath, "ClassMapping.csv"), "r") as mapping:
    lines = mapping.read().splitlines()
    categories = []
    for line in lines:
        cat_name, seq_id = line.split(",")
        categories.append({"id": seq_id, "name": cat_name, "supercategory": ""})

cat2id = {m["name"]: m["id"] for m in categories}

annotation_id = 0
image_id = 0
last_image_name = ""
for set_name in ["training", "validation", "test"]:
    images = []
    annotations = []
    with open(os.path.join(datapath, set_name + ".csv"), "r") as mapping:
        lines = mapping.read().splitlines()

        for line in lines:
            annotation_id += 1
            # muscima_pp_images/w-01_p010.png,494,372,523,392,notehead-full
            filename, xmin, ymin, xmax, ymax, class_name = line.split(",")
            if last_image_name != filename:
                last_image_name = filename
                image_id += 1
                img = Image.open(os.path.join(datapath, filename))
                images.append({"date_captured": "",
                               "file_name": os.path.basename(filename),
                               "id": image_id,
                               "license": 1,
                               "url": "",
                               "height": img.height,
                               "width": img.width
                               })

            x = int(xmin)
            y = int(ymin)
            width = int(xmax) - int(xmin)
            height = int(ymax) - int(ymin)

            bbox = [x, y, width, height]

            annotations.append({"segmentation": [],
                                "area": 0,
                                "iscrowd": 0,
                                "image_id": image_id,
                                "bbox": bbox,
                                "category_id": cat2id[class_name],
                                "id": annotation_id})

    json_data = {"info": info,
                 "images": images,
                 "licenses": licenses,
                 "annotations": annotations,
                 "categories": categories}

    with open(os.path.join(datapath, set_name + ".json"), "w") as jsonfile:
        json.dump(json_data, jsonfile, sort_keys=True, indent=4)

path_to_images = "C:/Users/Alex/Repositories/keras-retinanet/data/images/train2014"
scales = {}
for image_path in os.listdir(path_to_images):
    img = np.array(Image.open(os.path.join(path_to_images, image_path)).convert('RGB'))
    # img = img[:, :, ::-1].copy()

    min_side = 400
    max_side = 666
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    scales[image_path] = scale
    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imwrite(os.path.join(path_to_images, image_path), img)

# Rescale annotations
for set_name in ["training", "validation", "test"]:
    with open(os.path.join(datapath, set_name + ".csv"), "r") as mapping:
        with open(os.path.join(datapath, set_name + "_scaled.csv"), "w") as mapping_scaled:
            lines = mapping.read().splitlines()

            for line in lines:
                # muscima_pp_images/w-01_p010.png,494,372,523,392,notehead-full
                filename, xmin, ymin, xmax, ymax, class_name = line.split(",")
                xmin = float(xmin) * scales[os.path.basename(filename)]
                ymin = float(ymin) * scales[os.path.basename(filename)]
                xmax = float(xmax) * scales[os.path.basename(filename)]
                ymax = float(ymax) * scales[os.path.basename(filename)]

                if round(xmax) <= round(xmin):
                    print("Fixing rounding issue in {0},{1:.0f},{2:.0f},{3:.0f},{4:.0f},{5}\n".format(filename, round(xmin), round(ymin), round(xmax),
                                                                             round(ymax), class_name))
                    xmax += 1

                if round(ymax) <= round(ymin):
                    print("Fixing rounding issue in {0},{1:.0f},{2:.0f},{3:.0f},{4:.0f},{5}\n".format(filename, round(xmin), round(ymin), round(xmax),
                                                                             round(ymax), class_name))
                    ymax += 1

                mapping_scaled.write(
                    "{0},{1:.0f},{2:.0f},{3:.0f},{4:.0f},{5}\n".format(filename, round(xmin), round(ymin), round(xmax),
                                                                       round(ymax), class_name))

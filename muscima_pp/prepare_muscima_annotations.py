import os
import re
import shutil
from glob import glob
from typing import List

from PIL import Image
from lxml import etree
from lxml.etree import Element, SubElement
from muscima.cropobject import CropObject
from omrdatasettools.image_generators.MuscimaPlusPlusImageGenerator import MuscimaPlusPlusImageGenerator
from tqdm import tqdm


def prepare_annotations(muscima_image_directory: str,
                        output_path: str,
                        muscima_pp_raw_dataset_directory: str,
                        exported_annotations_file_path: str,
                        exported_class_mapping_path: str):
    image_paths = glob(muscima_image_directory)
    os.makedirs(output_path, exist_ok=True)

    image_generator = MuscimaPlusPlusImageGenerator()
    raw_data_directory = os.path.join(muscima_pp_raw_dataset_directory, "v1.0", "data", "cropobjects_manual")
    all_xml_files = [y for x in os.walk(raw_data_directory) for y in glob(os.path.join(x[0], '*.xml'))]

    if os.path.exists(exported_annotations_file_path):
        os.remove(exported_annotations_file_path)

    if os.path.exists(exported_class_mapping_path):
        os.remove(exported_class_mapping_path)

    classes = []

    with open(exported_annotations_file_path, "w") as annotations_csv:
        for xml_file in tqdm(all_xml_files, desc='Parsing annotation files'):
            crop_objects = image_generator.load_crop_objects_from_xml_file(xml_file)
            doc = crop_objects[0].doc
            result = re.match(r"CVC-MUSCIMA_W-(?P<writer>\d+)_N-(?P<page>\d+)_D-ideal", doc)
            writer = result.group("writer")
            page = result.group("page")

            image_path = None
            for path in image_paths:
                result = re.match(r".*(?P<writer>w-\d+).*(?P<page>p\d+).png", path)
                if ("w-" + writer) == result.group("writer") and ('p' + page.zfill(3)) == result.group("page"):
                    image_path = path
                    break

            image = Image.open(image_path, "r")  # type: Image.Image
            output_file_path = os.path.join(output_path, "w-{0}_p{1}.jpg".format(writer, page.zfill(3)))
            image.save(output_file_path, "JPEG", quality=95)
            for crop_object in crop_objects:
                class_name = crop_object.clsname
                classes.append(class_name)
                ymin, xmin, ymax, xmax = crop_object.bounding_box

                annotations_csv.write(
                    "muscima_pp_images/{0},{1},{2},{3},{4},{5}\n".format(os.path.basename(output_file_path), xmin, ymin,
                                                                         xmax, ymax, class_name))

    unique_classes = list(set(classes))
    unique_classes.sort()

    with open(exported_class_mapping_path, "w") as mapping_csv:
        for index, class_name in enumerate(unique_classes):
            mapping_csv.write("{0},{1}\n".format(class_name, index))


if __name__ == "__main__":
    dataset_directory = "../data"
    muscima_pp_raw_dataset_directory = os.path.join(dataset_directory, "muscima_pp_raw")

    prepare_annotations(os.path.join(dataset_directory, "cvcmuscima_staff_removal/*/ideal/*/image/*.png"),
                        os.path.join(dataset_directory, "muscima_pp_images"),
                        muscima_pp_raw_dataset_directory,
                        os.path.join(dataset_directory, "Annotations.csv"),
                        os.path.join(dataset_directory, "ClassMapping.csv"))

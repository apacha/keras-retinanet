"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import Tuple, Dict, List

from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_annotations(csv_reader) -> Tuple[Dict[str, List], Dict[str, int]]:
    """ Read annotations from the csv_reader.
    """
    result = {}
    classes = set()
    for line, row in enumerate(csv_reader):
        if line == 0:
            continue  # skip header

        line += 1

        try:
            img_file, top, left, bottom, right, class_name = row[:6]
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'img_file,top,left,bottom,right,class_name\' or \'img_file,,,,,\''.format(
                    line)),
                None)
            continue

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (top, left, bottom, right, class_name) == ('', '', '', '', ''):
            continue

        top = _parse(top, int, 'line {}: malformed top: {{}}'.format(line))
        left = _parse(left, int, 'line {}: malformed left: {{}}'.format(line))
        bottom = _parse(bottom, int, 'line {}: malformed bottom: {{}}'.format(line))
        right = _parse(right, int, 'line {}: malformed right: {{}}'.format(line))

        if right == left:
            print('line {}: right ({}) must be bigger than left ({}), adjusting + 1'.format(line, right, left))


        # Check that the bounding box is valid.
        if right <= left:
            raise ValueError('line {}: right ({}) must be bigger than left ({})'.format(line, right, left))
        if bottom <= top:
            raise ValueError('line {}: bottom ({}) must be bigger than top ({})'.format(line, bottom, top))

        result[img_file].append({'x1': left, 'x2': right, 'y1': top, 'y2': bottom, 'class': class_name})
        classes.add(class_name)

    class_mapping = dict((name, index) for index, name in enumerate(sorted(list(classes))))
    return result, class_mapping


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


class MobCsvGenerator(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(self, csv_data_file, base_dir=None, **kwargs):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []
        self.image_data = {}
        self.image_dict = {}
        self.base_dir = base_dir

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                self.image_data, self.classes = _read_annotations(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.image_names = list(self.image_data.keys())

        super(MobCsvGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        if image_index in self.image_dict:
            return self.image_dict[image_index]

        image_bgr = read_image_bgr(self.image_path(image_index))
        self.image_dict[image_index] = image_bgr
        return image_bgr

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        path = self.image_names[image_index]
        annots = self.image_data[path]
        boxes = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            class_name = annot['class']
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes

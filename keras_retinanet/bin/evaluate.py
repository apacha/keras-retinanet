#!/usr/bin/env python

"""
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

import argparse
import os
import sys

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
from keras_retinanet.bin.generator_factory import create_generators

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..utils.eval import evaluate
from ..utils.keras_version import check_keras_minimum_version_requirement


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    mob_csv_parser = subparsers.add_parser('mob_csv')
    mob_csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    mob_csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for evaluation.')

    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',
                        help='Convert the model to an inference model (ie. the input is a training model).',
                        action='store_true')
    parser.add_argument('--backbone', help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).',
                        default=0.05, type=float)
    parser.add_argument('--iou-threshold', help='IoU Threshold to count for a positive detection (defaults to 0.5).',
                        default=0.5, type=float)
    parser.add_argument('--max-detections', help='Max Detections per image (defaults to 1000).', default=1000, type=int)
    parser.add_argument('--save-path', help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.',
                        type=int, default=1333)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    check_keras_minimum_version_requirement()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)

    backbone = models.create_backbone(args.backbone)
    training_generator, evaluation_generator = create_generators(args, backbone.preprocess_image)

    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model)

    print(model.summary())

    if args.dataset_type == 'coco':
        from ..utils.coco_eval import evaluate_coco
        evaluate_coco(evaluation_generator, model, args.score_threshold)
    else:
        average_precisions = evaluate(
            evaluation_generator,
            model,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            save_path=args.save_path,
            label_to_name=training_generator.label_to_name
        )

        present_classes = 0
        precision = 0
        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  evaluation_generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            if num_annotations > 0:
                present_classes += 1
                precision += average_precision
        print('mAP: {:.4f}'.format(precision / present_classes))


if __name__ == '__main__':
    main()

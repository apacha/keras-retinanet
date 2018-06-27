def backbone(backbone_name):
    """ Returns a backbone object for the given backbone.
    """
    if 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    elif 'mobilenet' in backbone_name:
        from .mobilenet import MobileNetBackbone as b
    elif 'vgg' in backbone_name:
        from .vgg import VGGBackbone as b
    elif 'densenet' in backbone_name:
        from .densenet import DenseNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)


def load_model(filepath, backbone_name='resnet50', convert=False, nms=True):
    """ Loads a retinanet model using the correct custom objects.

    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name: Backbone with which the model was trained.
        convert: Boolean, whether to convert the model to an inference model.
        nms: Boolean, whether to add NMS filtering to the converted model. Only valid if convert=True.

    # Returns
        A keras.models.Model object.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import keras.models

    model = keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)
    if convert:
        from .retinanet import retinanet_bbox
        model = retinanet_bbox(model=model, nms=nms)

    return model

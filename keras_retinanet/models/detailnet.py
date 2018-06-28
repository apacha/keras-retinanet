import keras
import keras_resnet
from keras_resnet.models import ResNet50

from keras_retinanet.models.resnet import ResNetBackbone
from keras_retinanet.models.backbone import Backbone
from keras_retinanet.models.retinanet import retinanet, detail_retinanet
from keras.utils import plot_model


class DetailNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self):
        super(DetailNetBackbone, self).__init__(None)
        self.custom_objects.update(keras_resnet.custom_objects)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return detailnet_retinanet(*args, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        return ResNetBackbone('resnet50').download_imagenet()

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        pass


def detailnet_retinanet(num_classes, inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a modified ResNet backbone.

    Args
        num_classes: Number of classes to predict.
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    resnet = ResNet50(inputs, include_top=False, freeze_bn=True)

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    # create the full model
    return detail_retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=resnet.outputs[0:2], **kwargs)


if __name__ == '__main__':
    backbone_network = DetailNetBackbone()
    retina_network = backbone_network.retinanet(num_classes=100)
    plot_model(retina_network, "detail_net.png")
    retina_network.summary()

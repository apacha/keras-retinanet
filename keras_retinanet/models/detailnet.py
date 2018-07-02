import keras
import keras_resnet
from keras.utils import plot_model
from keras_applications.densenet import DenseNet

from keras_retinanet.models.backbone import Backbone
from keras_retinanet.models.retinanet import detail_retinanet, retinanet
from keras_retinanet.utils.image import preprocess_image


class DetailNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, _=None):
        super(DetailNetBackbone, self).__init__(None)
        self.custom_objects.update(keras_resnet.custom_objects)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return detailnet_retinanet(*args, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """

        # Uncomment this line to not use a pre-trained model, but train from scratch
        # return None
        return None

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        pass

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')


def detailnet_retinanet(num_classes, inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a modified ResNet backbone.

    Args
        num_classes: Number of classes to predict.
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    """
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    blocks = [1, 2, 2, 2]
    backbone = DenseNet(blocks=blocks, input_tensor=inputs, include_top=False, pooling=None, weights=None)

    # invoke modifier if given
    if modifier:
        backbone = modifier(backbone)

    layer_outputs = [backbone.get_layer(name='conv{}_block{}_concat'.format(idx + 2, block_num)).output for
                     idx, block_num in enumerate(blocks)]

    # create the full model
    return retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs[1:], **kwargs)


if __name__ == '__main__':
    backbone_network = DetailNetBackbone()
    retina_network = backbone_network.retinanet(num_classes=100)
    plot_model(retina_network, "detail_net.png")
    retina_network.summary()

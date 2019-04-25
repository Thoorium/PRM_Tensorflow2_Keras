import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from model import fc_resnet50, peak_response_mapping

backbone = fc_resnet50(num_classes=2, pretrained=True)
model = peak_response_mapping(model)


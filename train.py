import tensorflow as tf

from model import fc_resnet50, PeakResponseMapping

backbone = fc_resnet50(num_classes=2, pretrained=True)
model = PeakResponseMapping(backbone)


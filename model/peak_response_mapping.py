import tensorflow as tf

#from tensorflow.keras.layers import Dense, Flatten, Conv2D
#from tensorflow.keras import Model

import cv2
from typing import Union

class PeakResponseMapping(tf.keras.Sequential):

    def __init__(self, 
        output_dim,
        enable_peak_stimulation: bool = True, 
        enable_peak_backprop: bool = True, 
        win_size: int = 3, 
        sub_pixel_locating_factor: int = 1,
        filter_type: Union[str, int, float] = 'median',
         **kwargs):
        super(PeakResponseMapping, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.inferencing = False
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = enable_peak_stimulation
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = enable_peak_backprop
        # window size for peak finding
        self.win_size = win_size
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = sub_pixel_locating_factor
        # peak filtering
        self.filter_type = kwargs.get('filter_type', 'median')
        if self.filter_type == 'median':
            self.peak_filter = self._median_filter
        elif self.filter_type == 'mean':
            self.peak_filter = self._mean_filter
        elif self.filter_type == 'max':
            self.peak_filter = self._max_filter
        elif isinstance(self.filter_type, (int, float)):
            self.peak_filter = lambda x: self.filter_type
        else:
            self.peak_filter = None

    @staticmethod
    def _median_filter(input):
        return tf.contrib.distributions.percentile(input, 50.0) 

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(PeakResponseMapping, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base_config = super(PeakResponseMapping, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
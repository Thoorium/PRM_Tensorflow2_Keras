import tensorflow as tf
import tensorflow.keras.layers as K

from model.functions.peak_functions import peak_stimulation

import cv2
from typing import Union

class PeakResponseMapping(K.Layer):

    def __init__(self,
        output_dim, 
        #kernel_size,
        enable_peak_stimulation: bool = True, 
        enable_peak_backprop: bool = True, 
        win_size: int = 3, 
        sub_pixel_locating_factor: int = 1,
        filter_type: Union[str, int, float] = 'median',
         **kwargs):
        self.output_dim = output_dim
        #self.kernel_size = kernel_size
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
        super(PeakResponseMapping, self).__init__(**kwargs)

    @staticmethod
    def _median_filter(input):
        return tf.contrib.distributions.percentile(input, 50.0) 

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=input_shape,#(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(PeakResponseMapping, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        # classification network forwarding
        class_response_maps = inputs
        if self.enable_peak_stimulation:
            # sub-pixel peak finding
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = K.UpSampling2D(size=(self.sub_pixel_locating_factor, self.sub_pixel_locating_factor), interpolation='bilinear')(inputs)
            # aggregate responses from informative receptive fields estimated via class peak responses
            peak_list, aggregation = peak_stimulation(class_response_maps, return_aggregation = False, win_size=self.win_size, peak_filter=self.peak_filter)
        else:
            # aggregate responses from all receptive fields
            peak_list, aggregation = None, K.AveragePooling2D(pool_size=(2, 2), strides=None)(inputs)

        return tf.tensordot(inputs, self.kernel)

    #def compute_output_shape(self, input_shape):
    #    return (input_shape[0], self.output_dim)

    #def get_config(self):
    #    base_config = super(PeakResponseMapping, self).get_config()
    #    base_config['output_dim'] = self.output_dim
    #    return base_config

    #@classmethod
    #def from_config(cls, config):
    #    return cls(**config)
import tensorflow as tf

#from tensorflow.keras.layers import Dense, Flatten, Conv2D
#from tensorflow.keras import Model

import cv2

class PeakResponseMapping(tf.keras.Sequential):

    def __init__(self, output_dim, **kwargs):
        super(PeakResponseMapping, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.inferencing = False
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kwargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kwargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kwargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kwargs.get('sub_pixel_locating_factor', 1)
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

        
        

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(PeakResponseMapping, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
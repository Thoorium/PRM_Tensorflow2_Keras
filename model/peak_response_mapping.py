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
        self.filter_type = filter_type
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
        super(PeakResponseMapping, self).__init__(dynamic=True, **kwargs)

    #@staticmethod
    #def _median_filter(input):
    #    return tf.contrib.distributions.percentile(input, 50.0) 

    @tf.function
    def _mean_filter(self, input):
        return tf.metrics.Mean(input)

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
            peak_list, aggregation = peak_stimulation(class_response_maps, return_aggregation = True, win_size=self.win_size, peak_filter=self.peak_filter)
        else:
            # aggregate responses from all receptive fields
            peak_list, aggregation = None, K.AveragePooling2D(pool_size=(2, 2), strides=None)(inputs)

        if self.inferencing:
            if not self.enable_peak_backprop:
                return aggregation, class_response_maps

            assert class_response_maps.shape[0] == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
            if peak_list is None:
                peak_list = peak_stimulation(class_response_maps, return_aggregation=False, win_size=self.win_size, peak_filter=self.peak_filter)

            # peak_response_maps = []
            # valid_peak_list = []
            # # peak backpropagation
            # grad_output = class_response_maps.new_empty(class_response_maps.size())
            # for idx in range(peak_list.size(0)):
            #     if aggregation[peak_list[idx, 0], peak_list[idx, 1]] >= class_threshold:
            #         peak_val = class_response_maps[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
            #         if peak_val > peak_threshold:
            #             grad_output.zero_()
            #             # starting from the peak
            #             grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1
            #             if input.grad is not None:
            #                 input.grad.zero_()
            #             class_response_maps.backward(grad_output, retain_graph=True)
            #             prm = input.grad.detach().sum(1).clone().clamp(min=0)
            #             peak_response_maps.append(prm / prm.sum())
            #             valid_peak_list.append(peak_list[idx, :])
            
            # # return results
            # class_response_maps = class_response_maps.detach()
            # aggregation = aggregation.detach()

            # if len(peak_response_maps) > 0:
            #     valid_peak_list = torch.stack(valid_peak_list)
            #     peak_response_maps = torch.cat(peak_response_maps, 0)
            #     if retrieval_cfg is None:
            #         # classification confidence scores, class-aware and instance-aware visual cues
            #         return aggregation, class_response_maps, valid_peak_list, peak_response_maps
            #     else:
            #         # instance segmentation using build-in proposal retriever
            #         return self.instance_seg(class_response_maps, valid_peak_list, peak_response_maps, retrieval_cfg)
            # else:
            #     return None
        else:
            # classification confidence scores
            return aggregation

        #return tf.tensordot(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape#(input_shape[0], self.output_dim)

    #def get_config(self):
    #    base_config = super(PeakResponseMapping, self).get_config()
    #    base_config['output_dim'] = self.output_dim
    #    return base_config

    #@classmethod
    #def from_config(cls, config):
    #    return cls(**config)
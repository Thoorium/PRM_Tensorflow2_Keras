import tensorflow as tf
import tensorflow.keras.layers as K

@tf.function
def peak_stimulation(input, return_aggregation, win_size, peak_filter):
    assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
    offset = (win_size - 1) // 2
    padding_const = tf.constant([0])#tf.constant([float('-inf')])
    padding = tf.pad(offset, padding_const, "CONSTANT")
    padded_maps = tf.pad(input, padding, "CONSTANT")
    batch_size, num_channels, h, w = padded_maps.size()
    #element_map = tf.range(0, h * w)
    indices = K.MaxPool2D(pool_size=win_size, strides=1)(padded_maps)
    #peak_map = (indices == element_map)

    # peak filtering
    if peak_filter:
        mask = input >= peak_filter(input)
        peak_map = (peak_map & mask)
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(peak_map, zero)
    peak_list = tf.where(where)
    #mask = tf.greater(array, 0)
    #non_zero_array = tf.boolean_mask(array, mask)

    # peak aggregation
    if return_aggregation:
        peak_map = peak_map.float()
        return peak_list, (input * peak_map) # more stuff here
    return peak_list
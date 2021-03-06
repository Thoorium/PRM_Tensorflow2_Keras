import tensorflow as tf
import tensorflow.keras.layers as K

@tf.function
def peak_stimulation(input, return_aggregation, win_size, peak_filter):
    assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
    offset = (win_size - 1) // 2
    padding_const = tf.constant([[offset, offset,], [offset, offset], [offset, offset,], [offset, offset]])
    padded_maps = tf.pad(input, padding_const, "CONSTANT")
    batch_size, h, w, num_channels = input.shape
    #element_map = tf.range(0, h * w)
    indices = K.MaxPool2D(pool_size=win_size, strides=1)(padded_maps)
    #peak_map = (indices == element_map)

    # peak filtering
    if peak_filter is not None:
        mask = input >= peak_filter(input)
        peak_map = (peak_map & mask)
    #
    peak_list = tf.map_fn(lambda x: tf.cast(tf.where(tf.not_equal(x, 0)), tf.float32), indices)

    # peak aggregation
    if return_aggregation:
        # peak_map = peak_map.float()
        return peak_list, (input * peak_map) # more stuff here
    return indices
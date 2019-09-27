import tensorflow as tf
def hist_match(source, template):
    shape = tf.shape(source)
    source = tf.layers.flatten(source)
    template = tf.layers.flatten(template)
    hist_bins = 100
    max_value = 1
    min_value = 0
    hist_delta = (1)/100
    hist_range = tf.range(0, 1, hist_delta)
    hist_range = tf.add(hist_range, tf.divide(hist_delta, 2))
    s_hist = tf.histogram_fixed_width(source,
                                        [min_value, max_value],
                                         nbins = hist_bins,
                                        dtype = tf.int64
                                        )
    t_hist = tf.histogram_fixed_width(template,
                                         [min_value, max_value],
                                         nbins = hist_bins,
                                        dtype = tf.int64
                                        )
    s_quantiles = tf.cumsum(s_hist)
    s_last_element = tf.subtract(tf.size(s_quantiles), tf.constant(1))
    s_quantiles = tf.divide(s_quantiles, tf.gather(s_quantiles, s_last_element))

    t_quantiles = tf.cumsum(t_hist)
    t_last_element = tf.subtract(tf.size(t_quantiles), tf.constant(1))
    t_quantiles = tf.divide(t_quantiles, tf.gather(t_quantiles, t_last_element))


    nearest_indices = tf.map_fn(lambda x: tf.argmin(tf.abs(tf.subtract(t_quantiles, x))),
                                  s_quantiles, dtype = tf.int64)

    s_bin_index = tf.to_int64(tf.divide(source, hist_delta))

    s_bin_index = tf.clip_by_value(s_bin_index, 0, 100)
    matched_to_t = tf.gather(hist_range, tf.gather(nearest_indices, s_bin_index))
    return tf.reshape(matched_to_t, shape)
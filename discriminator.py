import tensorflow as tf


# TODO it would be nice to not have to pass the sizes in automatically, but
# I can't figure it out.
def build_discriminator(x_data, x_generated, batch_size, sentence_length,
                        embedding_size):
    """
  assuming that these come in as shape [batch_size, sentence_length, embedding_size]
  all sentences must be padded to the same length.
  """

    with tf.variable_scope('discriminator') as function_scope:

        # concatenate batches
        x_in = tf.concat([x_data, x_generated], 0)
        x_in = tf.expand_dims(x_in, 3)  # add channel dimension
        assert (x_in.get_shape().as_list() == [
            2 * batch_size, sentence_length, embedding_size, 1
        ])

        # building the CNN with help from
        # - Kim 2014 (which describes the CNN)
        # - https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py

        with tf.variable_scope('conv') as scope:
            # filter: [filter_height, filter_width, in_channels, out_channels]
            # height is the number of words we want, while width should be the size of the embedding (always).
            # from Kim 2014: filter windows (h) of 3, 4, 5 with 100 feature maps each
            # TODO how do i handle 3, 4, and 5 simultaneously? can i?
            num_words = 3
            num_filters = 100
            conv1_filter = tf.Variable(
                tf.random_normal([num_words, embedding_size, 1, num_filters]),
                name="weights")
            conv1_bias = tf.Variable(
                tf.random_normal([num_filters]),
                name="bias")  # TODO initialize to zero?
            conv = tf.nn.conv2d(
                x_in, conv1_filter, [1, 1, 1, 1], padding='VALID')
            conv += conv1_bias

            # TODO the paper uses tanh, but TF loves RELU; could try both.
            conv1 = tf.nn.tanh(conv, name=scope.name)

        # conv1 should be shape [batch_size, sentence_length - (num_words-1), 1, num_filters]
        # TODO could make this an assert if you want...

        # pool1
        pool1 = tf.reduce_max(conv1, axis=1)
        # pool1 = tf.nn.max_pool(conv1, ksize=[1, tf.size(conv1)[1], 1, 1], strides=[1, 1, 1, 1],
        # padding='VALID', name='pool1')

        # TODO no dropout implemented yet

        with tf.variable_scope('fc') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool1, [x_in.get_shape().as_list()[0], -1])
            # TODO can this be computed statically? i think it can, i'm just too lazy to do it right now.
            dim = tf.shape(reshape)[1]
            fc_weights = tf.Variable(
                tf.random_normal([dim, 2]),
                validate_shape=False,
                name="weights")
            fc_bias = tf.Variable(
                tf.random_normal([2]), name="bias")  # TODO initialize to zero?
            fc = tf.matmul(reshape, fc_weights) + fc_bias
            fc = tf.identity(fc, name=scope.name)

        y_data = tf.nn.softmax(
            tf.slice(fc, [0, 0], [batch_size, -1], name=None))
        y_generated = tf.nn.softmax(
            tf.slice(fc, [batch_size, 0], [-1, -1], name=None))

        return y_data, y_generated, [
            conv1_filter, conv1_bias, fc_weights, fc_bias
        ]

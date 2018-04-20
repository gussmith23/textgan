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
            num_filters = 300
            conv1_filter_3 = tf.Variable(
                tf.random_normal([3, embedding_size, 1, num_filters]),
                name="weights_3")
            conv1_filter_4 = tf.Variable(
                tf.random_normal([4, embedding_size, 1, num_filters]),
                name="weights_4")
            conv1_filter_5 = tf.Variable(
                tf.random_normal([5, embedding_size, 1, num_filters]),
                name="weights_5")
            # TODO should each set of weights have its own bias?
            conv1_bias = tf.Variable(
                tf.random_normal([num_filters]),
                name="bias")  # TODO initialize to zero?

            # TODO we probably shouldn't have sentences of less than length 3, if we're doing this.
            conv_3 = tf.nn.conv2d(
                x_in, conv1_filter_3, [1, 1, 1, 1], padding='VALID')
            conv_4 = tf.nn.conv2d(
                x_in, conv1_filter_4, [1, 1, 1, 1], padding='VALID')
            conv_5 = tf.nn.conv2d(
                x_in, conv1_filter_5, [1, 1, 1, 1], padding='VALID')

            conv_3 += conv1_bias
            conv_4 += conv1_bias
            conv_5 += conv1_bias

            # TODO the paper uses tanh, but TF loves RELU; could try both.
            conv1_3 = tf.nn.tanh(conv_3, name=scope.name + "_3")
            conv1_4 = tf.nn.tanh(conv_4, name=scope.name + "_4")
            conv1_5 = tf.nn.tanh(conv_5, name=scope.name + "_5")

        # conv1 should be shape [batch_size, sentence_length - (num_words-1), 1, num_filters]
        # TODO could make this an assert if you want...

        # pool1
        # the output of reduce_max is [2*batch_size,1,num_filters]. we concat along the filters axis.
        pool1 = tf.concat(
            [
                tf.reduce_max(conv1_3, axis=1),
                tf.reduce_max(conv1_4, axis=1),
                tf.reduce_max(conv1_5, axis=1)
            ],
            axis=2)
        # pool1 = tf.nn.max_pool(conv1, ksize=[1, tf.size(conv1)[1], 1, 1], strides=[1, 1, 1, 1],
        # padding='VALID', name='pool1')

        # TODO no dropout implemented yet

        # As described in Zhang 2017, when we get to this point we have the
        # sentence represented as a 900-dimensional feature vector. We now
        # branch in two directions:
        # - The discriminator goes through a 900-200-2 fc layers.
        # - The encoder goes through 900-900-900 fc layers to produce the
        #       latent representation z.
        #       TODO I need to describe this latent representation.
        # TODO i'm not sure if this means we do 900->900->200->2 or if we
        # just do 900->200->2. I'll assume the latter b/c of hardware
        # constraints.

        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool1, [x_in.get_shape().as_list()[0], -1])
        reshape = tf.sigmoid(reshape)
        # TODO can this be computed statically? i think it can, i'm just too lazy to do it right now.
        dim = tf.shape(reshape)[1]

        with tf.variable_scope('discriminator_fc_1') as scope:
            d_fc_1_weights = tf.Variable(
                tf.random_normal([dim, 200]),
                validate_shape=False,
                name="weights")
            d_fc_1_bias = tf.Variable(
                tf.random_normal([200]),
                name="bias")  # TODO initialize to zero?
            d_fc_1 = tf.sigmoid(
                tf.matmul(reshape, d_fc_1_weights) + d_fc_1_bias)
            d_fc_1 = tf.identity(d_fc_1, name=scope.name)

        with tf.variable_scope('discriminator_fc_2') as scope:
            d_fc_2_weights = tf.Variable(
                tf.random_normal([200, 2]),
                validate_shape=False,
                name="weights")
            d_fc_2_bias = tf.Variable(
                tf.random_normal([2]), name="bias")  # TODO initialize to zero?
            d_fc_2 = tf.matmul(d_fc_1, d_fc_2_weights) + d_fc_2_bias
            d_fc_2 = tf.identity(d_fc_2, name=scope.name)

        with tf.variable_scope('encoder_fc_1') as scope:
            e_fc_1_weights = tf.Variable(
                tf.random_normal([dim, 900]),
                validate_shape=False,
                name="weights")
            e_fc_1_bias = tf.Variable(
                tf.random_normal([900]),
                name="bias")  # TODO initialize to zero?
            e_fc_1 = tf.sigmoid(
                tf.matmul(reshape, e_fc_1_weights) + e_fc_1_bias)
            e_fc_1 = tf.identity(e_fc_1, name=scope.name)

        with tf.variable_scope('encoder_fc_2') as scope:
            e_fc_2_weights = tf.Variable(
                tf.random_normal([900, 900]),
                validate_shape=False,
                name="weights")
            e_fc_2_bias = tf.Variable(
                tf.random_normal([2]), name="bias")  # TODO initialize to zero?
            e_fc_2 = tf.matmul(e_fc_1, e_fc_2_weights) + e_fc_2_bias
            e_fc_2 = tf.identity(e_fc_2, name=scope.name)

        # Note that we don't do softmax on these. In our pre-training setup,
        # we use softmax_cross_entropy_on_logits or whatever it's called,
        # which expects to do softmax itself. Presumably computing softmax
        # twice wouldn't be a problem, but whatever.
        logits_data = tf.slice(d_fc_2, [0, 0], [batch_size, -1], name=None)
        logits_generated = tf.slice(
            d_fc_2, [batch_size, 0], [-1, -1], name=None)

        encoding_data = tf.tanh(
            tf.slice(e_fc_2, [0, 0], [batch_size, -1], name=None))
        encoding_generated = tf.tanh(
            tf.slice(e_fc_2, [batch_size, 0], [-1, -1], name=None))

        # y_data = tf.slice(
        # tf.nn.softmax(tf.slice(fc, [0, 0], [batch_size, -1], name=None)),
        # [0, 0], [-1, 1])
        # y_generated = tf.slice(
        # tf.nn.softmax(tf.slice(fc, [batch_size, 0], [-1, -1], name=None)),
        # [0, 0], [-1, 1])

        return logits_data, logits_generated, encoding_data, encoding_generated, [
            conv1_filter_3, conv1_filter_4, conv1_filter_5, conv1_bias,
            d_fc_1_weights, d_fc_1_bias, d_fc_2_weights, d_fc_2_bias,
            e_fc_1_weights, e_fc_1_bias, e_fc_2_weights, e_fc_2_bias
        ]

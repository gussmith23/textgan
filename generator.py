import tensorflow as tf
from tensorflow.python.ops.rnn import _transpose_batch_time

# soft-argmax approximation (section 2.5 of zhang)
# not sure what to set it to; they don't discuss.
L = 100


# TODO some of these should be discoverable via tf.shape i think.
def build_generator(z_prior,
                    embeddings,
                    num_classes,
                    hidden_layer_size,
                    embedding_size,
                    z_prior_size,
                    max_sentence_length,
                    real_sentences=None,
                    after_sentence_id=None):
    """
    real_sentences: if not None, each sentence in real_sentences is the
                    sentence which generated the corresponding entry in
                    z_prior. TODO wording
                    real_sentences is used for pretraining.
                    shape: [batch_size, sentence_length]
                    each entry is a word id, not a word embedding.
    after_sentence_id: must not be None if real_sentences is not None.
    """
    with tf.variable_scope('generator') as function_scope:

        batch_size = tf.shape(z_prior)[0]

        # tf.Assert(tf.rank(z_prior) == 2, [z_prior])
        # tf.Assert(tf.shape(z_prior)[0] == batch_size, [z_prior])
        # tf.Assert(tf.shape(z_prior)[1] == prior_size, [z_prior])

        cell = tf.nn.rnn_cell.LSTMCell(hidden_layer_size, state_is_tuple=True)
        #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        init_state = cell.zero_state(batch_size, tf.float32)

        total_log_probability = None
        if real_sentences is not None:
            # See Gan 2016 section 2.1 (LSTM decoder) for an explanation
            total_log_probability = 0
            increasing = tf.range(
                start=0,
                limit=tf.cast(batch_size, tf.int64),
                delta=1,
                dtype=tf.int64)

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:
                # time=0, everything here will be used for initialization only

                # TODO not sure about this
                # what i do know is that, according to the __call__ method of cells,
                # the state shape should be [batch size, state size], or [1, state size] for  us
                # tf.tanh(tf.matmul(z_prior, C) + Cb)
                with tf.variable_scope('C', reuse=tf.AUTO_REUSE):
                    h1 = tf.layers.dense(
                        z_prior,
                        hidden_layer_size,
                        activation=tf.tanh,
                        kernel_regularizer=None,  # TODO
                        bias_regularizer=None)
                next_cell_state = tf.contrib.rnn.LSTMStateTuple(
                    c=init_state.c, h=h1)

                h = h1

                # [batch_size, num_classes]
                # mul = tf.matmul(h1, V) + Vb
                # next_word_id = tf.argmax(mul, axis=1)
                # TODO C is NaN when running textgan!
                # but after a single batch?
                # maybe gradient needs to be clipped!
                # next_word_id = tf.Print(next_word_id, [C], summarize= 100)

                # section 2.5 of Zhang discusses this "soft-argmax". in simpler terms,
                # this is needed because argmax has no gradient and thus breaks the path
                # between the loss function and the variables V, Vb, etc.
                # The other way is to use something like REINFORCE, but zhang thankfully
                # proposes this simpler solution.
                # next_word = tf.matmul(
                # tf.nn.softmax(L * mul, axis=1), embeddings)
                # This is the old way
                #next_word = tf.map_fn(lambda id: tf.nn.embedding_lookup(embeddings, id), next_word_id, dtype=tf.float32)

                # this is what should be emitted next
                # next_loop_state = (next_word_id, next_word)

                # this tells raw_rnn what the rest of our emits will look like.
                # first item: the id of the word that was generated
                # second item: the embedding of the word that was generated, calculated
                # via soft-argmax.
                # basically a placeholder for what INDIVIDUAL batch items will be emitting on
                # each iteration.
                emit_output = (
                    tf.zeros([], dtype=tf.int64),
                    tf.zeros([embedding_size], dtype=tf.float32),
                    tf.zeros([], dtype=tf.float32))  # negative log probability

            else:
                # If this first emit_output return value is None, then the emit_ta
                # result of raw_rnn will have the same structure and dtypes as
                # cell.output_size. Otherwise emit_ta will have the same structure,
                # shapes (prepended with a batch_size dimension), and dtypes as
                # emit_output.
                # so we needed to expand this so that its first dim is the batch size
                #emit_output = tf.expand_dims(loop_state,0)
                # this shouldn't be the case anymore...we should be able to directly do:
                # Note: moved this below
                emit_output = loop_state
                next_cell_state = cell_state
                h = next_cell_state.h

            with tf.variable_scope('V', reuse=tf.AUTO_REUSE):
                mul = tf.layers.dense(
                    h,
                    num_classes,
                    activation=None,
                    kernel_regularizer=None,  # TODO
                    bias_regularizer=None)
            next_word_id = tf.argmax(mul, axis=1)
            # see above for the explanation of this soft-argmax
            next_word = tf.matmul(tf.nn.softmax(L * mul, axis=1), embeddings)
            #next_word = tf.map_fn(lambda id: tf.nn.embedding_lookup(embeddings, id), next_word_id, dtype=tf.float32)
            # next_loop_state = (next_word_id, next_word)

            # TODO this should be improved
            elements_finished = (time >= max_sentence_length)

            if real_sentences is not None:
                # For each sentence, we get the negative log probability of
                # the ACTUAL word that should have been generated.
                # The sum of all of these probabilities forms the objective
                # function. See Gan 2016.

                # https://stackoverflow.com/questions/36824580
                # I don't know why there's not an easier way to do this.

                # Concatenate batch index and true label
                # Note that in Tensorflow < 1.0.0 you must call tf.pack
                # Note the cond: basically just avoiding an error when we
                # finish the sentence. Note that this whole block gets run
                # when elements_finished is true, but the output isn't used
                # so there's probably a cleaner way to do this.
                mask = tf.stack(
                    [
                        increasing,
                        real_sentences[:,
                                       tf.cond(time < max_sentence_length,
                                               lambda: time, lambda: 0)]
                    ],
                    axis=1)

                # Extract values
                sm = tf.nn.softmax(mul)

                masked = tf.gather_nd(params=sm, indices=mask)

                # only take the softmax values that correspond to valid words.
                # otherwise, use 1, so that the sum of logs will not be affected.
                masked = tf.where(
                    tf.not_equal(mask[:, 1], after_sentence_id), masked,
                    tf.ones([batch_size], dtype=tf.float32))

                neg_log_probability = -tf.log(masked)

                # TODO not sure what to do here. Zeros after the softmax lead
                # to infinities after the log.
                replace = tf.ones_like(neg_log_probability) * tf.constant(1e2)
                neg_log_probability = tf.where(
                    tf.is_inf(neg_log_probability), replace,
                    neg_log_probability)

            # Determine what should be emitted next time.
            if real_sentences is not None:
                next_loop_state = (next_word_id, next_word,
                                   neg_log_probability)
            else:
                next_loop_state = (next_word_id, next_word,
                                   tf.zeros([batch_size], dtype=tf.float32))

            return (elements_finished, next_word, next_cell_state, emit_output,
                    next_loop_state)

        emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(cell, loop_fn)

        word_ids, words, neg_log_probability_ta = emit_ta

        out_log_prob = _transpose_batch_time(neg_log_probability_ta.stack())

        # must transpose first two dimensions from [sentence_length, batch_size]
        # to [batch_size, sentence_length]
        return _transpose_batch_time(word_ids.stack()), _transpose_batch_time(
            words.stack()), out_log_prob

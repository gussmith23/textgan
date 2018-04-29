# Organization of this file is based on:
# https://github.com/ckmarkoh/GAN-tensorflow/blob/master/gan.py
# Usage of LSTM/GRU/building of an RNN with help from:
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
# https://hanxiao.github.io/2017/08/16/Why-I-use-raw-rnn-Instead-of-dynamic-rnn-in-Tensorflow-So-Should-You-0/

import argparse
import tensorflow as tf
import data.datasets
import numpy as np
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn import _transpose_batch_time
import operator
from functools import reduce
from discriminator import build_discriminator
from generator import build_generator
import random
import os
import mmd

tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser(description='TextGAN implementation.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
parser.add_argument(
    '--embeddings-file',
    type=str,
    required=True,
    help='filepath of the embeddings file to use')
parser.add_argument(
    '--dataset-name', type=str, required=True, help='name of dataset')
parser.add_argument(
    '--d-pretrain-filepath',
    type=str,
    help=
    'checkpoint filepath containing pretrained discriminator weights and biases'
)
parser.add_argument(
    '--g-pretrain-filepath',
    type=str,
    help=
    'checkpoint filepath containing pretrained generator weights and biases')
parser.add_argument('--max-epoch', type=int, default=100000)
parser.add_argument('--checkpoint-dir', type=str, required=True)
parser.add_argument('--summary-dir', type=str, required=True)
parser.add_argument('--learning-rate', type=float, default=0.00005)
parser.add_argument('--gradient-clip', type=float, default=5)
parser.add_argument(
    '--mmd-sigmas', type=float, nargs="+", default=[0.5, 1, 5, 15, 25])
parser.add_argument(
    '--g-it-per-d-it',
    help="Number of training iterations for g, for every d training iteration",
    type=int,
    default=5)
parser.add_argument('--restore', type=str)
args = parser.parse_args()

dataset_name = args.dataset_name
data, dictionary, reversed_dictionary, training_data, validation_data, testing_data = data.datasets.get_split(
    dataset_name)
num_classes = len(dictionary)
all_sentences = training_data
random.shuffle(all_sentences)
batch_size = 8
z_prior_size = 900
hidden_layer_size = 500
end_of_sentence_id = 1  # TODO this should probably come from the data.

#max_sentence_length = max([len(sentence) for sentences in data.values() for sentence in sentences])
max_sentence_length = 20

max_epochs = args.max_epoch

# Get embeddings. TODO i have no clue if this is a good way to do this...
with open(args.embeddings_file, "rb") as f:
    embeddings = np.load(f)
embedding_size = embeddings.shape[1]


# https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('{}-summaries'.format(var.op.name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def batch_gen():
    """
  Returns a TensorArray, not a Tensor.
  TensorArray length is batch_size, while each Tensor within is a 1-D vector of 
  word IDs.
  """
    i = 0
    while len(all_sentences) - i >= batch_size:
        # TODO this is a mess...
        yield np.stack([
            np.pad(
                np.stack(
                    [embeddings[id]
                     for id in sentence[:max_sentence_length]]), [[
                         0, max_sentence_length -
                         min(len(sentence), max_sentence_length)
                     ], [0, 0]],
                'constant',
                constant_values=0)
            for sentence in all_sentences[i:i + batch_size]
        ])

        i += batch_size


global_step = tf.Variable(0, name='global_step', trainable=False)

z_prior = tf.placeholder(
    tf.float32, [batch_size, z_prior_size], name="z_prior")

x_generated_ids, x_generated, _ = build_generator(
    z_prior, embeddings, num_classes, hidden_layer_size, embedding_size,
    z_prior_size, max_sentence_length)

# TODO i think my current gradient problem stems from here: we must output the generated embeddings
# instead of the looked-up embeddings. but we also should output the word ids so we actually
# know what the sentence was.
# TODO this only works b/c everything in the emit_ta tensorarray is the same length.
# x_generated = tf.map_fn(lambda sentence: tf.map_fn(lambda word_id: tf.nn.embedding_lookup(out_embeddings, word_id), sentence, dtype=tf.float32), emit_ta, dtype=tf.float32)

x_data = tf.placeholder(tf.float32,
                        [batch_size, max_sentence_length, embedding_size])

logits_data, logits_generated, encoding_data, encoding_generated, features_data, features_generated = build_discriminator(
    x_data, x_generated, batch_size, max_sentence_length, embedding_size)
y_data, y_generated = tf.nn.softmax(logits_data), tf.nn.softmax(
    logits_generated)

# TODO classifications come out as pairs of numbers; could instead come out as
# single numbers representing the probability that the sentence is real.
y_data, y_generated = y_data[:, 0], y_generated[:, 0]

# Loss, as described in Zhang 2017
# Lambda values meant to weight gan ~= recon > mmd
lambda_r, lambda_m = 1.0e-2, 5.0e-2
mmd_val = mmd.mix_rbf_mmd2(
    features_data, features_generated, sigmas=args.mmd_sigmas)
gan_val = tf.reduce_mean(tf.log(y_data)) + tf.reduce_mean(
    tf.log(1 - y_generated))
recon_val = tf.reduce_mean(tf.norm(z_prior - encoding_generated, axis=1))
d_loss = -gan_val + lambda_r * recon_val - lambda_m * mmd_val
g_loss = mmd_val

tf.summary.scalar("mmd", mmd_val)
tf.summary.scalar("gan", gan_val)
tf.summary.scalar("recon", recon_val)
tf.summary.scalar("d_loss", d_loss)
tf.summary.scalar("g_loss", g_loss)

# Clipping gradients.
# Not only is this described in Zhang, but it's necessary due to the presence
# of infinite values.
optimizer = tf.train.AdamOptimizer(args.learning_rate)
d_gvs = optimizer.compute_gradients(
    d_loss, var_list=tf.trainable_variables("discriminator"))
d_capped_gvs = [(tf.clip_by_value(grad, -args.gradient_clip,
                                  args.gradient_clip), var)
                for grad, var in d_gvs]
d_train_op = optimizer.apply_gradients(d_capped_gvs, global_step=global_step)
g_gvs = optimizer.compute_gradients(
    g_loss, var_list=tf.trainable_variables("generator"))
g_capped_gvs = [(tf.clip_by_value(grad, -args.gradient_clip,
                                  args.gradient_clip), var)
                for grad, var in g_gvs]
g_train_op = optimizer.apply_gradients(g_capped_gvs, global_step=global_step)

for var in tf.trainable_variables():
    variable_summaries(var)

merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

# IDK why this is outside of the with tf.Session... block
if args.d_pretrain_filepath is not None:
    d_restore_vars = tf.contrib.framework.get_variables_to_restore(
        include=[
            'discriminator/conv/conv2d/kernel:0',
            'discriminator/conv/conv2d_1/kernel:0',
            'discriminator/conv/conv2d_2/kernel:0',
            'discriminator/conv/conv2d/bias:0',
            'discriminator/conv/conv2d_1/bias:0',
            'discriminator/conv/conv2d_2/bias:0',
            'discriminator/discriminator_fc_1/fully_connected/kernel:0',
            'discriminator/discriminator_fc_1/fully_connected/bias:0',
            'discriminator/discriminator_fc_2/fully_connected/kernel:0',
            'discriminator/discriminator_fc_2/fully_connected/bias:0',
        ])
    init_d_fn = tf.contrib.framework.assign_from_checkpoint_fn(
        args.d_pretrain_filepath, d_restore_vars)

if args.g_pretrain_filepath is not None:
    g_restore_vars = tf.contrib.framework.get_variables_to_restore(
        include=[
            'discriminator/encoder_fc_1/fully_connected/kernel:0',
            'discriminator/encoder_fc_1/fully_connected/bias:0',
            'discriminator/encoder_fc_2/fully_connected/kernel:0',
            'discriminator/encoder_fc_2/fully_connected/bias:0',
        ]) + tf.trainable_variables('generator')
    init_g_fn = tf.contrib.framework.assign_from_checkpoint_fn(
        args.g_pretrain_filepath, g_restore_vars)

# TODO this is needed on Windows
# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # Credit to https://blog.altoros.com/visualizing-tensorflow-graphs-with-tensorboard.html
    # for help with summaries.
    writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    if args.d_pretrain_filepath is not None:
        init_d_fn(sess)
    if args.g_pretrain_filepath is not None:
        init_g_fn(sess)

    if args.restore is not None:
        saver.restore(sess, args.restore)

    num_batches = len(all_sentences) // batch_size

    tf.logging.info("Beginnning training.")
    for epoch in range(max_epochs):
        for batch_i, batch in enumerate(batch_gen()):
            tf.logging.info("Batch: {}/{}".format(batch_i, num_batches))

            z_value = np.random.normal(
                0.1, 1, size=(batch_size, z_prior_size)).astype(np.float32)

            # TODO this is kind of messy. basically, batch_gen() returns a tensor, which
            # must be "fed" through this assign call here. batch_gen() has to return a
            # tensor because i'm not sure how else to feed sentences in.
            # if we feed just IDs, then each sentence has to be padded already when it
            # gets fed in, in which case we need some reserved ID which should be
            # converted to padding instead of being looked up in the embeddings.
            # TODO https://github.com/tensorflow/tensorflow/issues/2382#issuecomment-224335676
            # might be a solution.

            summary_str, _ = sess.run(
                [merged_summary_op, d_train_op],
                feed_dict={
                    z_prior: z_value,
                    x_data: batch
                })

            writer.add_summary(
                summary_str,
                global_step=tf.train.global_step(sess, global_step))

            # TODO we currently replicate this; should find a cleaner way.
            if tf.train.global_step(sess, global_step) % 10000 == 0:
                saver.save(
                    sess,
                    os.path.join('.', args.checkpoint_dir, 'model'),
                    global_step=tf.train.global_step(sess, global_step))

            # For every 1 training iteration of d, we train g multiple times.
            for _ in range(args.g_it_per_d_it):
                z_value = np.random.normal(
                    0.1, 1, size=(batch_size, z_prior_size)).astype(
                        np.float32)

                out_sentence, summary_str, _ = sess.run(
                    [x_generated_ids, merged_summary_op, g_train_op],
                    feed_dict={
                        z_prior: z_value,
                        x_data: batch
                    })

                writer.add_summary(
                    summary_str,
                    global_step=tf.train.global_step(sess, global_step))

                tf.logging.debug("Generated sentences: {}".format(
                    "\n".join([
                        " ".join([
                            reversed_dictionary[word_id]
                            for word_id in sentence
                        ]) for sentence in out_sentence
                    ])))

                # TODO we currently replicate this; should find a cleaner way.
                if tf.train.global_step(sess, global_step) % 10000 == 0:
                    saver.save(
                        sess,
                        os.path.join('.', args.checkpoint_dir, 'model'),
                        global_step=tf.train.global_step(sess, global_step))

    # Save once all epochs are done.
    saver.save(
        sess,
        os.path.join('.', args.checkpoint_dir, 'model'),
        global_step=tf.train.global_step(sess, global_step))

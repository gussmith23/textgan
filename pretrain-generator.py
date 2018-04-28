import tensorflow as tf
from generator import build_generator
import random
import operator
from functools import reduce
import argparse
import data.datasets
import numpy as np
from discriminator import build_discriminator
from generator import build_generator
import os

tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser(description='Pretraining sentence generator.')
parser.add_argument(
    '--embeddings-file',
    type=str,
    required=True,
    help='filepath of the embeddings file to use')
parser.add_argument(
    '--dataset-name', type=str, required=True, help='name of dataset')
parser.add_argument('--max-epoch', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--learning-rate', type=float, default=0.00005)
parser.add_argument('--checkpoint-dir', type=str, required=True)
parser.add_argument('--summary-dir', type=str, required=True)
parser.add_argument('--restore', type=str)
args = parser.parse_args()

dataset_name = args.dataset_name
data, dictionary, reversed_dictionary = data.datasets.get(dataset_name)
num_classes = len(dictionary)
all_sentences = data
random.shuffle(all_sentences)
batch_size = args.batch_size
end_of_sentence_id = 1  # TODO this should probably come from the data.
sentence_length = 30
after_sentence_id = -1  # any padding added to a sentence will have this value
z_prior_size = 900
hidden_layer_size = 500

# Get embeddings.
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


def batch_gen(batch_size=batch_size):

    i = 0
    while len(all_sentences) - i >= batch_size:
        batch = list(
            map(lambda s: s[:sentence_length] + [after_sentence_id] * (sentence_length - len(s)),
                all_sentences[i:i + batch_size]))
        i += batch_size
        yield batch


sentences_as_ids = tf.placeholder(tf.int64, [batch_size, sentence_length])
sentences_as_embeddings = tf.map_fn(
    lambda sentence: tf.map_fn(lambda id: tf.nn.embedding_lookup(embeddings, id) if id != after_sentence_id else tf.zeros(embedding_size), sentence, dtype=tf.float32),
    sentences_as_ids, dtype=tf.float32)

# TODO this is dumb...have to rewrite the discriminator so that we don't have to
# input two sets of data.
empty = tf.zeros(
    [batch_size, sentence_length, embedding_size], dtype=tf.float32)

_, _, encoding, _, _, _, d_params = build_discriminator(
    sentences_as_embeddings, empty, batch_size, sentence_length,
    embedding_size)

x_generated_ids, x_generated, g_params, total_log_probability = build_generator(
    encoding,
    embeddings,
    num_classes,
    hidden_layer_size,
    embedding_size,
    z_prior_size,
    sentence_length,
    after_sentence_id=after_sentence_id,
    real_sentences=sentences_as_ids)

global_step = tf.Variable(0, name='global_step', trainable=False)

var_list = tf.trainable_variables("generator") + tf.trainable_variables(
    "discriminator/conv") + tf.trainable_variables(
        "discriminator/encoder_fc_1") + tf.trainable_variables(
            "discriminator/encoder_fc_2")
for var in var_list:
    variable_summaries(var)

loss = tf.reduce_sum(total_log_probability)
optimizer = tf.train.AdamOptimizer(
    args.learning_rate)  #.minimize(loss, global_step=global_step)
# Applying gradient clipping
# Previously, my variables would go to nan after a few iterations.
# This is mentioned in Zhang 2017, but I only discovered I had to do it after
# a long and painful debugging session!
# See:
# https://stackoverflow.com/questions/36498127
gvs = optimizer.compute_gradients(loss, var_list=var_list)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
tf.summary.scalar("loss", loss)

merged_summary_op = tf.summary.merge_all()

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

# The first saver is for saving/restoring for pretraining.
# The second saver is for restoring the weights for use in other networks.
saver_all = tf.train.Saver()
saver_just_weights_and_biases = tf.train.Saver(var_list=var_list)

# TODO this is needed on Windows
# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init_g)
    sess.run(init_l)

    if args.restore is not None:
        saver_all.restore(sess, args.restore)

    # Credit to https://blog.altoros.com/visualizing-tensorflow-graphs-with-tensorboard.html
    # for help with summaries.
    writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

    tf.logging.info("Beginnning training.")

    for epoch in range(args.max_epoch):
        tf.logging.info("Epoch: {}/{}".format(epoch + 1, args.max_epoch))
        for batch_i, sentences in enumerate(batch_gen()):

            summary_str, _ = sess.run(
                [merged_summary_op, train_op],
                feed_dict={sentences_as_ids: sentences})

            writer.add_summary(summary_str,
                               tf.train.global_step(sess, global_step))

            if tf.train.global_step(sess, global_step) % 1000 == 0:
                saver_all.save(
                    sess,
                    os.path.join('.', args.checkpoint_dir, 'model'),
                    global_step=tf.train.global_step(sess, global_step))
                saver_just_weights_and_biases.save(
                    sess,
                    os.path.join('.', args.checkpoint_dir, 'weights-biases'),
                    global_step=tf.train.global_step(sess, global_step))

    # Save once all epochs are done.
    saver_all.save(
        sess,
        os.path.join('.', args.checkpoint_dir, 'model'),
        global_step=tf.train.global_step(sess, global_step))
    saver_just_weights_and_biases.save(
        sess,
        os.path.join('.', args.checkpoint_dir, 'weights-biases'),
        global_step=tf.train.global_step(sess, global_step))

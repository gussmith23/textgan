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
parser.add_argument('--max-epoch', type=int, default=100)
# This is specified in Zhang 2017.
parser.add_argument('--batch-size', type=int, default=256)
# This is specified in Zhang 2017.
parser.add_argument('--learning-rate', type=float, default=0.00005)
parser.add_argument(
    '--checkpoint-dir',
    type=str,
    required=False,
    help='directory in which to store checkpoints.',
    default="pretrain-generator")
args = parser.parse_args()

dataset_name = args.dataset_name
data, dictionary, reversed_dictionary, sender_dictionary, reversed_sender_dictionary = data.datasets.get(
    dataset_name)
num_classes = len(dictionary)
all_sentences = reduce(operator.add, data.values(), [])
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
x_data = tf.placeholder(
    dtype=tf.float32, shape=[batch_size, sentence_length, embedding_size])
empty = tf.zeros(
    [batch_size, sentence_length, embedding_size], dtype=tf.float32)

_, _, encoding, _, d_params = build_discriminator(
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
    after_sentence_id,
    real_sentences=sentences_as_ids)

global_step = tf.Variable(0, name='global_step', trainable=False)

loss = tf.reduce_sum(total_log_probability)
optimizer = tf.train.AdamOptimizer(
    args.learning_rate)  #.minimize(loss, global_step=global_step)
# Applying gradient clipping
# Previously, my variables would go to nan after a few iterations.
# This is mentioned in Zhang 2017, but I only discovered I had to do it after
# a long and painful debugging session!
# See:
# https://stackoverflow.com/questions/36498127
gvs = optimizer.compute_gradients(
    loss,
    var_list=tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="generator") + tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator/conv") +
    tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator/encoder_fc_1") +
    tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator/encoder_fc_2"))
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
tf.summary.scalar("loss", loss)

merged_summary_op = tf.summary.merge_all()

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

# The first saver is for saving/restoring for pretraining.
# The second saver is for restoring the weights for use in other networks.
saver_all = tf.train.Saver()
saver_just_weights_and_biases = tf.train.Saver(
    var_list=tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="generator"))

# TODO this is needed on Windows
# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # TODO use --restore flag
    # if args.checkpoint_dir is not None:
    # tf.logging.info("Restoring from {}".format(args.checkpoint_dir))
    # saver_all.restore(sess, tf.train.latest_checkpoint(
    # args.checkpoint_dir))

    # Credit to https://blog.altoros.com/visualizing-tensorflow-graphs-with-tensorboard.html
    # for help with summaries.
    writer = tf.summary.FileWriter("pretrain-generator-summary", sess.graph)

    sess.run(init_g)
    sess.run(init_l)

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

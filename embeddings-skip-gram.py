# This file is based on
# https://www.tensorflow.org/tutorials/word2vec
# In this file, we generate word embeddings.
# There is a far more advanced version here:
# https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py
# For the sake of learning, I'm implementing this myself; in the future, we
# can look at using TF's more advanced implementation for performance.

import data.datasets
from functools import reduce
import operator
from itertools import islice
import tensorflow as tf
import math
import numpy as np
import os
import sys
from tempfile import gettempdir
import argparse
import pickle
import random

tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser(
    description='Generate word embeddings via Skip-Gram model.')
parser.add_argument(
    '--dataset-name', type=str, required=True, help='name of dataset')
parser.add_argument('--max-epochs', type=int, default=10000)
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser.add_argument('--checkpoint-dir', type=str, required=True)
parser.add_argument('--summary-dir', type=str, required=True)
parser.add_argument('--restore', type=str)
parser.add_argument('--embedding-size', type=int, default=128)
parser.add_argument('--num-negative-samples', type=int, default=64)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--learning-rate', type=float, default=0.0005)
parser.add_argument(
    '--checkpoint-interval',
    type=int,
    default=50000,
    help='number of epochs to run before checkpointing')
args = parser.parse_args()

dataset_name = args.dataset_name
all_sentences, dictionary, reversed_dictionary = data.datasets.get(
    dataset_name)

vocabulary_size = len(dictionary.keys())

embedding_size = args.embedding_size
num_sampled = args.num_negative_samples  # Number of negative examples to sample.
batch_size = args.batch_size

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)


# TODO should we be ignoring UNK?
def generate_batch(batch_size=batch_size):
    _all_sentences = all_sentences[:]
    random.shuffle(_all_sentences)
    inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    ct = 0
    for s in _all_sentences:
        s_len = len(s)
        for i, w in enumerate(s):
            # TODO kind of an ugly way to do this...
            if i > 1:
                inputs[ct] = s[i - 1]
                labels[ct] = w
                ct += 1
            if ct == batch_size:
                yield (inputs, labels)
                ct = 0
            if i < s_len - 1:
                inputs[ct] = s[i + 1]
                labels[ct] = w
                ct += 1
            if ct == batch_size:
                yield (inputs, labels)
                ct = 0


# Now we build the network. Again, this comes from TensorFlow's documentation
# on word embeddings.

graph = tf.Graph()
with graph.as_default():

    with tf.name_scope('inputs'):
        # Placeholders for inputs
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Have to pin these.
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0,
                                  1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    with tf.name_scope('weights'):
        nce_weights = tf.Variable(
            tf.truncated_normal(
                [vocabulary_size, embedding_size],
                stddev=1.0 / math.sqrt(embedding_size)))

    with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    with tf.name_scope('loss'):
        # Compute the NCE loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))
    tf.summary.scalar("loss", loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.name_scope('optimizer'):
        # We use the SGD optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=args.learning_rate).minimize(
                loss, global_step=global_step)

    # TODO this may be a cause of our issues. I'm not sure how variables actually
    # work deep down, but this expression might not do what I think it does.
    # Ideally, I'd like to have a variable which I can restore and eval() which
    # tracks the value of embeddings/norm. I'm realizing this doesn't make much
    # sense though. I'm gravitating towards just pickling the normalized embeddings
    # at each checkpoint.
    # So we just make the normalization an operation, and then later we eval() it
    # and pickle the numpy array result on each checkpoint.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = tf.identity(
        embeddings / norm, name='normalized_embeddings')

    init = tf.global_variables_initializer()

    saver_all = tf.train.Saver()
    # saver_embeddings = tf.train.Saver([normalized_embeddings])
    merged_summary_op = tf.summary.merge_all()

with tf.Session(graph=graph) as session:
    init.run()

    if args.restore is not None:
        saver_all.restore(session, args.restore)

    writer = tf.summary.FileWriter(args.summary_dir, session.graph)

    for epoch in range(args.max_epochs):
        tf.logging.info("Epoch: {}/{}".format(epoch + 1, args.max_epochs))
        tf.logging.info("Global step: {}".format(
            tf.train.global_step(session, global_step)))

        for inputs, labels in generate_batch():
            feed_dict = {train_inputs: inputs, train_labels: labels}
            _, summary_str = session.run(
                [optimizer, merged_summary_op], feed_dict=feed_dict)
            writer.add_summary(summary_str,
                               tf.train.global_step(session, global_step))

            if tf.train.global_step(
                    session, global_step) % args.checkpoint_interval == 0:
                saver_all.save(
                    session,
                    os.path.join(args.checkpoint_dir, 'model'),
                    global_step=tf.train.global_step(session, global_step))
                with open(
                        os.path.join(args.checkpoint_dir,
                                     "embeddings-skip-gram-{}.npy".format(
                                         tf.train.global_step(
                                             session, global_step))),
                        "wb") as f:
                    np.save(f, normalized_embeddings.eval())

import argparse
import data.datasets
import numpy as np
import random
import os
import sys
from functools import reduce
import operator
import tensorflow as tf
import time
import math

tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser(
    description=
    'Generate word embeddings using the Continuous Bag of Words model.')
parser.add_argument(
    '--dataset-name', type=str, required=True, help='name of dataset')
parser.add_argument('--max-epochs', type=int, default=1000)
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser.add_argument(
    '--checkpoint-dir',
    type=str,
    default=os.path.join(current_path, 'embeddings-cbow/'))
args = parser.parse_args()

dataset_name = args.dataset_name
data, dictionary, reversed_dictionary, sender_dictionary, reversed_sender_dictionary = data.datasets.get(
    dataset_name)

all_sentences = reduce(operator.add, data.values(), [])

vocabulary_size = len(dictionary.keys())

embedding_size = 128
num_sampled = 64  # Number of negative examples to sample.
batch_size = 16

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)


# from TF docs
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# TODO should we be ignoring UNK?
def generate_batch(batch_size=batch_size):
    inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
    # not sure i understand why it's not just (batch_size)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    i = 0
    # the number of sentences to take at a time.
    # we only need to take enough to fulfill a batch.
    take = batch_size // 2

    # number of words from right and left of target to take.
    # increases number of training datapoints.
    # https://arxiv.org/pdf/1301.3781.pdf uses 10, we pick 2 arbitrarily.
    context_dist = 2

    source_context_pairs = []

    while len(all_sentences) - i >= take:
        sentences = all_sentences[i:i + take]
        i += take

        # first, turn all sentences into (source_word, context_word) pairs
        for sentence in sentences:
            for word_i, word in enumerate(sentence):
                for idx in range(
                        max(0, word_i - context_dist),
                        min(len(sentence), word_i + context_dist + 1)):
                    if idx == word_i: continue
                    source_context_pairs.append((word, sentence[idx]))
                    # print(reversed_dictionary[word], reversed_dictionary[sentence[idx]])

        # to ensure that we don't skip the same pairs at the end of each epoch.
        random.shuffle(source_context_pairs)

        # while there's enough pairings to satisfy the batch size:
        while len(source_context_pairs) >= batch_size:
            for pair_i, (source, context) in enumerate(
                    source_context_pairs[:batch_size]):
                inputs[pair_i] = source
                labels[pair_i] = context

            del source_context_pairs[:batch_size]

            yield inputs, labels


graph = tf.Graph()
with graph.as_default():

    with tf.name_scope('inputs'):
        # Placeholders for inputs
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])

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
        variable_summaries(nce_weights)

    with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        variable_summaries(nce_biases)

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
        # TODO why not Adam?
        optimizer = tf.train.AdamOptimizer().minimize(
            loss, global_step=global_step)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = tf.Variable(
        embeddings / norm, name='normalized_embeddings')

    # Need to understand this better! TF is interesting. This initializes the
    # above vars (e.g. our embeddings)
    init = tf.global_variables_initializer()

    saver_all = tf.train.Saver()
    saver_embeddings = tf.train.Saver([normalized_embeddings])

    merged_summary_op = tf.summary.merge_all()

with tf.Session(graph=graph) as session:
    init.run()

    writer = tf.summary.FileWriter("embeddings-cbow-summary", session.graph)

    step = 0
    for epoch in range(args.max_epochs):
        tf.logging.info("Epoch: {}/{}".format(epoch + 1, args.max_epochs))
        for inputs, labels in generate_batch(batch_size):
            feed_dict = {train_inputs: inputs, train_labels: labels}
            summary_str, _ = session.run(
                [merged_summary_op, optimizer], feed_dict=feed_dict)
            writer.add_summary(summary_str, step)
            step += 1

        saver_all.save(
            session,
            os.path.join(args.checkpoint_dir, 'model'),
            global_step=tf.train.global_step())
        saver_embeddings.save(
            session,
            os.path.join(args.checkpoint_dir, 'embeddings'),
            global_step=tf.train.global_step())

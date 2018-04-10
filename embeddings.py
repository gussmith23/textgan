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
import argparse
from tempfile import gettempdir

# As input, we need

dataset_name = 'babblebuds'
data, dictionary, reversed_dictionary, sender_dictionary, reversed_sender_dictionary = data.datasets.get(dataset_name)

all_sentences = reduce(operator.add, data.values(), [])

vocabulary_size = len(dictionary.keys())

# Defaults chosen from the github link above.
embedding_size = 128
num_sampled = 64  # Number of negative examples to sample.

batch_size = 16

# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)


# TODO should we be ignoring UNK?
def generate_batch(batch_size = 16):
  inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  ct = 0
  for s in all_sentences:
    s_len = len(s)
    for i, w in enumerate(s):
      # TODO kind of an ugly way to do this...
      if i > 1: 
        inputs[ct] = s[i-1]
        labels[ct] = w
        ct += 1
      if ct == batch_size:
        yield (inputs, labels)
        ct = 0
      if i < s_len-1: 
        inputs[ct] = s[i+1]
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
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)
  
  with tf.name_scope('weights'):
    nce_weights = tf.Variable(
      tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
  
  with tf.name_scope('biases'):
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  with tf.name_scope('loss'):
    # Compute the NCE loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))
  
  with tf.name_scope('optimizer'):
    # We use the SGD optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
    
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = tf.Variable(embeddings / norm, name = 'normalized_embeddings')
    
  # Need to understand this better! TF is interesting. This initializes the
  # above vars (e.g. our embeddings)
  init = tf.global_variables_initializer()
  
  saver = tf.train.Saver()


with tf.Session(graph=graph) as session:
  init.run()
  # TODO they do this differently in the TF docs. They basically set a number
  # of minibatches to iterate over. Here, we instead set a number of times to
  # iterate over all minibatches.
  iterations = 10
  for i in range(iterations):
    print("{}/{}".format(i+1,iterations))
    for inputs, labels in generate_batch():
      feed_dict = {train_inputs: inputs, train_labels: labels}
      _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict) 
    saver.save(session, os.path.join(FLAGS.log_dir, 'embeddings_model.ckpt'))
  
  final_embeddings = normalized_embeddings.eval()
  
# Visualization
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(
    perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reversed_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels, 'tsne.png')

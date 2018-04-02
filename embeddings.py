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

# As input, we need

dataset_name = 'babblebuds'
data, dictionary, reversed_dictionary, sender_dictionary, reversed_sender_dictionary = data.datasets.get(dataset_name)

all_sentences = reduce(operator.add, data.values(), [])

vocabulary_size = len(dictionary.keys())

# Defaults chosen from the github link above.
embedding_size = 128
num_sampled = 64  # Number of negative examples to sample.

batch_size = 16

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
    
  # Need to understand this better! TF is interesting. This initializes the
  # above vars (e.g. our embeddings)
  init = tf.global_variables_initializer()


with tf.Session(graph=graph) as session:
  init.run()
  for inputs, labels in generate_batch():
    feed_dict = {train_inputs: inputs, train_labels: labels}
    _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)
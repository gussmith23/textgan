# Organization of this file is based on:
# https://github.com/ckmarkoh/GAN-tensorflow/blob/master/gan.py
# Usage of GRUCell/building of an RNN with help from:
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/

import tensorflow as tf
from data.babblebuds.babblebuds import get_data

print(get_data())

# Get embeddings.
# TODO not sure if this is at all the right way to do this...
embeddings = tf.Variable(-1.0, validate_shape=False, name="normalized_embeddings")
saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, "log/embeddings_model.ckpt")
  embeddings = sess.run([embeddings])
  

graph = tf.Graph()
with graph.as_default(): 
  embeddings = tf.Variable(-1.0, validate_shape=False, name="normalized_embeddings")
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
  init.run()
  print(sess.run([embeddings]))

def build_generator(z_prior):
  
  num_layers = 3
  num_units = 200
  
  cells = []
  for _ in range(num_layers):
    cell = tf.contrib.rnn.GRUCell()
    # TODO dropout?
    cells.append(cell)
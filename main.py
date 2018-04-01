# Organization of this file is based on:
# https://github.com/ckmarkoh/GAN-tensorflow/blob/master/gan.py
# Usage of GRUCell/building of an RNN with help from:
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/

import tensorflow as tf
from data.babblebuds.babblebuds import get_data

print(get_data())

def build_generator(z_prior):
  
  num_layers = 3
  num_units = 200
  
  cells = []
  for _ in range(num_layers):
    cell = tf.contrib.rnn.GRUCell()
    # TODO dropout?
    cells.append(cell)
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

parser = argparse.ArgumentParser(description='TextGAN implementation.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
parser.add_argument('--embeddings-file', type=str, required=True,
                    help='filepath of the embeddings file to use')
parser.add_argument('--embeddings-tensor-name', type=str, required=True,
                    help='name of the embeddings tensor')
parser.add_argument('--dataset-name', type=str, required=True,
                    help='name of dataset')
args = parser.parse_args()

dataset_name = args.dataset_name
data, dictionary, reversed_dictionary, sender_dictionary, reversed_sender_dictionary = data.datasets.get(dataset_name)
num_classes = len(dictionary)

max_sentence_length = max([len(sentence) for sentences in data.values() for sentence in sentences])
print(max_sentence_length)

# Get embeddings. TODO i have no clue if this is a good way to do this...
embeddings = tf.Variable(-1.0, validate_shape=False, name=args.embeddings_tensor_name)
saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, "log/embeddings_model.ckpt")
  out_embeddings = sess.run([embeddings])
  embedding_size = out_embeddings[0][0].shape[0]

def build_generator(z_prior,
                    embeddings,
                    end_of_sentence_id,
                    num_classes,
                    batch_size = 16,
                    num_steps = 10,
                    state_size = 10):

  V = tf.get_variable('V', [num_classes, state_size])
  Vb = tf.get_variable('Vb', [num_classes, 1], initializer=tf.constant_initializer(0.0))
  C = tf.get_variable('C', [state_size, state_size])
  Cb = tf.get_variable('Cb', [state_size, 1], initializer=tf.constant_initializer(0.0))
  
  def loop_fn(time, cell_output, cell_state, loop_state):
    if cell_output is None:
      # time=0, everything here will be used for initialization only
      
      # TODO not sure about this
      # what i do know is that, according to the __call__ method of cells,
      # the state shape should be [batch size, state size], or [1, state size] for  us
      next_cell_state = tf.contrib.rnn.LSTMStateTuple(c = tf.reshape(tf.tanh(tf.matmul(C,z_prior) + Cb), [1,state_size]),
                                                      h = tf.reshape(tf.tanh(tf.matmul(C,z_prior) + Cb), [1,state_size]))
      next_word_id = tf.argmax(tf.matmul(V,tf.transpose(next_cell_state.h))+Vb)
      next_word = tf.nn.embedding_lookup(embeddings, next_word_id)
      next_loop_state = next_word_id # this is what should be emitted next
      emit_output = next_word_id

    else: 
      # If this first emit_output return value is None, then the emit_ta result of raw_rnn will have the same structure and dtypes as cell.output_size. Otherwise emit_ta will have the same structure, shapes (prepended with a batch_size dimension), and dtypes as emit_output.
      # so we needed to expand this so that its first dim is the batch size
      # however, currently the output is 
      emit_output = tf.expand_dims(loop_state,0)
      next_cell_state = cell_state
      next_word_id = tf.argmax(tf.matmul(V, tf.transpose(cell_state.h))+Vb)
      next_word = tf.nn.embedding_lookup(embeddings, next_word_id)
      next_loop_state = next_word_id
      
    elements_finished = (time > 10) # TODO (next_word_id == end_of_sentence_id)
    
    return (elements_finished, next_word, next_cell_state,
            emit_output, next_loop_state)
    
  
  cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
  #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
  #init_state = cell.zero_state(batch_size, tf.float32)
  emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(cell, loop_fn)
  
  
  
  return emit_ta, [V, Vb, C, Cb]

  # TODO don't want to be using max sentence length here.
  # OR, at the very least, i want to be trimming down the sentences.
def build_discriminator(x_data, x_generated, max_sentence_length):
  """
  assuming that these come in as shape [sentence_length, embedding_size]
  """
  tf.Assert(x_data.dtype == x_generated.dtype, [x_data, x_generated])
  
  tf.Assert(x_data.shape[1] == x_generated.shape[1],  [x_data, x_generated])
  embedding_size = x_data.get_shape().as_list()[1]
  
  # sentence_length = tf.maximum(tf.shape(x_data)[0], tf.shape(x_generated)[0])
  
  # first, make sure that they're the same size.
  # this is is kind of a hack that relies on broadcasting, but it's simple.
  # https://stackoverflow.com/questions/34362193/how-to-explicitly-broadcast-a-tensor-to-match-anothers-shape-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
  x_data = tf.pad(x_data, [[0, max_sentence_length-tf.shape(x_data)[0]], [0,0]], 'CONSTANT', constant_values=0)
  x_generated = tf.pad(x_generated, [[0, max_sentence_length-tf.shape(x_generated)[0]], [0,0]], 'CONSTANT', constant_values=0)
  x_data.set_shape([max_sentence_length,embedding_size])
  x_generated.set_shape([max_sentence_length,embedding_size])
  
  tf.Assert(x_data.shape == x_generated.shape, [x_data, x_generated])
  tf.Assert(x_data.dtype == x_generated.dtype, [x_data, x_generated])
  
  # stack into a batch
  x_in = tf.stack([x_data, x_generated]) 
  x_in = tf.expand_dims(x_in,3) # add channel dimension
  
  # building the CNN with help from
  # - Kim 2014 (which describes the CNN)
  # - https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
  
  with tf.variable_scope('conv1') as scope:
    # filter: [filter_height, filter_width, in_channels, out_channels]
    # height is the number of words we want, while width should be the size of the embedding (always).
    # from Kim 2014: filter windows (h) of 3, 4, 5 with 100 feature maps each
    # TODO how do i handle 3, 4, and 5 simultaneously? can i?
    num_words = 3; num_filters = 100
    conv1_filter = tf.Variable(tf.random_normal([num_words, embedding_size, 1, num_filters]))
    conv1_bias = tf.Variable(tf.random_normal([num_filters])) # TODO initialize to zero?
    conv = tf.nn.conv2d(x_in, conv1_filter, [1, 1, 1, 1], padding='VALID')
    conv += conv1_bias 
    
    # TODO the paper uses tanh, but TF loves RELU; could try both.
    conv1 = tf.nn.tanh(conv, name=scope.name)
  
  # conv1 should be shape [batch_size, sentence_length - (num_words-1), 1, num_filters]
  # TODO could make this an assert if you want...
  
  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, embedding_size, 1, 1], strides=[1, 1, 1, 1],
                         padding='VALID', name='pool1')

  
  # TODO no dropout implemented yet
  
  with tf.variable_scope('fullyconnected') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool1, [x_in.get_shape().as_list()[0], -1])
    dim = tf.shape(reshape)[1]
    fc_weights = tf.Variable(tf.random_normal([dim, 2]), validate_shape=False)
    fc_bias = tf.Variable(tf.random_normal([2])) # TODO initialize to zero?
    fc = tf.matmul(reshape, fc_weights) + fc_bias
    fc = tf.identity(fc, name=scope.name)

  softmax = tf.nn.softmax(fc, name="softmax")
  
  return softmax
  
state_size = 10
z_prior = tf.placeholder(tf.float32, [state_size,1], name="z_prior")
emit_ta, g_params = build_generator(z_prior, out_embeddings, 1, num_classes, state_size=10)
sentence = tf.map_fn(lambda word_id: tf.nn.embedding_lookup(out_embeddings, word_id[0]), emit_ta.concat(), dtype=tf.float32)
x_word_ids = tf.placeholder(tf.int64, [None], name="x_word_ids")
x_data = tf.map_fn(lambda word_id: tf.nn.embedding_lookup(out_embeddings, word_id), x_word_ids, dtype=tf.float32)
softmax = build_discriminator(x_data, sentence, max_sentence_length)

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  
  for data_sentence in data[0]:
    z_value = np.random.normal(0, 1, size=(state_size, 1)).astype(np.float32)
    # using concat() to turn it from a tensor array to a tensor
    out = sess.run(softmax,
                    feed_dict={z_prior: z_value, x_word_ids:data_sentence})
    print(out)
  
# Organization of this file is based on:
# https://github.com/ckmarkoh/GAN-tensorflow/blob/master/gan.py
# Usage of LSTM/GRU/building of an RNN with help from:
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
# https://hanxiao.github.io/2017/08/16/Why-I-use-raw-rnn-Instead-of-dynamic-rnn-in-Tensorflow-So-Should-You-0/

import argparse
import tensorflow as tf
import data.datasets

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

# Get embeddings. TODO i have no clue if this is a good way to do this...
embeddings = tf.Variable(-1.0, validate_shape=False, name=args.embeddings_tensor_name)
saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, "log/embeddings_model.ckpt")
  out_embeddings = sess.run([embeddings])
  
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
    
    elements_finished = (next_word_id == end_of_sentence_id)
    
    return (elements_finished, next_word, next_cell_state,
            emit_output, next_loop_state)
    
  
  cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
  #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
  #init_state = cell.zero_state(batch_size, tf.float32)
  emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(cell, loop_fn)
  
  return emit_ta, [V, Vb, C, Cb]

state_size = 10
z_prior = tf.placeholder(tf.float32, [state_size,1], name="z_prior")
emit_ta, g_params = build_generator(z_prior, out_embeddings, 1, num_classes, state_size=10)


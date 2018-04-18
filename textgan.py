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
from tensorflow.python.ops.rnn import _transpose_batch_time
import operator
from functools import reduce


tf.logging.set_verbosity(tf.logging.DEBUG)

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
all_sentences = reduce(operator.add, data.values(), [])
batch_size = 16
prior_size = 10
end_of_sentence_id = 1 # TODO this should probably come from the data.

# soft-argmax approximation (section 2.5 of zhang)
# not sure what to set it to; they don't discuss.
L = 100

# Internal LSTM state size.
state_size = 4

#max_sentence_length = max([len(sentence) for sentences in data.values() for sentence in sentences])
max_sentence_length = 20

# Get embeddings. TODO i have no clue if this is a good way to do this...
embeddings = tf.Variable(-1.0, validate_shape=False, name=args.embeddings_tensor_name)
saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, "log/embeddings_model.ckpt")
  out_embeddings = sess.run(embeddings)
  embedding_size = out_embeddings[0].shape[0]

def batch_gen():
  """
  Returns a TensorArray, not a Tensor.
  TensorArray length is batch_size, while each Tensor within is a 1-D vector of 
  word IDs.
  """
  i = 0
  while i < len(all_sentences):
    # TODO this is a mess...
    yield tf.stack([tf.pad(tf.stack([tf.nn.embedding_lookup(out_embeddings, id) for id in sentence[:max_sentence_length]]), [[0,max_sentence_length-min(len(sentence),max_sentence_length)],[0,0]], 'CONSTANT', constant_values=0) for sentence in all_sentences[i:i+batch_size]])
    i += batch_size
  
def build_generator(z_prior,
                    embeddings,
                    num_steps = 10,
                    state_size = 10):

  
  tf.Assert(tf.rank(z_prior) == 2, [z_prior])
  tf.Assert(tf.shape(z_prior)[0] == batch_size, [z_prior])
  tf.Assert(tf.shape(z_prior)[1] == prior_size, [z_prior])
  
  cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
  #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
  init_state = cell.zero_state(batch_size, tf.float32)

  V = tf.get_variable('V', [state_size, num_classes])
  Vb = tf.get_variable('Vb', [num_classes], initializer=tf.constant_initializer(0.0))
  C = tf.get_variable('C', [prior_size, state_size])
  Cb = tf.get_variable('Cb', [state_size], initializer=tf.constant_initializer(0.0))
  
  def loop_fn(time, cell_output, cell_state, loop_state):
    if cell_output is None:
      # time=0, everything here will be used for initialization only
      
      # TODO not sure about this
      # what i do know is that, according to the __call__ method of cells,
      # the state shape should be [batch size, state size], or [1, state size] for  us
      h1 = tf.tanh(tf.matmul(z_prior, C) + Cb)
      tf.Assert(tf.shape(h1)[0] == batch_size and tf.shape(h1)[1] == state_size, [h1])
      next_cell_state = tf.contrib.rnn.LSTMStateTuple(c = init_state.c,
                                                      h = h1)
      # [batch_size, num_classes]
      mul = tf.matmul(h1, V)+Vb
      next_word_id = tf.argmax(mul, axis = 1)
      # TODO used incorrectly -- has to be in the control chain or it won't run
      tf.Assert(tf.rank(next_word_id) == 1 and tf.shape(next_word_id)[0] == batch_size, [next_word_id])
      
      # section 2.5 of Zhang discusses this "soft-argmax". in simpler terms,
      # this is needed because argmax has no gradient and thus breaks the path
      # between the loss function and the variables V, Vb, etc.
      # The other way is to use something like REINFORCE, but zhang thankfully
      # proposes this simpler solution.
      next_word = tf.matmul(tf.nn.softmax(L*mul, axis=1), embeddings)
      # This is the old way
      #next_word = tf.map_fn(lambda id: tf.nn.embedding_lookup(embeddings, id), next_word_id, dtype=tf.float32)
      # TODO i'm fairly certain this is the correct shape, but i'm not 100%
      # TODO these don't work.
      tf.Assert(tf.rank(next_word) == 2, [next_word])
      tf.Assert(tf.shape(next_word)[0] == batch_size and tf.shape(next_word_id)[1] == embedding_size, [next_word])

      next_loop_state = (next_word_id, next_word) # this is what should be emitted next
      # this tells raw_rnn what the rest of our emits will look like.
      # first item: the id of the word that was generated
      # second item: the embedding of the word that was generated, calculated
      # via soft-argmax.
      # basically a placeholder for what INDIVIDUAL batch items will be emitting on
      # each iteration.
      emit_output = (tf.zeros([], dtype=tf.int64), tf.zeros([embedding_size], dtype=tf.float32))

    else: 
      # If this first emit_output return value is None, then the emit_ta
      # result of raw_rnn will have the same structure and dtypes as 
      # cell.output_size. Otherwise emit_ta will have the same structure,
      # shapes (prepended with a batch_size dimension), and dtypes as 
      # emit_output.
      # so we needed to expand this so that its first dim is the batch size
      #emit_output = tf.expand_dims(loop_state,0)
      # this shouldn't be the case anymore...we should be able to directly do:
      emit_output = loop_state
      next_cell_state = cell_state
      mul = tf.matmul(cell_state.h, V)+Vb
      next_word_id = tf.argmax(mul, axis = 1)
      # see above for the explanation of this soft-argmax
      next_word = tf.matmul(tf.nn.softmax(L*mul, axis=1), embeddings)
      #next_word = tf.map_fn(lambda id: tf.nn.embedding_lookup(embeddings, id), next_word_id, dtype=tf.float32)
      next_loop_state = (next_word_id, next_word)
      
    elements_finished = (time >= max_sentence_length) # TODO this should be improved
       
    return (elements_finished, next_word, next_cell_state,
            emit_output, next_loop_state)
    
  
  emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(cell, loop_fn)
  
  word_ids, words = emit_ta
  
  # must transpose first two dimensions from [sentence_length, batch_size] 
  # to [batch_size, sentence_length]
  return _transpose_batch_time(word_ids.stack()), _transpose_batch_time(words.stack()), [V, Vb, C, Cb]

def build_discriminator(x_data, x_generated):
  """
  assuming that these come in as shape [batch_size, sentence_length, embedding_size]
  """
  # TODO my asserts are all messed up.
  # assert(x_data.dtype == x_generated.dtype)
  # assert(x_data.shape.as_list() == x_generated.shape.as_list())
  # assert(x_data.shape[2] == x_generated.shape[2])
  
  # sentence_length = tf.maximum(tf.shape(x_data)[0], tf.shape(x_generated)[0])
  
  # first, make sure that they're the same size.
  # this is is kind of a hack that relies on broadcasting, but it's simple.
  # https://stackoverflow.com/questions/34362193/how-to-explicitly-broadcast-a-tensor-to-match-anothers-shape-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
  # x_data = tf.pad(x_data, [[0,0], [0, max_sentence_length-tf.shape(x_data)[0]], [0,0]], 'CONSTANT', constant_values=0)
  # x_generated = tf.pad(x_generated, [[0,0], [0, max_sentence_length-tf.shape(x_generated)[0]], [0,0]], 'CONSTANT', constant_values=0)
  # x_data.set_shape([batch_size,max_sentence_length,embedding_size])
  # x_generated.set_shape([batch_size,max_sentence_length,embedding_size])
    
  # concatenate batches
  x_in = tf.concat([x_data, x_generated], 0) 
  x_in = tf.expand_dims(x_in,3) # add channel dimension
  assert(x_in.get_shape().as_list() == [2*batch_size, max_sentence_length, embedding_size, 1])
  
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
  pool1 = tf.reduce_max(conv1, axis=1)
  # pool1 = tf.nn.max_pool(conv1, ksize=[1, tf.size(conv1)[1], 1, 1], strides=[1, 1, 1, 1],
                         # padding='VALID', name='pool1')

  
  # TODO no dropout implemented yet
  
  with tf.variable_scope('fullyconnected') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool1, [x_in.get_shape().as_list()[0], -1])
    # TODO can this be computed statically? i think it can, i'm just too lazy to do it right now.
    dim = tf.shape(reshape)[1]
    fc_weights = tf.Variable(tf.random_normal([dim, 2]), validate_shape=False)
    fc_bias = tf.Variable(tf.random_normal([2])) # TODO initialize to zero?
    fc = tf.matmul(reshape, fc_weights) + fc_bias
    fc = tf.identity(fc, name=scope.name)

  y_data = tf.nn.softmax(tf.slice(fc, [0, 0], [batch_size, -1], name=None))
  y_generated = tf.nn.softmax(tf.slice(fc, [batch_size, 0], [-1, -1], name=None))

  return y_data, y_generated, [conv1_filter, conv1_bias, fc_weights, fc_bias]
  
z_prior = tf.placeholder(tf.float32, [batch_size, prior_size], name="z_prior")

x_generated_ids, x_generated, g_params = build_generator(z_prior, out_embeddings, state_size=10)

# TODO i think my current gradient problem stems from here: we must output the generated embeddings
# instead of the looked-up embeddings. but we also should output the word ids so we actually
# know what the sentence was.
# TODO this only works b/c everything in the emit_ta tensorarray is the same length.
# x_generated = tf.map_fn(lambda sentence: tf.map_fn(lambda word_id: tf.nn.embedding_lookup(out_embeddings, word_id), sentence, dtype=tf.float32), emit_ta, dtype=tf.float32)

x_data = tf.Variable(tf.zeros(dtype=tf.float32, shape=[batch_size, max_sentence_length, embedding_size]))

y_data, y_generated, d_params = build_discriminator(x_data, x_generated)

d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
g_loss = - tf.log(y_generated)
optimizer = tf.train.AdamOptimizer(0.0001)
d_trainer = optimizer.minimize(d_loss, var_list=d_params)
# TODO having problems with this -- no path to g_params
# g_trainer = optimizer.minimize(g_loss, var_list=g_params)

# TODO this is needed on Windows
# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
  init = tf.global_variables_initializer()	
  writer = tf.summary.FileWriter("output", sess.graph)
  sess.run(init)
  writer.close()
  
  num_batches = len(all_sentences)//batch_size
  
  tf.logging.info("Beginnning training.")
  for batch_i, batch in enumerate(batch_gen()):
    tf.logging.info("Batch: {}/{}".format(batch_i, num_batches))
    
    z_value = np.random.normal(0, 1, size=(batch_size, prior_size)).astype(np.float32)
    
    # TODO this is kind of messy. basically, batch_gen() returns a tensor, which
    # must be "fed" through this assign call here. batch_gen() has to return a
    # tensor because i'm not sure how else to feed sentences in. 
    # if we feed just IDs, then each sentence has to be padded already when it
    # gets fed in, in which case we need some reserved ID which should be
    # converted to padding instead of being looked up in the embeddings.
    sess.run(x_data.assign(batch))
    out = sess.run(d_trainer,
                    feed_dict={z_prior: z_value})
    print(out)
  
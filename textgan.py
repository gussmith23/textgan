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
from discriminator import build_discriminator
from generator import build_generator

tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser(description='TextGAN implementation.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
parser.add_argument('--embeddings-file', type=str, required=True,
                    help='filepath of the embeddings file to use')
parser.add_argument('--dataset-name', type=str, required=True,
                    help='name of dataset')
parser.add_argument('--d-pretrain-filepath', type=str,
                    help='checkpoint filepath containing pretrained discriminator weights and biases')
args = parser.parse_args()

dataset_name = args.dataset_name
data, dictionary, reversed_dictionary, sender_dictionary, reversed_sender_dictionary = data.datasets.get(dataset_name)
num_classes = len(dictionary)
all_sentences = reduce(operator.add, data.values(), [])
batch_size = 8
prior_size = 10
end_of_sentence_id = 1 # TODO this should probably come from the data.

# Internal LSTM state size.
# TODO i think this should be 500
state_size = 10

#max_sentence_length = max([len(sentence) for sentences in data.values() for sentence in sentences])
max_sentence_length = 20

max_epochs = 100

# Get embeddings. TODO i have no clue if this is a good way to do this...
with open(args.embeddings_file, "rb") as f:
    embeddings = np.load(f)
embedding_size = embeddings.shape[1]

def batch_gen():
  """
  Returns a TensorArray, not a Tensor.
  TensorArray length is batch_size, while each Tensor within is a 1-D vector of 
  word IDs.
  """
  i = 0
  while i < len(all_sentences):
    # TODO this is a mess...
    yield tf.stack([tf.pad(tf.stack([tf.nn.embedding_lookup(embeddings, id) for id in sentence[:max_sentence_length]]), [[0,max_sentence_length-min(len(sentence),max_sentence_length)],[0,0]], 'CONSTANT', constant_values=0) for sentence in all_sentences[i:i+batch_size]])
    i += batch_size
  

  
z_prior = tf.placeholder(tf.float32, [batch_size, prior_size], name="z_prior")

x_generated_ids, x_generated, g_params = build_generator(z_prior, embeddings, num_classes, state_size, embedding_size, prior_size, max_sentence_length)

# TODO i think my current gradient problem stems from here: we must output the generated embeddings
# instead of the looked-up embeddings. but we also should output the word ids so we actually
# know what the sentence was.
# TODO this only works b/c everything in the emit_ta tensorarray is the same length.
# x_generated = tf.map_fn(lambda sentence: tf.map_fn(lambda word_id: tf.nn.embedding_lookup(out_embeddings, word_id), sentence, dtype=tf.float32), emit_ta, dtype=tf.float32)

x_data = tf.Variable(tf.zeros(dtype=tf.float32, shape=[batch_size, max_sentence_length, embedding_size]))

logits_data, logits_generated, encoding_data, encoding_generated, d_params = build_discriminator(x_data, x_generated, batch_size, max_sentence_length, embedding_size)
y_data, y_generated = tf.nn.softmax(logits_data), tf.nn.softmax(logits_generated)

d_loss = tf.reduce_mean(- (tf.log(y_data) + tf.log(1 - y_generated)))
g_loss = tf.reduce_mean(- tf.log(y_generated))
tf.summary.scalar("d_loss", d_loss)
tf.summary.scalar("g_loss", g_loss)

optimizer = tf.train.AdamOptimizer(0.0001)
d_trainer = optimizer.minimize(d_loss, var_list=d_params)
# TODO having problems with this -- no path to g_params
g_trainer = optimizer.minimize(g_loss, var_list=g_params)

merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

# IDK why this is outside of the with tf.Session... block
if args.d_pretrain_filepath is not None:
    init_d_fn = tf.contrib.framework.assign_from_checkpoint_fn(args.d_pretrain_filepath, tf.contrib.framework.get_variables_to_restore(include=['discriminator/conv/weights_3','discriminator/conv/weights_4', 'discriminator/conv/weights_5', 'discriminator/conv/bias', 'discriminator/discriminator_fc_1/weights', 'discriminator/discriminator_fc_1/bias', 'discriminator/discriminator_fc_2/weights', 'discriminator/discriminator_fc_2/bias', 'discriminator/encoder_fc_1/weights', 'discriminator/encoder_fc_1/bias', 'discriminator/encoder_fc_2/weights', 'discriminator/encoder_fc_2/bias']))


# TODO this is needed on Windows
# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

  if args.d_pretrain_filepath is not None:
    init_d_fn(sess)

  # Credit to https://blog.altoros.com/visualizing-tensorflow-graphs-with-tensorboard.html
  # for help with summaries.
  writer = tf.summary.FileWriter("output", sess.graph)

  init = tf.global_variables_initializer()	
  sess.run(init)
  
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
    # TODO https://github.com/tensorflow/tensorflow/issues/2382#issuecomment-224335676
    # might be a solution.
    sess.run(x_data.assign(batch))
    
    sess.run(d_trainer, feed_dict={z_prior: z_value})
    out_sentence, summary_str, _ = sess.run([x_generated_ids, merged_summary_op, g_trainer], feed_dict={z_prior: z_value})
    
    writer.add_summary(summary_str, batch_i)

    tf.logging.debug("Generated sentences: {}".format("\n".join([" ".join([reversed_dictionary[word_id] for word_id in sentence]) for sentence in out_sentence])))
  
  print("done!")
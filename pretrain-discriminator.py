# This file implements the pretraining described in Zhang 2017.
# Trains the discriminator to detect unmodified sentences vs from sentences
# which have two words randomly flipped.

import tensorflow as tf
from discriminator import build_discriminator
import data.datasets
import operator
from functools import reduce
import random
import argparse

tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser(
    description='Pretraining sentence discriminator.')
parser.add_argument(
    '--embeddings-file',
    type=str,
    required=True,
    help='filepath of the embeddings file to use')
parser.add_argument(
    '--embeddings-tensor-name',
    type=str,
    required=True,
    help='name of the embeddings tensor')
parser.add_argument(
    '--dataset-name', type=str, required=True, help='name of dataset')
parser.add_argument('--max-epoch', type=int, default=10)
args = parser.parse_args()

dataset_name = args.dataset_name
data, dictionary, reversed_dictionary, sender_dictionary, reversed_sender_dictionary = data.datasets.get(
    dataset_name)
num_classes = len(dictionary)
all_sentences = reduce(operator.add, data.values(), [])
batch_size = 8
prior_size = 10
end_of_sentence_id = 1  # TODO this should probably come from the data.
sentence_length = 20
max_epochs = 100

# Get embeddings. TODO i have no clue if this is a good way to do this...
embeddings = tf.Variable(
    -1.0, validate_shape=False, name=args.embeddings_tensor_name)
with tf.Session() as sess:
    tf.train.Saver().restore(sess, args.embeddings_file)
    embeddings = sess.run(embeddings)
    embedding_size = embeddings.shape[1]


# TODO though we call it "batch_size", the real batch size is 2*batch_size.
def batch_gen():
    """
  Returns a pair of tensors:
  - [batch_size, sentence_length, embedding_size]: the "pure" sentences
  - [batch_size, sentence_length, embedding_size]: the "tweaked" sentences
  """

    i = 0
    # While there are enough sentences left.
    # TODO could be cleaner?
    while len(all_sentences) - i >= 2 * batch_size:

        # get sentences, truncate to correct length.
        tweaked_sentences = [
            sentence[:sentence_length]
            for sentence in all_sentences[i:i + batch_size]
        ]
        i += batch_size

        # tweak
        for sentence in tweaked_sentences:
            # nice code from https://stackoverflow.com/questions/47724017
            idx = range(len(sentence))
            i1, i2 = random.sample(idx, 2)
            sentence[i1], sentence[i2] = sentence[i2], sentence[i1]

        # get sentences, truncate to correct length.
        pure_sentences = [
            sentence[:sentence_length]
            for sentence in all_sentences[i:i + batch_size]
        ]
        i += batch_size

        # TODO this is a mess...
        yield tf.stack([
            tf.pad(
                tf.stack([
                    tf.nn.embedding_lookup(embeddings, id) for id in sentence
                ]),
                [[0, sentence_length - min(len(sentence), sentence_length)],
                 [0, 0]],
                'CONSTANT',
                constant_values=0) for sentence in pure_sentences
        ]), tf.stack([
            tf.pad(
                tf.stack([
                    tf.nn.embedding_lookup(embeddings, id) for id in sentence
                ]),
                [[0, sentence_length - min(len(sentence), sentence_length)],
                 [0, 0]],
                'CONSTANT',
                constant_values=0) for sentence in tweaked_sentences
        ])

        i += batch_size


x_data = tf.Variable(
    tf.zeros(
        dtype=tf.float32, shape=[batch_size, sentence_length, embedding_size]))
x_data_tweaked = tf.Variable(
    tf.zeros(
        dtype=tf.float32, shape=[batch_size, sentence_length, embedding_size]))

y_data, y_data_tweaked, d_params = build_discriminator(
    x_data, x_data_tweaked, batch_size, sentence_length, embedding_size)

d_loss = tf.reduce_mean(-(tf.log(y_data) + tf.log(1 - y_data_tweaked)))
tf.summary.scalar("d_loss", d_loss)

optimizer = tf.train.AdamOptimizer(0.0001)
d_trainer = optimizer.minimize(d_loss, var_list=d_params)

merged_summary_op = tf.summary.merge_all()

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

saver = tf.train.Saver()

# TODO this is needed on Windows
# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # Credit to https://blog.altoros.com/visualizing-tensorflow-graphs-with-tensorboard.html
    # for help with summaries.
    writer = tf.summary.FileWriter("pretrain-discriminator-summary",
                                   sess.graph)

    sess.run(init_g)
    sess.run(init_l)

    num_batches = len(all_sentences) // batch_size

    tf.logging.info("Beginnning training.")

    for epoch in range(args.max_epoch):
        for batch_i, (pure_sentences, tweaked_sentences) in enumerate(
                batch_gen()):
            tf.logging.info("Epoch: {}/{}\tBatch: {}/{}".format(
                epoch + 1, args.max_epoch, batch_i + 1, num_batches))

            # TODO this is kind of messy. basically, batch_gen() returns tensors, which
            # must be "fed" through these assign call here. batch_gen() has to return a
            # tensor because i'm not sure how else to feed sentences in.
            # if we feed just IDs, then each sentence has to be padded already when it
            # gets fed in, in which case we need some reserved ID which should be
            # converted to padding instead of being looked up in the embeddings.
            sess.run(x_data.assign(pure_sentences))
            sess.run(x_data_tweaked.assign(tweaked_sentences))

            summary_str = sess.run(merged_summary_op)

            writer.add_summary(summary_str, batch_i)

            step = epoch * num_batches + batch_i
            if step % 10 == 0:
                # TODO hardcoded path.
                # TODO this is weird, i don't like how the output is done here.
                saver.save(
                    sess,
                    './pretrain-discriminator/pretrain-discriminator',
                    global_step=step)

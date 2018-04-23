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
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser(
    description='Pretraining sentence discriminator.')
parser.add_argument(
    '--embeddings-file',
    type=str,
    required=True,
    help='filepath of the embeddings file to use')
parser.add_argument(
    '--dataset-name', type=str, required=True, help='name of dataset')
parser.add_argument('--max-epoch', type=int, default=100)
# This is specified in Zhang 2017.
parser.add_argument('--batch-size', type=int, default=256)
# This is specified in Zhang 2017.
parser.add_argument('--learning-rate', type=float, default=0.00005)
parser.add_argument(
    '--checkpoint-dir',
    type=str,
    required=False,
    help=
    'directory containing latest checkpoint. if this flag is set, the model will be restored from this location.',
    default="pretrain-discriminator"
)
args = parser.parse_args()

dataset_name = args.dataset_name
data, dictionary, reversed_dictionary, sender_dictionary, reversed_sender_dictionary = data.datasets.get(
    dataset_name)
num_classes = len(dictionary)
all_sentences = reduce(operator.add, data.values(), [])
batch_size = args.batch_size
end_of_sentence_id = 1  # TODO this should probably come from the data.
sentence_length = 30

# Get embeddings. TODO i have no clue if this is a good way to do this...
with open(args.embeddings_file, "rb") as f:
    embeddings = np.load(f)
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

        tweaked_sentences, pure_sentences = [], []

        for sentence in all_sentences[i:i + 2 * batch_size]:
            if len(sentence) > 2 and len(tweaked_sentences) < batch_size:
                tweaked_sentences.append(sentence[:sentence_length])
            elif len(pure_sentences) < batch_size:
                pure_sentences.append(sentence[:sentence_length])
            else:
                # In this case we assume it's a 1-word sentence but pure_sentences
                # is already full.
                # TODO this batch creation is still non-ideal.
                tweaked_sentences.append(sentence[:sentence_length])

        assert (len(pure_sentences) == batch_size)
        assert (len(tweaked_sentences) == batch_size)
        i += 2 * batch_size

        # tweak
        for sentence in tweaked_sentences:
            # TODO this is non-ideal. we shouldn't add tweaked sentences that
            # aren't actually tweaked, and you can't tweak a sentence that has
            # only one word.
            # I've eliminated this problem for now by filtering out really
            # short messages.
            if len(sentence) < 5:
                tf.logging.warn(
                    "Adding sentence of length {} into tweaked sentences...this is non-ideal; see TODO".
                    format(len(sentence)))
                continue
            # nice code from https://stackoverflow.com/questions/47724017
            # though we've modified it heavily.
            # note the [:-1]. this is so that we don't swap the END token.
            # TODO i'm not sure if this is actually helpful, but I have a
            # hunch it is.
            # We actually permute MORE than they do in Zhang. We shuffle more words.
            idx = range(len(sentence))[:-1]
            idxs = random.sample(idx, 4)
            shuffled_idxs = idxs[:]
            random.shuffle(shuffled_idxs)
            for i1, i2 in zip(idxs, shuffled_idxs):
                sentence[i1], sentence[i2] = sentence[i2], sentence[i1]

        # TODO this is a mess...
        yield np.stack([
            np.pad(
                np.stack([embeddings[id] for id in sentence]),
                [[0, sentence_length - min(len(sentence), sentence_length)],
                 [0, 0]],
                'constant',
                constant_values=0) for sentence in pure_sentences
        ]), np.stack([
            np.pad(
                np.stack([embeddings[id] for id in sentence]),
                [[0, sentence_length - min(len(sentence), sentence_length)],
                 [0, 0]],
                'constant',
                constant_values=0) for sentence in tweaked_sentences
        ])


x_data = tf.placeholder(
    dtype=tf.float32, shape=[batch_size, sentence_length, embedding_size])
x_data_tweaked = tf.placeholder(
    dtype=tf.float32, shape=[batch_size, sentence_length, embedding_size])

logits_data, logits_tweaked, embedding_data, embedding_tweaked, d_params = build_discriminator(
    x_data, x_data_tweaked, batch_size, sentence_length, embedding_size)

global_step = tf.Variable(0, name='global_step', trainable=False)

#d_loss = tf.reduce_mean(-(tf.log(y_data) + tf.log(1 - y_data_tweaked)))
d_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.concat([logits_data, logits_tweaked], axis=0),
        labels=tf.constant(
            [[1, 0]] * batch_size + [[0, 1]] * batch_size, dtype=tf.float32)))
tf.summary.scalar("d_loss", d_loss)

optimizer = tf.train.AdamOptimizer(args.learning_rate)
d_trainer = optimizer.minimize(
    d_loss, var_list=d_params, global_step=global_step)

merged_summary_op = tf.summary.merge_all()

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

# The first saver is for saving/restoring for pretraining.
# The second saver is for restoring the weights for use in other networks.
saver_all = tf.train.Saver()
saver_just_weights_and_biases = tf.train.Saver(
    var_list=tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator"))

# TODO this is needed on Windows
# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # TODO use --restore flag
    # if args.checkpoint_dir is not None:
        # tf.logging.info("Restoring from {}".format(args.checkpoint_dir))
        # saver_all.restore(sess, tf.train.latest_checkpoint(
            # args.checkpoint_dir))

    # Credit to https://blog.altoros.com/visualizing-tensorflow-graphs-with-tensorboard.html
    # for help with summaries.
    writer = tf.summary.FileWriter("pretrain-discriminator-summary",
                                   sess.graph)

    sess.run(init_g)
    sess.run(init_l)

    # TODO batch size is actually batch_size*2
    num_batches = len(all_sentences) // batch_size * 2

    tf.logging.info("Beginnning training.")

    for epoch in range(args.max_epoch):
        tf.logging.info("Epoch: {}/{}".format(epoch + 1, args.max_epoch))
        for batch_i, (pure_sentences, tweaked_sentences) in enumerate(
                batch_gen()):

            # TODO this is kind of messy. basically, batch_gen() returns tensors, which
            # must be "fed" through these assign call here. batch_gen() has to return a
            # tensor because i'm not sure how else to feed sentences in.
            # if we feed just IDs, then each sentence has to be padded already when it
            # gets fed in, in which case we need some reserved ID which should be
            # converted to padding instead of being looked up in the embeddings.
            # sess.run(x_data.assign(pure_sentences))
            # sess.run(x_data_tweaked.assign(tweaked_sentences))

            # Alright, I fixed my big issue with training crashing. the problem
            # was with the fact that we were using tf calls in the generator,
            # which creates new nodes.

            step = epoch * num_batches + batch_i

            summary_str = sess.run(
                merged_summary_op,
                feed_dict={
                    x_data: pure_sentences,
                    x_data_tweaked: tweaked_sentences
                })

            writer.add_summary(summary_str, step)

            if step % 1000 == 0:
                saver_all.save(
                    sess,
                    os.path.join('.', args.checkpoint_dir, 'model'),
                    global_step=global_step.eval())
                saver_just_weights_and_biases.save(
                    sess,
                    os.path.join('.', args.checkpoint_dir, 'weights-biases'),
                    global_step=global_step.eval())

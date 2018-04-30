import argparse
import tensorflow as tf
import data.datasets
import numpy as np
from generator import build_generator
from bleu import compute_bleu

parser = argparse.ArgumentParser(description='Generate sentences from trained TextGAN model.')
parser.add_argument(
    '--num-to-generate',
    type=int,
    default=320,
    help='number of sentences to generate')
parser.add_argument(
    '--embeddings-file',
    type=str,
    required=True,
    help='filepath of the embeddings file to use')
parser.add_argument(
    '--textgan-filepath',
    type=str,
    required=True,
    help='filepath to checkpoint containing trained textgan parameters')
parser.add_argument(
    '--dataset-name', type=str, required=True, help='name of dataset')
args = parser.parse_args()

data, dictionary, reversed_dictionary = data.datasets.get(args.dataset_name)

num_classes = len(reversed_dictionary)
z_prior_size = 900
end_of_sentence_id = dictionary["<END>"] # TODO shouldn't be hardcoded like this.
max_sentence_length = 20
hidden_layer_size = 500

with open(args.embeddings_file, "rb") as f:
    embeddings = np.load(f)
embedding_size = embeddings.shape[1]

z_prior = tf.placeholder(
    tf.float32, [args.num_to_generate, z_prior_size], name="z_prior")

x_generated_ids, _, _ = build_generator(z_prior, embeddings, num_classes,
                                        hidden_layer_size, embedding_size,
                                        z_prior_size, max_sentence_length)

# TODO this is needed on Windows
# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    tf.contrib.framework.assign_from_checkpoint_fn(
        args.textgan_filepath, tf.trainable_variables("generator"))(sess)

    z_value = np.random.normal(
        0, 1, size=(args.num_to_generate, z_prior_size)).astype(np.float32)

    out_sentence = sess.run(x_generated_ids, feed_dict={z_prior: z_value})

    sentences_as_ids, sentences_as_strs = [], []
    for sentence in out_sentence:
        try:
            take_len = 1 + np.where(sentence==end_of_sentence_id)[0][0]
        except IndexError:
            take_len = len(sentence)
        sentences_as_ids.append(sentence[:take_len].tolist())
        sentences_as_strs.append(" ".join([reversed_dictionary[word_id] for word_id in sentence[:take_len]]))

bleu_out = compute_bleu([data]*len(sentences_as_ids), sentences_as_ids)

for sentence in sentences_as_strs: print(sentences_as_strs)
print(bleu_out)
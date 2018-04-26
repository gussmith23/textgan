import argparse
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import data.datasets
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description='Visualize embeddings.')
parser.add_argument(
    '--embeddings-file',
    type=str,
    required=True,
    help='filepath of the embeddings file to use')
# parser.add_argument(
# '--embeddings-tensor-name',
# type=str,
# required=True,
# help='name of the embeddings tensor')
parser.add_argument(
    '--dataset-name', type=str, required=True, help='name of dataset')
parser.add_argument('--filename', type=str)
args = parser.parse_args()

# The default filename. Can't be set in add_argument because it relies on
# another flag.
if args.filename is None:
    args.filename = 'tsne-{}-{}.png'.format(args.dataset_name,
                                            time.strftime("%Y%m%d-%H%M%S"))

dataset_name = args.dataset_name
data, dictionary, reversed_dictionary, sender_dictionary, reversed_sender_dictionary = data.datasets.get(
    dataset_name)

# Get embeddings. TODO i have no clue if this is a good way to do this...
# embeddings = tf.Variable(
# -1.0, validate_shape=False, name=args.embeddings_tensor_name)
# with tf.Session() as sess:
# tf.train.Saver().restore(sess, args.embeddings_file)
# embeddings = sess.run(embeddings)
# embedding_size = embeddings.shape[1]
# Switched to using saved numpy arrays instead.
with open(args.embeddings_file, "rb") as f:
    embeddings = np.load(f)
embedding_size = embeddings.shape[1]


# Visualization
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    plt.savefig(filename)


tsne = TSNE(
    perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
plot_only = 500
low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
labels = [reversed_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels, args.filename)

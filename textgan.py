import argparse

parser = argparse.ArgumentParser(description='TextGAN implementation.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
parser.add_argument('--embeddings-file', type=str, required=True,
                    help='filepath of the embeddings file to use')
parser.add_argument('--embeddings-tensor-name', type=str, required=True,
                    help='name of the embeddings tensor')
args = parser.parse_args()


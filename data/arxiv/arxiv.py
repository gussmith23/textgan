import os
import re
import util.sentences
import pickle
import operator
from functools import reduce
import collections  # for counter

r = re.compile(r"^\\\\$", flags=re.MULTILINE)

pickle_filepath = os.path.join(os.path.dirname(__file__), "arxiv.p")


def get_data():
    if os.path.isfile(pickle_filepath):
        return pickle.load(open(pickle_filepath, "rb"))
    else:
        return parse_data()


def parse_data():
    all_sentences = []
    for dirpath, dirnames, filenames in os.walk(
            os.path.join(os.path.dirname(__file__), "data")):
        for filename in filenames:
            with open(os.path.join(dirpath, filename), "r") as f:
                text = f.read()
            sentences = util.sentences.split_and_clean_sentences(
                " ".join(r.split(text)[2].strip().split('\n')))
            sentences = list(map(util.sentences.sentence_to_words, sentences))
            sentences = list(filter(lambda s: len(s) > 5, sentences))
            all_sentences += sentences

    # Get a giant list of all words.
    all_words = reduce(operator.add, all_sentences, [])

    n_words = 6000

    # TODO should be moved to another function so it can be used for parsing
    # other datasets. It doesn't just apply here.
    # This code comes from wherever the code in
    # https://stackoverflow.com/questions/45735357/what-is-unk-token-in-vector-representation-of-words
    # came from. I have a feeling it's originally from tensorflow docs.
    # Histogram of words.
    count = [['UNK', -1]]
    count.extend(collections.Counter(all_words).most_common(n_words - 1))
    # TODO we're assuming that the <END> token ends up in the collection. this is really bad practice!
    # however, it should always be among the top most common words.
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = []
    unk_count = 0
    for sentence in all_sentences:
        converted_sentence = []
        for word in sentence:
            if word in dictionary:
                converted_sentence.append(dictionary[word])
            else:
                unk_count += 1
                converted_sentence.append(0)
        data.append(converted_sentence)
    count[0][1] = unk_count

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    data_out = (data, dictionary, reversed_dictionary)
    pickle.dump(data_out, open(pickle_filepath, "wb"))
    return data_out

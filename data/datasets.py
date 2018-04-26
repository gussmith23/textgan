import random
import operator
from functools import reduce


def get(name):
    """Get raw dataset (not split into train/test/validation.)
    """
    if name == 'babblebuds':
        from data.babblebuds.babblebuds import get_data
        return get_data()
    elif name == 'arxiv':
        from data.arxiv.arxiv import get_data
        return get_data()
    else:
        raise ValueError("Unrecognized dataset {}.".format(name))


def get_split(name):
    """Get dataset split into train/test/validation sets.
    
    Returns: the same structure as get(), but appended with the train, test,
    and validation sets.
    """
    if name == 'babblebuds':
        from data.babblebuds.babblebuds import get_data
        data = get_data()

        all_sentences = reduce(operator.add, data[0].values(), [])

        rand_state = random.getstate()
        random.seed(23)  # make the results reproducible!
        random.shuffle(all_sentences)
        random.setstate(rand_state)

        training = .6
        validation = .2
        test = .2
        assert (training + validation + test == 1)

        num_training = int(training * len(all_sentences))
        num_validation = int(validation * len(all_sentences))

        training_data = all_sentences[:num_training]
        validation_data = all_sentences[num_training:
                                        num_training + num_validation]
        testing_data = all_sentences[num_training + num_validation:]

        return data + (training_data, validation_data, testing_data)

    else:
        raise ValueError("Unrecognized dataset {}.".format(name))

These directories provide data and methods to access the data. 

Datasets should be accessed via `data.datasets.get(name)`. This function returns a tuple of the following values:
- The dataset itself, which will be a dictionary mapping a sender ID to a list of sentences. The sender ID will be a number identifying the sender of the sentence. Each sentence will be a list of word IDs.
- The dataset dictionary, which will map words to IDs.
- The reverse dictionary, which maps IDs to words.
- The sender ID table, mapping senders to names.

To implement a new dataset, you can follow the example of the babblebuds dataset. Implement a function which parses your dataset into the format requested above, and then add a call to this function in `data.datasets.get(name)`.

The datasets should include the special token <UNK> for unknown words, plus the special token <END> for end-of-sentence.
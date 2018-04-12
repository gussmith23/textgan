import re
import os
import pickle
import util.sentences
from data.babblebuds.babblebuds_config import user_whitelist
import collections # for counter
import logging

data_filepath = os.path.join(os.path.dirname(__file__), "babble-buds-dump.txt")
pickle_filepath = os.path.join(os.path.dirname(__file__), "babble-buds-dump.p")

# https://regex101.com/r/A0pX4G/3/
pat = re.compile(r"\[\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d\] (?P<sender>(?:\w+ ?)+)(?: \[.*\])? >>> (?P<message>.*)")

def get_data():
  if os.path.isfile(pickle_filepath):
    return pickle.load(open(pickle_filepath,"rb"))
  else: return parse_data()
  
def parse_data():
  """Parse the raw data into a friendly format and pickle it.
  """
  
  # { sender name : [ message, message, ... ], ...} 
  sentences_by_sender = {}

  with open(data_filepath, encoding="utf8") as f:
    # Skip the first two lines (chat name and info)
    next(f);next(f)
    
    message = None
    sender = None
    
    logging.info("parsing messages from file, cleaning messages, and splitting them into setences...")
    
    for l in f:
      match =  pat.match(l)
      # If we match, it's the beginning of a new message.
      if match: 
        # First, add the just-completed message to the list.
        if message and sender and sender in user_whitelist:
          # Split a message into sentences, and then filter out "bad" sentences
          new_sentences = util.sentences.split_and_clean_sentences(message)
          # Split each sentence into words
          new_sentences = list(map(util.sentences.sentence_to_words, new_sentences))
          if (sender in sentences_by_sender): sentences_by_sender[sender].extend(new_sentences)
          else: sentences_by_sender[sender] = new_sentences
        
        sender = match.group('sender')
        message = match.group('message')
      
      # If we don't match, it's a continuation of a previous message.
      else:
        assert((message is not None) and (sender is not None))
        message = message + l
    
  # Now we generate the format expected.
  # Currently, sentences_by_sender looks like
  # {
  #   sender: [sentence1, sentence2, ...],
  #   ....
  # }
  # Where sentence1 looks like [word1,word2,word3,...]
    
  # sender_dictionary maps name to ID.
  sender_enum = enumerate(sentences_by_sender.keys())
  sender_dictionary = dict(map(lambda x: (x[1],x[0]), sender_enum))
  reversed_sender_dictionary = dict(sender_enum)
  del sender_enum
  
  import operator
  from functools import reduce
  
  # Get a giant list of all words.
  all_words = reduce(operator.add, reduce(operator.add, sentences_by_sender.values(), []), [])
  
  logging.info("converting sentences into arrays of word ids...")
  
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
  
  data = {}
  unk_count = 0
  for sender in sentences_by_sender.keys():
    data[sender_dictionary[sender]] = []
    for sentence in sentences_by_sender[sender]:
      converted_sentence = []
      for word in sentence:
        if word in dictionary:
          converted_sentence.append(dictionary[word])
        else:
          unk_count += 1
          converted_sentence.append(0)
      data[sender_dictionary[sender]].append(converted_sentence)
  count[0][1] = unk_count
  
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
  data_out = (data, dictionary, reversed_dictionary, sender_dictionary, reversed_sender_dictionary)
  pickle.dump(data_out, open(pickle_filepath,"wb"))
  return data_out
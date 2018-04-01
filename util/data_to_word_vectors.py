import string

def data_to_word_vectors(data):

  # Vocab histogram
  vocabulary = {}
  
  for l in data:
    # Clean non-printable chars.
    # TODO this is really terrible -- there's gotta be a better way
    l = ''.join(list(filter(lambda c: c in set(string.printable), l)))
    
    for word in l.lower().split(): 
      if word in vocabulary: vocabulary[word] = vocabulary[word] + 1
      else: vocabulary[word] = 1
   
  return vocabulary
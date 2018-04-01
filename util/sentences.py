# This code comes from a stackoverflow answer.
# https://stackoverflow.com/questions/4576077/python-split-text-on-sentences?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# -*- coding: utf-8 -*-
import re
caps = "([A-Z])"
digits = "([0-9])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = text.replace("e.g.","e<prd>g<prd>")
    text = text.replace("E.g.","E<prd>g<prd>")
    text = text.replace("i.e.","i<prd>e<prd>")
    text = text.replace("I.e.","I<prd>e<prd>")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
    
import string

website_parts = re.compile(r"(https?:\/\/|[.](com|net|org|io|gov|me|edu))",flags=re.IGNORECASE)
    
def clean_sentences(sentences): 
  
  # clean non-printable chars
  new_sentences = []
  for s in sentences:
    # Clean non-printable chars.
    # TODO this is really terrible -- there's gotta be a better way
    s = ''.join(list(filter(lambda c: c in set(string.printable), s)))
    s = s.strip() # Strip again after cleaning.
    new_sentences.append(s)
  
  # Step 1: filter out entire sentences.
  # TODO should probably be its own function
  # Filter sentences with links
  f = filter(lambda s: website_parts.search(s) is None, new_sentences)
  # Filter single-character sentences
  f = filter(lambda s: len(s) > 1, f)
    
  return list(f)
  
def split_and_clean_sentences(sentences):
  return clean_sentences(split_into_sentences(sentences))
  
# TODO this does not handle contractions.
def sentence_to_words(sentence):
  sentence = sentence.lower()
  # translate punctuation to spaces. it'll be stripped out when we split the sentence.
  sentence = sentence.translate(str.maketrans(string.punctuation," "*len(string.punctuation)))
  sentence = sentence.strip()
  return sentence.split()

def to_word_vectors(sentences, dictionary = None):
  """
    
  dictionary: {word:id} pairs
  """  
  
  
import collections

def something(sentences):
  
  sentences = map(sentence_to_words, sentences)
  
  words = [word for sentence in sentences for word in sentence]
  
  # This code comes from wherever the code in 
  # https://stackoverflow.com/questions/45735357/what-is-unk-token-in-vector-representation-of-words
  # came from. I have a feeling it's originally from tensorflow docs.
  # Histogram of words.
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

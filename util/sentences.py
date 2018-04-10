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
    # this was their attempt at getting rid of the empty string that comes after the split.
    # i.e. hello. becomes ['hello','']
    #sentences = sentences[:-1]
    # because people so often don't use punctuation, we will do this a different way: 
    # by getting rid of empty strings (after stripping)
    sentences = [s.strip() for s in sentences]
    sentences = list(filter(lambda sentence: sentence != '', sentences))
    return sentences
    
import string

    
def clean_sentences(sentences): 
  
  website_parts = re.compile(r"(https?:\/\/|[.](com|net|org|io|gov|me|edu))",flags=re.IGNORECASE)
  telegram_keywords = re.compile(r"(\[document\]|\[photo\]|\[webpage\])",flags=re.IGNORECASE)
  
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
  # Telegram-specific strings to ignore.
  f = filter(lambda s: telegram_keywords.search(s) is None, f)
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
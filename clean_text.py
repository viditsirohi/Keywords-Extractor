import spacy

import gensim
from gensim.utils import simple_preprocess

import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
eng_words = set(nltk.corpus.words.words())

# You can use any spacy language or any size for vocab 
nlp = spacy.load("en_core_web_sm")


# Clean the text using 'simple_preprocess()'. tokenization, pancuation removal, remove unnecessary characters
def sent_to_words(sentence):
    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


# Removing stopwords and non english words
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# Lemmatising the text
def lemmatization(texts, allowed_postags=['NOUN','PROPN', 'ADV', 'ADJ', 'NUM', 'VERB']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Removing non english words
def remove_non_english(texts):
        return [[word for word in doc if word.lower() in eng_words] for doc in texts]


# Pre processing the text

def textPreProcessing(doc):
    words = list(sent_to_words(doc))
    stopwords_removed = remove_stopwords(words)
    lemmatised_words = lemmatization(stopwords_removed)
    non_english_removed = remove_non_english(lemmatised_words)
    return (' '.join(non_english_removed[0]))
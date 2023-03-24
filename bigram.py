import spacy


import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
eng_words = set(nltk.corpus.words.words())

# You can use any spacy language or any size for vocab 
nlp = spacy.load("en_core_web_sm")


# Processing the text

def textProcessing(doc):
    # Prepocessing of input text with 
    # 1. tokenisation and Lemmatisation
    # 2. Removing stop words 
    # 3. Creating and removing custom stop words.
    # 4. Generating required Vocabulary from input
    # 5. Preprocessing the input 
    Words = []
    Word_set = []
    trimmed_word_set = []
    removing_duplicates = []
    arr = []
    vocab = []
    vocab_dict = {}

    # Tokenising the text
    doc = nlp(doc.upper())

    for possible_words in doc:
        if possible_words.pos_ in ['NOUN','PROPN', 'ADV', 'ADJ', 'NUM', 'VERB'] :
            Words.append([possible_words , [child for child in possible_words.children]])
       
    
    for i,j in Words:
        for k in j:
            Word_set.append([k,i])

    
    for i , j in Word_set:
        if i.pos_ in ['NOUN','PROPN', 'ADV', 'ADJ', 'NUM', 'VERB']:
            trimmed_word_set.append([i ,j])
            
    # Removing Duplicates
    for word in trimmed_word_set:
        if word not in removing_duplicates:
            removing_duplicates.append(word)
    
    
    for i in removing_duplicates:
        strs = ''
        for j in i:
            strs += str(j)+" "
        arr.append(strs.strip())

    
    for word in Word_set:
        string = ''
        for j in word:
            string+= str(j)+ " "
        vocab.append(string.strip())

    
    for word in vocab:
        vocab_dict[word]= 0
        
    for word in arr:
        vocab_dict[word]+= 1

    return vocab_dict , arr

def computeTF(wordDict,bow):
    '''Computing TF(Term Frequency of the vocab) '''
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict


def computeIDF(doclist):
    '''Computing IDF for the vocab '''
    import math 
    count = 0
    idfDict = {}
    for element in doclist:
        for j in element:
            count+=1
    N = count

    # count no of documents that contain the word w
    idfDict = dict.fromkeys(doclist[0].keys(),0)

    for doc in doclist:
        for word,val in doc.items():
            if val>0:
                idfDict[word]+= 1

    # divide N by denominator above
    for word,val in idfDict.items():
        if val == 0:
            idfDict[word] = 0.0
        else:
            idfDict[word] = math.log(N / float(val))

    return idfDict

def computeTfidf(tf,idf):
    '''Computing TF-IDF for the words in text '''
    tfidf = {}
    sorted_list = []
    for word , val in tf.items():
        tfidf[word] = val * idf[word]

    ranking_list  = sorted(tfidf.items(),reverse=True, key = lambda kv:(kv[1], kv[0]))[:25]
    for i, _ in ranking_list:
        sorted_list.append(i)

    return sorted_list
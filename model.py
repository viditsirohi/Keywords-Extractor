import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import clean_text as ct

def textprocessing(doc):
    doc = ct.textPreProcessing(doc)
    dataset = [doc.upper()]
    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(dataset)
    df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    return df.head(25)
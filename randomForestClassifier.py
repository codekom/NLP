import pandas as pd
from pandas import DataFrame
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def word_extract(text):
    cln_text = BeautifulSoup(text)
    cd_t = cln_text.get_text()
    cln_textL = re.sub("[^a-zA-Z]"," ",cd_t)
    new_text = cln_textL.lower()
    words = new_text.split()
    stopw = set(stopwords.words("english")) 
    words = [w for w in words if not w in stopw]
    #print(words)
    return (" ".join(words))

if __name__=="__main__":
    train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
    n = train["review"].size
    print(n)
    cleaned_review = list()
    x = int(n/2)
    for i in range(x):
        w_text = word_extract(train["review"][i])
        cleaned_review.append(w_text)
    vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
    word_features = vectorizer.fit_transform(cleaned_review)
    wword_features = DataFrame(word_features.A)
    print(wword_features)
    print(word_features.shape)
    
    #mfile = open('mfile.txt','w')
    #mfile.write(word_features)
    print(word_features)
    forest = RandomForestClassifier(n_estimators=100)
    
    p=list()
    q=list()
    
    for i in range(x):
        p.append(train["sentiment"][i])
        
    for j in range(x+1,n):
        q.append(train["sentiment"][j])
    forest = forest.fit(word_features,p)

    clean_testR = list()
    for i in range(x+1,n):
        w_t = word_extract(train["review"][i])
        clean_testR.append(w_t)
    test_features = vectorizer.transform(clean_testR)
    result = forest.predict(test_features)
    #print(result)
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(len(result)):
        if result[i]==q[i]:
            if result[i]==1:
                tp=tp+1
            else:
                tn=tn+1
        else:
            if result[i]==1:
                fp=fp+1
            else:
                fn=fn+1

    print("True positive = %d" % tp)
    print("False positive = %d" % fp)
    print("False negative = %d" % fn)
    print("True negative = %d" % tn)
    c = tp+tn+fp+fn
    print("Accuracy = %f" % (((tp+tn)/c)*100))
        

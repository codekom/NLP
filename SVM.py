import pandas as pd
from pandas import DataFrame
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

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
    test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
    n = train["review"].size
    nt = test["review"].size
    cleaned_review = list()
    x = n
    y = nt
    for i in range(x):
        w_text = word_extract(train["review"][i])
        cleaned_review.append(w_text)
    
    vectorizer = TfidfVectorizer(min_df=5,max_df = 0.8,sublinear_tf=True,use_idf=True)
    train_vectors = vectorizer.fit_transform(cleaned_review)
    
    p=list()
    q=list()
    
    for i in range(x):
        p.append(train["sentiment"][i])
        
    clean_testR = list()
    for i in range(y):
        w_t = word_extract(test["review"][i])
        clean_testR.append(w_t)
		
    test_vectors = vectorizer.transform(clean_testR)

    # Perform classification with SVM, kernel=linear
    print('Classification with linear kernel\n')
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, p)
    prediction_linear = classifier_linear.predict(test_vectors)

    #print(result)
    print(prediction_linear)
	
        

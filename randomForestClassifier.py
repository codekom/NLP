import pandas as pd
import glob
import nltk
import operator
import math
from bs4 import BeautifulSoup
    
train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3);
print(train.shape)
print(train.columns.values)
print(train["review"][0])
cln_text = BeautifulSoup(train["review"][0])
print(cln_text.get_text())

    
    

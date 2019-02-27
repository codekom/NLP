import pandas as pd
import glob
import nltk
import operator
import math
    
files = glob.glob('E:/machine_learning/dataset/aclImdb/train/pos/*.txt')

BOW_df = pd.DataFrame( columns=['Positive','Negative'])
PN_file = pd.DataFrame( columns=['Total_Words'])
PN_total = pd.DataFrame( columns=['WordCount'])
words_set = set()

##myfile = open("training_set.csv",'w')
##wr = csv.writer(myfile)
##a=[]
##a.append("Review")
##a.append("Label")
##wr.writerow(a)

def expand_around_chars(text, characters):
    for char in characters:
        text = text.replace(char, " "+char+" ")
    return text
 
def split_text(text):
    #text = strip_quotations_newline(text)
    text = expand_around_chars(text, '".,()[]{}:;~')
    splitted_text = text.split(" ")
    cleaned_text = [x for x in splitted_text if len(x)>1]
    text_lowercase = [x.lower() for x in cleaned_text]
    return text_lowercase


senti_words_in_currDoc = []
def no_of_sentiment_words(text):
    #print(text)
    #for line in text:
        #print(line)
    text = nltk.word_tokenize(text)
    tags = nltk.pos_tag(text)
    sentiWords = 0
    for tagged_token in tags:
        if tagged_token[1] in ['JJ','JJR','JJS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ','UH']:
            word = tagged_token[0]
            sentiWords+=1
            senti_words_in_currDoc.append(word)
                
    return sentiWords


tot_pwords = 0
for file in files:
    p = open(file,encoding="utf8")
    tfile = p.read()
    #print(tfile)
    cn_word = 0

    splitted_text = split_text(tfile)
    text = nltk.word_tokenize(tfile)
    tags = nltk.pos_tag(text)
    label = 'Positive'
    
    for tagged_token in tags:
        if tagged_token[1] in ['JJ','JJR','JJS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ','UH']:
            word = tagged_token[0]
            cn_word+=1
            if len(words_set)==0 or (word not in words_set) :
                words_set.add(word)
                #print(word)
                BOW_df.loc[word] = [0,0]
                BOW_df.ix[word][label] += 1
            else:
                BOW_df.ix[word][label] += 1

    PN_file.loc[file] = cn_word
    tot_pwords+= cn_word
    p.close()

PN_total.loc['Positive'] = tot_pwords
files = glob.glob('E:/machine_learning/dataset/aclImdb/train/neg/*.txt') 

tot_nwords = 0
for file in files:
    p = open(file,encoding="utf8")
    tfile = p.read()
    cn_word = 0
    
    text = nltk.word_tokenize(tfile)
    tags = nltk.pos_tag(text)
    label = 'Negative'
    
    for tagged_token in tags:
        if tagged_token[1] in ['JJ','JJR','JJS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ','UH']:
            word = tagged_token[0]
            cn_word+=1
            if len(words_set)==0 or (word not in words_set) :
                words_set.add(word)
                #print(word)
                BOW_df.loc[word] = [0,0]
                BOW_df.ix[word][label] += 1
            else:
                BOW_df.ix[word][label] += 1

    PN_file.loc[file] = cn_word
    tot_nwords+= cn_word
    p.close()

PN_total.loc['Negative'] = tot_nwords
#print(BOW_df)
#print(PN_file)
#print(PN_total)

all_words = list(BOW_df.index.values)
class_prob_pos = tot_pwords/(tot_pwords + tot_nwords)
class_prob_neg = tot_nwords/(tot_pwords + tot_nwords)
class_probabilities = {'Positive':class_prob_pos , 'Negative':class_prob_neg}
labels = class_probabilities.keys()
words_per_class = {}

for label in labels:
    words_per_class[label] = BOW_df[label].sum()

def nb_classify(document):
    #print(document)
    no_words_in_doc = no_of_sentiment_words(document)
    # senti_words_in_currDoc initialized now
    current_class_prob = {}
    for label in labels:
        prob = math.log(class_probabilities[label],2) - no_words_in_doc * math.log(words_per_class[label],2)
        for word in senti_words_in_currDoc:
            if word in all_words:
                occurence = BOW_df.loc[word][label]
                if occurence > 0:
                    prob += math.log(occurence,2)
                else:
                    prob += math.log(1,2)
            else:
                prob += math.log(1,2)
        current_class_prob[label] = prob
    #sort the current_class_prob dictionary by its values, so we can take the key with the maximum value
    sorted_labels = sorted(current_class_prob.items(), key=operator.itemgetter(1))
    most_probable_class = sorted_labels[-1][0]
    return most_probable_class


total = 0
files = glob.glob('E:/machine_learning/dataset/aclImdb/test/pos/*.txt')
total = len(files)
corr=0
wrong=0
for file in files:
    p = open(file,encoding="utf8")
    tfile = p.read()
    senti_words_in_currDoc = []
    classification = nb_classify(tfile)
    if classification == 'Positive':
        # Correctly classified
        corr+=1
    else:
        # Wrongly classified
        wrong+=1
    p.close()

files = glob.glob('E:/machine_learning/dataset/aclImdb/test/neg/*.txt')
total+= len(files)
for file in files:
    p = open(file,encoding="utf8")
    tfile = p.read()
    senti_words_in_currDoc = []
    classification = nb_classify(tfile)
    if classification == 'Negative':
        # Correctly classified
        corr+=1
    else:
        # Wrongly classified
        wrong+=1
    p.close()

accuracy = corr/total
print(accuracy)

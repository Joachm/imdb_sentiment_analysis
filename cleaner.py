import numpy as np
import re
import pandas as pd
from nltk.tokenize import WordPunctTokenizer

df = pd.read_csv('dataset.csv', encoding='latin-1')

texts  = df.SentimentText
sents = df.Sentiment

lens = texts.str.len()

print("#reviews before", len(texts))


tok =  WordPunctTokenizer()

neg_dic = {"isn't":"is not",
        "aren't":"are not",
        "wasn't":"was not",
        "weren't":"were not",
        "didnt": "did not",
        "haven't":"have not",
        "hasn't":"has not",
        "hadn't":"had not",
        "won't":"will not",
        "wouldn't":"would not",
        "don't":"do not",
        "doesn't":"does not",
        "didn't":"did not",
        "can't":"can not",
        "couldn't":"could not",
        "shouldn't":"should not",
        "mightn't":"might not",
        "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(neg_dic.keys())+r')\b')

def cleaner(text):
    
    clean = text.lower()# sets all letters to be lower case
    clean = neg_pattern.sub(lambda x: neg_dic[x.group()], clean) #replaces "didn't" with "did not" etc.
    
    clean = re.sub("[^a-z\d\s]", "", clean) #removes everything that is not a letter

    words = tok.tokenize(clean) # split up words, to remove unnecessary spaces
    words = (" ".join(words)).strip() # joing splitted words to one string
    

    return words




#remove reviews shorter than 100 chars and longer than 2000 chars
texts = texts[lens>100]
texts.reset_index(drop=True, inplace=True)
texts = texts[lens<2000]
texts.reset_index(drop=True, inplace=True)

sents = sents[lens>100]
sents.reset_index(drop=True, inplace=True)
sents = sents[lens<2000]
sents.reset_index(drop=True, inplace=True)

print("#reviews after", len(texts))



clean_text =[]
counter = 0
for i in texts:
    if counter %1000 == 0:
        print(counter, 'texts have been processed')
    clean_text.append(cleaner(i))
    counter +=1
clean_df = pd.DataFrame(clean_text, columns=['texts'])


clean_df['labels'] = sents
print(clean_df.labels[:30])


clean_df.to_csv('cleaned.csv', encoding='utf-8')
csv = 'cleaned.csv'
my_df = pd.read_csv(csv,index_col=0)
print(my_df.head())


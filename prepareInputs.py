import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
'''
df = pd.read_csv('cleaned.csv',index_col=0)


print(len(df))
#print(df[df['text'].str.contains("<smile>")])

x = df.texts

y = df.labels

embedding = pickle.load(open('embd300.pickle', 'rb'))


x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=0)

length = []
for x in x_train:
    length.append(len(x.split()))
print('max:',max(length))


#'''
def toHot(Y):
    count = 0
    num=len(Y)
    hots = np.zeros((num,2))
    for i in Y:
        hots[count,int(i)]=1
        count+=1
    return hots

y_train = toHot(y_train)
y_val = toHot(y_val)
#'''


token = Tokenizer(num_words=90461, split=" ")
token.fit_on_texts(x_train)
seqs = token.texts_to_sequences(x_train)

x_train_seq = pad_sequences(seqs, maxlen=300)

seqs_val = token.texts_to_sequences(x_val)
x_val_seq = pad_sequences(seqs_val, maxlen=300)


'''
lens = []
for x in x_train:
    lens.append(len(x.split()))
print(max(lens)) = 33
#'''



numWords = 90461
embMat = np.zeros((numWords, 300))
for word, i in token.word_index.items():
    if i >= numWords:
        continue
    embedding_vector = embedding.get(word)
    if embedding_vector is not None:
        embMat[i] = embedding_vector

pickle.dump(embMat, open('embMat'+str(300)+'.pickle', 'wb'))
pickle.dump((x_train_seq, y_train, x_val_seq, y_val), open('trainValData.pickle', 'wb'))




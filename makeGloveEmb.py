import numpy as np
import pickle

embedding = {}
glove = open("glove.840B.300d.txt")
counter = 0

for line in glove:
   
   if counter%10000==0:
      print(counter)
   values = line.split()
   word = values[0]
   try:
      vector = np.asarray(values[1:], dtype='float32')
      embedding[word] = vector
   except:
      continue
   counter +=1
   if counter > 1200000:
       break
glove.close()


pickle.dump(embedding, open('embd300.pickle', 'wb'))


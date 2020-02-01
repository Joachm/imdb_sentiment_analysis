import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def checkLenghts():
    df = pd.read_csv('dataset.csv', encoding='latin-1')

    #for col in df.columns:
    #    print(col)
            #SentimentText
            #Sentiment

    sents = df.Sentiment


    """
    plt.hist(sents)
    plt.title('Distribution of Sentiments')
    plt.show()
    """

    texts = df.SentimentText


    #'''
    plt.hist(texts.str.len(), bins= 100)
    plt.xlabel("Number of Characters")
    plt.ylabel("Number of Texts")
    plt.title("Distribution of Lenghts by Character")
    plt.show()
    #'''

    lens = texts.str.len()
    print('max length')
    print(np.max(lens))
    print('mean length')
    print(np.mean(lens))
    print('min length')
    print(np.min(lens))




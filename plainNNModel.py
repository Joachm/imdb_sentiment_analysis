import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint


def runFFNN(epoch):
    embMat = pickle.load(open('embMat300.pickle', 'rb'))
    x_train_seq, y_train, x_val_seq, y_val = pickle.load(open('trainValData.pickle','rb'))



    #"""
    model = Sequential()
    e = Embedding(90461, 300, weights=[embMat], input_length=300, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.8))
    #model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="binary_crossentropy",
            optimizer='adam', metrics=['accuracy'])

    filepath="plain_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_val), epochs=epoch, batch_size=128, verbose=1, callbacks=[checkpoint])
    #"""

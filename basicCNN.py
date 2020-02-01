import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization
from keras.callbacks import ModelCheckpoint

def runBasicCNN(epoch, fltrs, ksize, strds):
    gloveD = 300
    embMat = pickle.load(open('embMat'+str(gloveD)+'.pickle', 'rb'))
    x_train_seq, y_train, x_val_seq, y_val = pickle.load(open('trainValData.pickle','rb'))



    model = Sequential()
    e = Embedding(90461, gloveD, weights=[embMat], input_length=300, trainable=False)
    model.add(e)
    model.add(Conv1D(filters=fltrs, kernel_size=ksize, padding='valid', activation="relu", strides=strds))
    model.add(Dropout(0.75))
    model.add(GlobalMaxPooling1D())
    #model.add(BatchNormalization())
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="binary_crossentropy",
            optimizer='adam', metrics=['accuracy'])


    filepath="CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


    model.fit(x_train_seq, y_train,
            validation_data=(x_val_seq, y_val), 
            epochs=epoch, 
            batch_size=128, 
            verbose=1,
            callbacks=[checkpoint])



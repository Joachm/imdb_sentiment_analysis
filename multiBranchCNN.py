import pickle
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation, concatenate
from keras.layers import Flatten, Add
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, GlobalMaxPooling1D

gloveD = 300
embMat = pickle.load(open('embMat'+str(gloveD)+'.pickle', 'rb'))
x_train_seq, y_train, x_val_seq, y_val = pickle.load(open('trainValData.pickle','rb'))


act = 'relu'

inp = Input(shape=(300,), dtype='int32')
e = Embedding(90461, gloveD, weights=[embMat], input_length=300, trainable=False)(inp)


bigram = Conv1D(filters=100, kernel_size=2, padding='valid', activation=act, strides=1)(e)
bigram = Dropout(0.75)(bigram)
bigram = GlobalMaxPooling1D()(bigram)

trigram = Conv1D(filters=100, kernel_size=3, padding='valid', activation=act, strides=1)(e)
trigram = Dropout(0.75)(trigram)
trigram = GlobalMaxPooling1D()(trigram)

fourgram = Conv1D(filters=100, kernel_size=4, padding='valid', activation=act, strides=1)(e)
fourgram = Dropout(0.75)(fourgram)
fourgram = GlobalMaxPooling1D()(fourgram)

merged = concatenate([bigram, trigram, fourgram], axis=1)


output = Dense(2, activation = 'softmax')(merged)

model = Model(inputs=[inp], outputs=[output])

model.compile(loss="categorical_crossentropy",
        optimizer='adam', metrics=['accuracy'])


filepath="branchCNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


model.fit(x_train_seq, y_train,
        validation_data=(x_val_seq, y_val), 
        epochs=155, 
        batch_size=124, 
        verbose=1,
        callbacks=[checkpoint])



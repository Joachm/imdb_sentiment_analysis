import pickle
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation
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


#model = Flatten()(e)

model = Conv1D(filters=100, kernel_size=2, padding='valid', activation=act, strides=1)(e)
model = Dropout(0.5)(model)
model = GlobalMaxPooling1D()(model)

short = model

model = Dense(256, activation = act)(model)
model = Dropout(0.8)(model)
short2 = model


model = Dense(256, activation = act)(model)
model = Dropout(0.8)(model)
short3 = model

model = Add()([model, short2])
model = Activation(act)(model)



model = Dense(256, activation = act)(model)
model = Dropout(0.8)(model)
short4 = model

model = Add()([model, short3])
model = Activation(act)(model)


model = Dense(256, activation = act)(model)
model = Dropout(0.8)(model)

model = Add()([model,short4])
model = Activation(act)(model)



model = Dense(100, activation = act)(model)
model = Dropout(0.8)(model)

model = Add()([model, short])
model = Activation(act)(model)

output = Dense(2, activation = 'softmax')(model)

model = Model(inputs=[inp], outputs=[output])

model.compile(loss="categorical_crossentropy",
        optimizer='adam', metrics=['accuracy'])


filepath="CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


model.fit(x_train_seq, y_train,
        validation_data=(x_val_seq, y_val), 
        epochs=155, 
        batch_size=124, 
        verbose=1,
        callbacks=[checkpoint])



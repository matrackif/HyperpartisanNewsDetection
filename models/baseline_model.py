from keras.models import Sequential
from keras.layers import Dense, Dropout
import doc2vec.d2v


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(doc2vec.d2v.VEC_SIZE, input_dim=doc2vec.d2v.VEC_SIZE, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(int(doc2vec.d2v.VEC_SIZE / 2), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(int(doc2vec.d2v.VEC_SIZE / 4), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(int(doc2vec.d2v.VEC_SIZE / 8), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

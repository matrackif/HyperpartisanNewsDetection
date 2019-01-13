from keras.models import Sequential
from keras.layers import Dense, Dropout
import doc2vec.d2v


def create_baseline(input_size=doc2vec.d2v.VEC_SIZE):
    # create model
    model = Sequential()
    model.add(Dense(input_size, input_dim=input_size, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(int(input_size / 2), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(int(input_size / 4), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(int(input_size / 8), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_baseline2(input_size):
    # create model
    model = Sequential()
    model.add(Dense(input_size, input_dim=input_size, kernel_initializer='normal', activation='relu', use_bias=True))
    model.add(Dropout(0.3))
    model.add(Dense(int(input_size), kernel_initializer='normal', activation='relu', use_bias=True))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

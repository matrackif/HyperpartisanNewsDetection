from keras.models import Sequential
from keras.layers import Dense
import doc2vec.d2v


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(doc2vec.d2v.VEC_SIZE, input_dim=100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

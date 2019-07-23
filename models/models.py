from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM, Embedding, Conv1D, MaxPooling1D, Flatten
from sklearn.linear_model import LogisticRegression


def get_model(model_type: str, args: dict):
    MODELS = {'dense': create_baseline2,
              'lstm': create_lstm,
              'logistic': create_logistic_regression,
              'CNN': create_CNN1D}

    if model_type in MODELS.keys() and MODELS[model_type] is not None:
        return MODELS[model_type](**args)
    return None


def create_baseline(**kwargs):
    train_x = kwargs.get('train_x', None)
    input_size = train_x.shape[1]
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


def create_baseline2(**kwargs):
    train_x = kwargs.get('train_x', None)
    input_size = train_x.shape[1]
    # create model
    NUM_NODES = 128
    model = Sequential()
    model.add(Dense(NUM_NODES, input_dim=input_size, kernel_initializer='normal', activation='relu', use_bias=True))
    model.add(BatchNormalization())
    model.add(Dense(NUM_NODES, kernel_initializer='normal', activation='relu', use_bias=True))
    model.add(BatchNormalization())
    model.add(Dense(NUM_NODES, kernel_initializer='normal', activation='relu', use_bias=True))
    model.add(BatchNormalization())
    model.add(Dense(int(NUM_NODES / 2), kernel_initializer='normal', activation='relu', use_bias=True))
    model.add(BatchNormalization())
    model.add(Dense(int(NUM_NODES / 4), kernel_initializer='normal', activation='relu', use_bias=True))
    model.add(BatchNormalization())
    model.add(Dense(int(NUM_NODES / 8), kernel_initializer='normal', activation='relu', use_bias=True))
    model.add(BatchNormalization())
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# https://nlpforhackers.io/keras-intro/
def create_lstm(**kwargs):
    model = Sequential()
    model.add(Embedding(kwargs.get('vocab_size', 0),
                        64,  # Embedding size
                        input_length=kwargs.get('input_length', 0)))
    model.add(LSTM(256))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_logistic_regression(**kwargs):
    return LogisticRegression(solver='lbfgs')


def create_CNN1D(**kwargs):
    trainX = kwargs.get('train_x', None)
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], 1
    model = Sequential()
    model.add(Conv1D(filters=1024, kernel_size=4, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=1024, kernel_size=4, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

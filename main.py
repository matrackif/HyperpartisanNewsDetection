import doc2vec.d2v
from gensim.models.doc2vec import Doc2Vec
import keras
import numpy as np
import codecs
from xml.dom import minidom
import models.baseline_model
import time
import matplotlib.pyplot as plt


if __name__ == '__main__':
    """
    doc_2_vec_model = doc2vec.d2v.Doc2VecModelBuilder('data/articles.xml')
    doc_2_vec_model.train()
    doc_2_vec_model.save_model()
    """
    doc_2_vec_model = Doc2Vec.load("data/news_articles.d2v")
    train_count = int(doc2vec.d2v.TRAIN_PERCENTAGE * doc_2_vec_model.docvecs.count)
    test_count = doc_2_vec_model.docvecs.count - train_count
    print('Train count:', train_count, 'Test count:', test_count)
    # initialize training/test x data
    train_x = np.zeros(shape=(train_count, doc_2_vec_model.docvecs[0].shape[0]))
    test_x = np.zeros(shape=(test_count, doc_2_vec_model.docvecs[0].shape[0]))
    train_y = np.zeros(shape=(train_count, 1))
    test_y = np.zeros(shape=(test_count, 1))
    with codecs.open("data/classifiedArticles.xml", 'r', encoding='utf8') as f:
        text = f.read()
        dom = minidom.parseString(text)
    articles_list = dom.getElementsByTagName('article')

    for i in range(train_count):
        train_x[i] = doc_2_vec_model.docvecs[i]
        train_y[i] = 1 if articles_list[i].getAttribute('hyperpartisan') == "true" else 0
    j = 0
    for i in range(train_count, train_count + test_count):
        test_x[j] = doc_2_vec_model.docvecs[i]
        test_y[j] = 1 if articles_list[i].getAttribute('hyperpartisan') == "true" else 0
        j += 1

    print("train_x.shape:", train_x.shape, "test_x.shape", test_x.shape)
    print("train_y.shape:", train_y.shape, "test_y.shape", test_y.shape)

    model = models.baseline_model.create_baseline()
    start_time = time.time()

    history = model.fit(train_x, train_y, epochs=200, batch_size=50, validation_data=(test_x, test_y),
                             verbose=2)
    print('Keras model finished training in %s seconds' % (time.time() - start_time))
    # plot history

    plt.figure(0)
    plt.plot(history.history['loss'], label='Training loss (error)')
    plt.plot(history.history['val_loss'], label='Test loss (error)')
    plt.title('Training/test loss of article classifier')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    prediction_train = model.predict(train_x)
    prediction_test = model.predict(test_x)
    print(prediction_test)
    for i in range(len(prediction_train)):
        prediction_train[i] = 0 if prediction_train[i] < 0.5 else 1
    for i in range(len(prediction_test)):
        prediction_test[i] = 0 if prediction_test[i] < 0.5 else 1

    print("Train correct percentage:", np.sum(prediction_train == train_y) / len(prediction_train))
    print("Test set:", np.sum(prediction_test == test_y) / len(prediction_test))

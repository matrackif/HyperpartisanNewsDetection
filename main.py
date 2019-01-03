from doc2vec.build_model import Doc2VecModelBuilder
import keras
import numpy as np

if __name__ == '__main__':
    doc_2_vec_model = Doc2VecModelBuilder('data/articles.xml')
    doc_2_vec_model.train()

    train_percentage = 0.8
    train_count = int(train_percentage * doc_2_vec_model.model.docvecs.count)
    test_count = doc_2_vec_model.model.docvecs.count - train_count
    print('Train count:', train_count, 'Test count:', test_count)
    # initialize training/test x data
    train_x = np.zeros(shape=(train_count, doc_2_vec_model.model.docvecs[0].shape[0]))
    test_x = np.zeros(shape=(test_count, doc_2_vec_model.model.docvecs[0].shape[0]))
    # TODO initialize train_y/test_y
    for i in range(train_count):
        train_x[i] = doc_2_vec_model.model.docvecs[i]
    j = 0
    for i in range(train_count, train_count + test_count):
        test_x[j] = doc_2_vec_model.model.docvecs[i]
        j += 1

    print("train_x.shape:", train_x.shape, "test_x.shape", test_x.shape)
    # print("train_x[train_count - 1]", train_x[train_count - 1], "test_x[test_count - 1]", test_x[test_count - 1])

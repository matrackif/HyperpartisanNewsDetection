from doc2vec.build_model import Doc2VecModelBuilder

if __name__ == '__main__':
    doc_2_vec_model = Doc2VecModelBuilder('data/articles.xml')
    doc_2_vec_model.train()

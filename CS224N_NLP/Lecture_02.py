# python3
# Create date : 2020-01-14
# Author: Scc_hy
# Func: NLP Learning
# Refrence: https://www.bilibili.com/video/BV1s4411N7fC


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

plt.style.use('ggplot')
# >python -m gensim.scripts.glove2word2vec -i glove.6B.100d.txt -o glove.6B.100d.word2vec.txt
glove_file = datapath(r'D:\Python_data\My_python\Video_learning\CS224N_NLP\glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
# model = KeyedVectors.load_word2vec_format(
#     r'D:\Python_data\My_python\Video_learning\CS224N_NLP\glove.6B.100d.word2vec.txt'
# )

# most_similar
model.most_similar('banana')
model.most_similar(negative='banana')

result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print('{}: {:.4f}'.format(*result[0]))


def analogy(x1, x2, y, word2vec_model=model):
    """
    x1 - x2 + y -> result
    Parameters
    ----------
    word2vec_model: model
    x1: 要减去的词语 man
    x2: 要被减去的词语 king
    y:  在该词语上增加 x2-x1
    Returns 新的词语 x1 - x2 + y
    -------

    """
    res = word2vec_model.most_similar(positive=[y, x2], negative=[x1])
    return res[0][0]


# -> beijing
analogy('japan', 'tokyo', 'china')
analogy('long', 'longer', 'high')
print(model.doesnt_match('breakfeat cereal dinner lunch'.split()))


def display_pca_scatterplot(word_model, words=None, sample=0):
    if words is None:
        words = np.random.choice(list(word_model.vocab.keys()), sample)
    else:
        words = [word for word in word_model.vocab]
    word_vectors = np.array([word_model[w] for w in words])
    two_dims = PCA().fit_transform(word_vectors)[:, :2]
    plt.figure(figsize=(6, 6))
    plt.scatter(two_dims[:, 0], two_dims[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, two_dims):
        plt.text(x + 0.05, y + 0.05, word)
    plt.show()


display_pca_scatterplot(model, sample=30)

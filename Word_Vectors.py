# python 3
# Create date 2020-12-05
# Func: kaggle word Vectors Learning
# reference: https://www.kaggle.com/matleonard/word-vectors

import numpy as np
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

nlp = spacy.load('en_core_web_log')
# Disabling other pipes because we dont need them and it'll speed up this part a bit
text = 'These vectors can be used as features for machine learning models.'

with nlp.disable_pipes():
  vectors = np.array([token.vector for token in nlp(text)]

# Classification Models
# --------------------------------------------
spam = pd.read_csv('')
# A simple and surprisingly effective approach is simply averaging the vectors for each word in the document
with nlp.disable_pipes():
  doc_vectors = np.array([nlp(text).vector for text in spam.text])
  
x_tr, x_te, y_tr, y_te = train_test_split(
  doc_vectors, spam.label, test_size=0.1, random_state = 1
)
svc = linearSVC(random_state = 1, dual=False, max_iter=10000)
svc.fit(x_tr, y_tr)
print(f'Accuracy : {svc.score(x_te, y_te)}%')

# Document Similarity
# -----------------------------------------------
def cosine_similarity(a, b):
  return a.dot(b) / np.sqrt(a.dot(a) * b.dot(b))
 
a = nlp('REPLY NOW FOR FREE TEA').vector
b = nlp('According to legend, Emperor Shen Nung discovered tea when leavers from a wild tree blew into his pot of boiling water.').vector

cosine_similarity(a, b)













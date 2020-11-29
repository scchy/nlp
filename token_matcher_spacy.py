# python 3
# author： scc_hy
# func： kaggle学习nlp
# tip：
"""
----- 
- nlp=spacy.load('en') / blanl('en')
- *nlp分词: nlp(text)
----- 
- matcher = PraseMatcher(nlp.vocab, attr='LOWER')
- matcher 增加系列+系列列表
- *匹配

"""


import spacy
nlp = spacy.load('en')

doc = nlp("Tea is healthy and calming, don't you think?")

# Tokenizing
# ---------------------------------------------------
for token in doc:
  print(token)
  
# Text preprocessing
# ----------------------------------------------------
print(f'Token \t\t\Lemma \t\tStopword'.format('Token', 'Lemma', 'stopword')
print('-'*40)
for token in doc:
  print(f"{repr(token)\t\t{token.lemma_}\t\t{token.is_stop}"}

# Pattern Matching
# -----------------------------------------------------
from spavy.matcher import PraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel')
patterns = [nlp(i) for i in terms]
## 加入匹配的 系列名(TermimologyList), 系列(patterns)
matcher.add('TermimologyList', patterns)

text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
               "photography tests pitting the iPhone 11 Pro against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.") 
matches = matcher(text_doc)
print(matches)

match_id, start, end = matches[0]
# 系列名 匹配的文本
print(nlp.vocab.strings[match_id], text_doc[start:end])


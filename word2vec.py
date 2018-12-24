import nltk
import ssl
from nltk import word_tokenize, sent_tokenize
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
from bokeh.io import output_notebook
from bokeh.plotting import show, figure


# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# #library
# nltk.download('gutenberg')

# #puctionation and tokenizer
# nltk.download('punkt')

from nltk.corpus import gutenberg
print(gutenberg.fileids())

gberg_sents = gutenberg.sents(
	fileids=[
		'bible-kjv.txt',
		'austen-emma.txt',
		'austen-persuasion.txt',
		'austen-sense.txt',
		'carroll-alice.txt'
		]
	)

#WORDCOUNT
print(len(
		gutenberg.sents(
			fileids=[
				'bible-kjv.txt',
				'austen-emma.txt',
				'austen-persuasion.txt',
				'austen-sense.txt',
				'carroll-alice.txt'
			]
		)
	)
)

model = Word2Vec(
	sentences=gberg_sents,
	size=64,
	sg=1,
	window=10,
	min_count=5,
	seed=42,
	workers=2
	)

print(model.wv.similarity('father', 'house'))
print(model.wv.similarity('father', 'mother'))







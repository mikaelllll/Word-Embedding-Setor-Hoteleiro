# Word-Embedding-Setor-Hoteleiro
Modelos gerados para o trabalho sobre word embeddings utilizando domínio do setor hoteleiro.

# Utilizar os modelos Word2Vec no Gensim: #
from gensim.models import Word2Vec
word2v_model = Word2Vec.load(filename)

# Utilizar os modelos FastText no Gensim: #
from gensim.models import FastText
fast_model = FastText.load(filename)

# Utilizar os modelos Wang2Vec no Gensim: #
from gensim.models import KeyedVectors
wang2v_model = word2vec.KeyedVectors.load_word2vec_format(filename, binary=True, encoding='utf-8')

# Utilizar o modelo Glove no Gensim: #
from gensim.scripts.glove2word2vec import glove2word2vec
tmp_file = "glove2word2vec.txt"
glove2word2vec(filename, tmp_file)
glove_model = KeyedVectors.load_word2vec_format(tmp_file)

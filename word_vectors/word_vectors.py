from nltk.tokenize import word_tokenize
import gensim
import gevent
import faiss
import numpy as np
import sys
import os
import pickle
from nltk.corpus import reuters
import nltk.data
from sklearn.datasets import fetch_20newsgroups

INDEX_FILE = 'index/faiss.index'

def pipeline(*funcs):
    def _pipeline(stream):
        for func in funcs:
            stream = func(stream)
        return stream

    return _pipeline

def to_raw(docs):
    raw = []
    for i in range(len(docs)):
        raw.append(reuters.raw(docs[i]))
    return raw

def formatted_raw(documents):
    formatted = []
    for doc in documents:
        output = ' '.join(doc.split('\n'))
        formatted.append(' '.join([token for token in output.split(' ') if u'' != token]).lower())

    return formatted

def vectorize(documents, model, raw_documents):
    embeddings = []
    valid_documents = []
    for index, document in enumerate(documents):
        try:
            embedding = create_embedding(document, model)
            embeddings.append(embedding)
            valid_documents.append(raw_documents[index])
        except:
            pass
    embeddings = np.array(embeddings).astype('float32')
    return embeddings, valid_documents

def create_embedding(text, model):
    words = word_tokenize(text)
    return create_mean_embedding(words, model)

def create_max_embedding(words, model):
    return np.amax([model[word] for word in words if word in model], axis=0)

def create_mean_embedding(words, model):
    vectors = np.array([model[word] for word in words if word in model])
    if len(vectors) == 0:
        raise ValueError
    return np.mean(vectors, axis=0)

def index_lookup(vector, index, dataset, n=5):
    result = index.search(np.expand_dims(vector, axis=0), n)
    texts = [dataset[int(idx)] for idx in result[1][0]]

    return texts

def interactive(model, index, documents):
    print("Starting interactive console...")
    while True:
        query = raw_input("> Query: ")
        if query == "exit":
            return
        vectorized = create_embedding(query, model)
        found_docs = index_lookup(vectorized, index, documents)
        for doc in found_docs:
            print('===========')
            print(doc)

model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True)
documents = reuters.fileids()
raw_documents = to_raw(documents)
parsed_documents = formatted_raw(raw_documents)
embeddings, valid_documents = vectorize(parsed_documents, model, raw_documents)
print("Unable to create vectors for {} / {} documents".format(
        len(documents) - len(valid_documents),
        len(documents)
    ))

if sys.argv[1] == 'build_index':
    # Create Index
    index = faiss.IndexFlatL2(300)

    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print("Built index for {} documents".format(len(valid_documents)))

interactive(
    model,
    faiss.read_index(INDEX_FILE),
    valid_documents,
)

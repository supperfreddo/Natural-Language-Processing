import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from glove import Corpus, Glove
from nltk.tokenize import word_tokenize

# Download the GloVe word embeddings
nltk.download('glove')

# Load the GloVe embeddings from NLTK
from nltk.data import find
glove_embedding = find('models/glove.6B.100d.txt')

# Load the GloVe word vectors
word_vectors = {}
with open(glove_embedding, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_vectors[word] = coefs

# Example document
document = "This is an example document for representation using GloVe embeddings"

# Tokenize the document
tokens = word_tokenize(document.lower())

# Filter stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Get GloVe embeddings for the document tokens
document_embeddings = []
for token in filtered_tokens:
    if token in word_vectors:
        document_embeddings.append(word_vectors[token])

# Calculate the average embedding for the document
if document_embeddings:
    document_embedding = np.mean(document_embeddings, axis=0)
    print(f"Shape of document embedding: {document_embedding.shape}")
    # You can use the document embedding for further analysis or similarity calculations
else:
    print("No valid word embeddings found for the document.")
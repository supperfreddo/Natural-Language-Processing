import nltk
import spacy
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import CountVectorizer

#download the corpus
nltk.download('gutenberg')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('stopwords')

#normalize the text
words = gutenberg.words()
words = [word.lower() for word in words]
#remove punctuation
words = [word for word in words if word.isalpha()]
#remove stop words
stop_words = set(nltk.corpus.stopwords.words('english'))
words = [word for word in words if word not in stop_words]
#stemming
porter = nltk.PorterStemmer()
words = [porter.stem(word) for word in words]

#vector representations of the raw documents using the bag-of-words model
cv = CountVectorizer()
bow = cv.fit_transform(gutenberg.words())
print(bow.shape)
# print(bow.toarray()) #seems to be to large to print

#vector representations of the preprocessed documents using the bag-of-words model
bow = cv.fit_transform(words)
print(bow.shape)
# print(bow.toarray()) #seems to be to large to print
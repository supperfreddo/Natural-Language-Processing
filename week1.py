import nltk
import spacy
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg

#download the corpus
nltk.download('gutenberg')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

#count the number of documents or files in the corpus
print("Number of documents in the corpus: " + str(len(gutenberg.fileids())))

#count the number of words
words = gutenberg.words()
print("Number of words in the corpus: " + str(len(words)))

#count the number of unique words
unique_words = set(words)
print("Unique words in the corpus: " + str(len(unique_words)))

#make a plot of the distribution of the number of words per documents
plt.hist([len(gutenberg.words(fileid)) for fileid in gutenberg.fileids()])
plt.xlabel('Words per document')
plt.ylabel('Number of documents')
plt.title('Distribution of the number of words per documents')
plt.show()

#print the top 10 most common words
fdist = nltk.FreqDist(words)
print("Top 10 most common words: " + str(fdist.most_common(10)))

#apply part-of-speech tagging and count the number of occurrences of each tag
tagged_words = nltk.pos_tag(words, tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in tagged_words)
print("Number of occurrences of each tag: " + str(tag_fd.most_common()))
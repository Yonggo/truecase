"""
This script trains the TrueCase System
"""
import pickle
import time

from EvaluateTruecaser import defaultTruecaserEvaluation
from TrainFunctions import *

#nltk.download('punkt')

uniDist = nltk.FreqDist()
backwardBiDist = nltk.FreqDist() 
forwardBiDist = nltk.FreqDist() 
trigramDist = nltk.FreqDist() 
wordCasingLookup = {}
        
"""
There are three options to train the true caser:
1) Use the sentences in NLTK
2) Use the train.txt.back file. Each line must contain a single sentence. Use a large corpus, for example Wikipedia
3) Use Bigrams + Trigrams count from the website http://www.ngrams.info/download_coca.asp

The more training data, the better the results
"""
         

# :: Option 1: Train it based on NLTK corpus ::
"""
print "Update from NLTK Corpus"
NLTKCorpus = brown.sents()+reuters.sents()+nltk.corpus.semcor.sents()+nltk.corpus.conll2000.sents()+nltk.corpus.state_union.sents()
updateDistributionsFromSentences(NLTKCorpus, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
"""

# :: Option 2: Train it based the train.txt.back file ::
#Uncomment, if you want to train from train.txt.back
def fetchToken(sentence, language):
    print(sentence)
    return nltk.word_tokenize(sentence, "german")

train_file = "Dataset/deu_news_2018_1M-sentences.txt"
print("Update from {0} file".format(train_file))
start_counter = time.time()
sentences = []
for line in open(train_file, encoding="utf8"):
    sentences.append(line.strip())

#size_sentences = len(sentences)
#size_for_train = int(size_sentences * 0.95)
tokens = [fetchToken(sentence, "german") for sentence in sentences]
updateDistributionsFromSentences(tokens, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
end_counter = time.time()
   
# :: Option 3: Train it based ngrams tables from http://www.ngrams.info/download_coca.asp ::    
""" #Uncomment, if you want to train from train.txt.back
print "Update Bigrams / Trigrams"
updateDistributionsFromNgrams('ngrams/w2.txt', 'ngrams/w3.txt', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
"""

f = open('distributions.obj', 'wb')
pickle.dump(uniDist, f, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(backwardBiDist, f, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(forwardBiDist, f, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(trigramDist, f, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(wordCasingLookup, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

        
# :: Correct sentences ::
defaultTruecaserEvaluation(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist, train_file)

print('Training time: {:.2f}s'.format(end_counter - start_counter))

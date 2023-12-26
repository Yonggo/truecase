import fileinput
import pickle

import nltk

from Truecaser import *

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('files', metavar='FILE', nargs='*', help='files to truecase, if empty, STDIN is used')
    #parser.add_argument('-d', '--distribution_object', help='language distribution file', type=os.path.abspath, required=True)
    #args = parser.parse_args()

    f = open("distributions.obj", 'rb')
    uniDist = pickle.load(f)
    backwardBiDist = pickle.load(f)
    forwardBiDist = pickle.load(f)
    trigramDist = pickle.load(f)
    wordCasingLookup = pickle.load(f)
    f.close()
    
    for sentence in fileinput.input(files="text.txt"):
        tokensCorrect = nltk.word_tokenize(sentence)
        tokens = [token.lower() for token in tokensCorrect]
        tokensTrueCase = getTrueCase(tokens, 'title', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
        print(" ".join(tokensTrueCase))


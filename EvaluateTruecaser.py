import nltk
import pickle
import time
import sys

from Truecaser import *
from tqdm import tqdm

verbose = 1

def gprint(s):
    if verbose:
        if isinstance(s, str):
            s = s.replace('\n', '<n>')
        print(s, file=sys.stderr)

def evaluateTrueCaser(testSentences, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist, train_file_path):
    train_file = train_file_path[:train_file_path.rfind(".")]
    f_log = open(train_file + "-" + "log_wrong_prediction.txt", "w", encoding="utf8")
    wrong_pred = []
    correctTokens = 0
    predicted_true = 0
    actual_true = 0
    true_positive = 0
    totalTokens = 0

    progress = range(len(testSentences))
    progress = tqdm(progress, desc="Prediction Truecase")

    for i in progress:
        tokensCorrect = nltk.word_tokenize(testSentences[i])
        tokens = [token.lower() for token in tokensCorrect]
        tokensTrueCased = getTrueCase(tokens, 'title', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)

        #perfectMatch = True

        for idx in range(len(tokensCorrect)):
            totalTokens += 1
            if tokensTrueCased[idx] != tokensTrueCased[idx].lower():
                predicted_true += 1

            if tokensCorrect[idx] != tokensCorrect[idx].lower():
                actual_true += 1

            if tokensCorrect[idx] == tokensTrueCased[idx]:
                correctTokens += 1
                if tokensCorrect[idx] != tokensCorrect[idx].lower():
                    true_positive += 1
            else:
                #perfectMatch = False
                cor_to_wrong = tokensCorrect[idx] + " => " + tokensTrueCased[idx]
                wrong_pred.append(cor_to_wrong + "\n")
                wrong_pred.append(" ".join(tokensCorrect) + "\n")
                wrong_pred.append(" ".join(tokensTrueCased) + "\n\n")

        """"
        if not perfectMatch:
            gprint(tokensCorrect)
            gprint(tokensTrueCased)
            gprint("-------------------")
        """

        #gprint("------------------- Current Accuracy: %.2f%%" % (correctTokens / float(totalTokens) * 100))

    print("writting logs...")
    f_log.write("".join(wrong_pred))
    f_log.close()
    acc = correctTokens / float(totalTokens) * 100
    try:
        P = float(true_positive) / predicted_true
        R = float(true_positive) / actual_true
        F = 2 * P * R / (P + R)
    except:
        P = 0
        R = 0
        F = 0
    print("=============== Result ===============")
    print('Accuracy: {:.2f}'.format(acc))
    print('Precision: {:.2f}'.format(P * 100))
    print('Recall: {:.2f}'.format(R * 100))
    print('F1: {:.2f}'.format(F * 100))
    print("Total Tokens: {}".format(totalTokens))
    print("======================================")


def defaultTruecaserEvaluation(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist, train_file_path):
    start_counter = time.time()
    text_file = "Dataset/wiki_300K/eval.txt"
    sentences = []
    for line in open(text_file, encoding="utf8"):
        sentences.append(line.strip())

    evaluateTrueCaser(sentences, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist, text_file)
    end_counter = time.time()
    print('Prediction time: {:.2f}s'.format(end_counter - start_counter))


if __name__ == "__main__":
    start_latency_time_counter = time.time()
    f = open('distributions.obj', 'rb')
    uniDist = pickle.load(f)
    backwardBiDist = pickle.load(f)
    forwardBiDist = pickle.load(f)
    trigramDist = pickle.load(f)
    wordCasingLookup = pickle.load(f)
    f.close()
    end_latency_time_counter = time.time()
    print('latency time: {:.2f}s'.format(end_latency_time_counter - start_latency_time_counter))

    train_file = "Dataset/deu_news_2018_1M-sentences.txt"
    start_counter = time.time()
    defaultTruecaserEvaluation(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist, train_file)
    end_counter = time.time()
    print('Processing time: {:.2f}s'.format(end_counter - start_counter))

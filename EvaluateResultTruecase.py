import nltk
from tqdm import tqdm


def evaluate_truecase(truecasedSentences, referenceSentences):
    wrong_pred = []
    correctTokens = 0
    predicted_true = 0
    actual_true = 0
    true_positive = 0
    totalTokens = 0

    progress = range(len(referenceSentences))
    progress = tqdm(progress, desc="Evaluation")

    for i in progress:
        tokensCorrect = nltk.word_tokenize(referenceSentences[i])
        tokensTrueCased = nltk.word_tokenize(truecasedSentences[i])

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


if __name__ == "__main__":
    with open("result_truecase.txt", "r", encoding="utf8") as file:
        truecasedSentences = file.readlines()
    with open("Dataset/de_train.ft.txt", "r", encoding="utf8") as file:
        referenceSentences = file.readlines()

    evaluate_truecase(truecasedSentences, referenceSentences)

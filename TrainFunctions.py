import nltk
from tqdm import tqdm


def getCasing(word):
    """ Returns the casing of a word"""
    if len(word) == 0:
        return 'other'
    elif word.isdigit(): #Is a digit
        return 'numeric'
    elif word.islower(): #All lower case
        return 'allLower'
    elif word.isupper(): #All upper case
        return 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        return 'initialUpper'
    
    return 'other'


def checkSentenceSanity(sentence):
    """ Checks the sanity of the sentence. If the sentence is for example all uppercase, it is recjected"""
    caseDist = nltk.FreqDist()
    
    for token in sentence:
        caseDist[getCasing(token)] += 1
    
    if caseDist.most_common(1)[0][0] != 'allLower':        
        return False
    
    return True

def updateDistributionsFromSentences(text, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    """
    Updates the NLTK Frequency Distributions based on a list of sentences.
    text: Array of sentences.
    Each sentence must be an array of Tokens.
    """
    # :: Create unigram lookup ::
    progress = range(len(text))
    progress = tqdm(progress, desc="Uni-gram Training")
    for i in progress:
        if not checkSentenceSanity(text[i]):
            continue
        
        for tokenIdx in range(1, len(text[i])):
            word = text[i][tokenIdx]
            uniDist[word] += 1
                        
            if word.lower() not in wordCasingLookup:
                wordCasingLookup[word.lower()] = set()
            
            wordCasingLookup[word.lower()].add(word)
    
    # :: Create backward + forward bigram lookup ::
    progress = range(len(text))
    progress = tqdm(progress, desc="Bi-gram Training")
    for i in progress:
        if not checkSentenceSanity(text[i]):
            continue
        
        for tokenIdx in range(2, len(text[i])): #Start at 2 to skip first word in sentence
            word = text[i][tokenIdx]
            wordLower = word.lower()
            
            if wordLower in wordCasingLookup and len(wordCasingLookup[wordLower]) > 1: #Only if there are multiple options
                prevWord = text[i][tokenIdx-1]
                
                backwardBiDist[prevWord.lower()+"_"+word] +=1
                backwardBiDist[prevWord + "_" + word] += 1

        for tokenIdx in range(1, len(text[i])-1):
            word = text[i][tokenIdx]
            wordLower = word.lower()

            if wordLower in wordCasingLookup and len(wordCasingLookup[wordLower]) > 1: #Only if there are multiple options
                nextWord = text[i][tokenIdx+1]

                forwardBiDist[word + "_" + nextWord.lower()] += 1
                forwardBiDist[word + "_" + nextWord] += 1
                    
    # :: Create trigram lookup ::
    progress = range(len(text))
    progress = tqdm(progress, desc="Tri-gram Training")
    for i in progress:
        if not checkSentenceSanity(text[i]):
            continue
        
        for tokenIdx in range(2, len(text[i])-1): #Start at 2 to skip first word in sentence
            prevWord = text[i][tokenIdx-1]
            curWord = text[i][tokenIdx]
            curWordLower = curWord.lower()
            #nextWordLower = sentence[tokenIdx+1].lower()
            nextWord = text[i][tokenIdx + 1] #changed by Yong to use instead of nextWordLower
            
            if curWordLower in wordCasingLookup and len(wordCasingLookup[curWordLower]) > 1: #Only if there are multiple options
                trigramDist[prevWord.lower()+"_"+curWord+"_"+nextWord.lower()] += 1
                trigramDist[prevWord + "_" + curWord + "_" + nextWord] += 1

        # added by Yong to consider the tri-gram used by paper of Lita
        for tokenIdx in range(3, len(text[i])):  # Start at 3 to skip first and second word in sentence
            prevPrevWord = text[i][tokenIdx - 2]
            prevWord = text[i][tokenIdx - 1]
            curWord = text[i][tokenIdx]
            curWordLower = curWord.lower()

            if curWordLower in wordCasingLookup and len(wordCasingLookup[curWordLower]) > 1:  # Only if there are multiple options
                trigramDist[prevPrevWord.lower() + "_" + prevWord.lower() + "_" + curWord] += 1
                trigramDist[prevPrevWord + "_" + prevWord + "_" + curWord] += 1

        # added by Yong to consider forwarding case
        for tokenIdx in range(1, len(text[i])-2):  # Start at 2 to skip first word in sentence
            curWord = text[i][tokenIdx]
            nextWord = text[i][tokenIdx + 1]
            nextNextWord = text[i][tokenIdx + 2]
            curWordLower = curWord.lower()

            if curWordLower in wordCasingLookup and len(wordCasingLookup[curWordLower]) > 1:  # Only if there are multiple options
                trigramDist[curWord + "_" + nextWord.lower() + "_" + nextNextWord.lower()] += 1
                trigramDist[curWord + "_" + nextWord + "_" + nextNextWord] += 1
 

def updateDistributionsFromNgrams(bigramFile, trigramFile, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    """
    Updates the FrequencyDistribitions based on an ngram file,
    e.g. the ngram file of http://www.ngrams.info/download_coca.asp
    """
    for line in open(bigramFile):
        splits = line.strip().split('\t')
        cnt, word1, word2 = splits
        cnt = int(cnt)
        
        # Unigram
        if word1.lower() not in wordCasingLookup:
            wordCasingLookup[word1.lower()] = set()
            
        wordCasingLookup[word1.lower()].add(word1)
        
        if word2.lower() not in wordCasingLookup:
            wordCasingLookup[word2.lower()] = set()
            
        wordCasingLookup[word2.lower()].add(word2)
        
        
        uniDist[word1] += cnt
        uniDist[word2] += cnt
        
        # Bigrams
        backwardBiDist[word1+"_"+word2] +=cnt
        forwardBiDist[word1+"_"+word2.lower()] += cnt
        
        
    #Tigrams
    for line in open(trigramFile):
        splits = line.strip().split('\t')
        cnt, word1, word2, word3 = splits
        cnt = int(cnt)
        
        trigramDist[word1+"_"+word2+"_"+word3.lower()] += cnt
        
        

        

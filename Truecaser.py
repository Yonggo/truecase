import pickle
import string
import math
import time

import nltk

"""
This file contains the functions to truecase a sentence.
"""

def getScore(prevPrevToken, prevToken, possibleToken, nextToken, nextNextToken, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    pseudoCount = 5.0
    
    #Get Unigram Score
    nominator = uniDist[possibleToken]+pseudoCount    
    denominator = 0    
    for alternativeToken in wordCasingLookup[possibleToken.lower()]:
        denominator += uniDist[alternativeToken]+pseudoCount
        
    unigramScore = nominator / denominator
        
        
    #Get Backward Score  
    bigramBackwardScore = 1
    if prevToken != None:  
        nominator = backwardBiDist[prevToken+'_'+possibleToken]+pseudoCount
        denominator = 0    
        for alternativeToken in wordCasingLookup[possibleToken.lower()]:
            denominator += backwardBiDist[prevToken+'_'+alternativeToken]+pseudoCount
            
        bigramBackwardScore = nominator / denominator
        
    #Get Forward Score  
    bigramForwardScore = 1
    if nextToken != None:  
        nextToken = nextToken.lower() #Ensure it is lower case
        nominator = forwardBiDist[possibleToken+"_"+nextToken]+pseudoCount
        denominator = 0    
        for alternativeToken in wordCasingLookup[possibleToken.lower()]:
            denominator += forwardBiDist[alternativeToken+"_"+nextToken]+pseudoCount
            
        bigramForwardScore = nominator / denominator
        
        
    #Get Trigram Score  
    trigramScore = 1
    if prevToken != None and nextToken != None:
        #nextToken = nextToken #Ensure it is lower case #commented by Yong as nextToken is not anymore stored in lower
        nominator = trigramDist[prevToken+"_"+possibleToken+"_"+nextToken]+pseudoCount
        denominator = 0    
        for alternativeToken in wordCasingLookup[possibleToken.lower()]:
            denominator += trigramDist[prevToken+"_"+alternativeToken+"_"+nextToken]+pseudoCount
            
        trigramScore = nominator / denominator

    # Get Trigram-Backward Score
    trigramBackwardScore = 1
    if prevPrevToken != None and prevToken != None:
        nominator = trigramDist[prevPrevToken + "_" + prevToken + "_" + possibleToken] + pseudoCount
        denominator = 0
        for alternativeToken in wordCasingLookup[possibleToken.lower()]:
            denominator += trigramDist[prevPrevToken + "_" + prevToken + "_" + alternativeToken] + pseudoCount

        trigramBackwardScore = nominator / denominator

    # Get Trigram-Forward Score
    trigramForwardScore = 1
    if nextNextToken != None and nextToken != None:
        nominator = trigramDist[possibleToken + "_" + nextToken + "_" + nextNextToken] + pseudoCount
        denominator = 0
        for alternativeToken in wordCasingLookup[possibleToken.lower()]:
            denominator += trigramDist[alternativeToken + "_" + nextToken + "_" + nextNextToken] + pseudoCount

        trigramForwardScore = nominator / denominator
        
    result = (math.log(unigramScore)
              + math.log(bigramBackwardScore) + math.log(bigramForwardScore)
              + math.log(trigramScore) + math.log(trigramBackwardScore) + math.log(trigramForwardScore))
    #print "Scores: %f %f %f %f = %f" % (unigramScore, bigramBackwardScore, bigramForwardScore, trigramScore, math.exp(result))
  
  
    return result

def getTrueCase(tokens, outOfVocabularyTokenOption, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist) -> str:
    """
    Returns the true case for the passed tokens.
    @param tokens: Tokens in a single sentence
    @param outOfVocabulariyTokenOption:
        title: Returns out of vocabulary (OOV) tokens in 'title' format
        lower: Returns OOV tokens in lower case
        as-is: Returns OOV tokens as is
    """
    tokensTrueCase = []
    for tokenIdx in range(len(tokens)):
        token = tokens[tokenIdx]
        if token in string.punctuation or token.isdigit():
            tokensTrueCase.append(token)
        else:
            if token in wordCasingLookup:
                if len(wordCasingLookup[token]) == 1:
                    tokensTrueCase.append(list(wordCasingLookup[token])[0])
                else:
                    prevPrevToken = tokensTrueCase[tokenIdx-2] if tokenIdx > 0  else None
                    prevToken = tokensTrueCase[tokenIdx-1] if tokenIdx > 0  else None
                    nextToken = tokens[tokenIdx+1] if tokenIdx < len(tokens)-1 else None
                    nextNextToken = tokens[tokenIdx+2] if tokenIdx < len(tokens)-2 else None
                    
                    bestToken = None
                    highestScore = float("-inf")
                    
                    for possibleToken in wordCasingLookup[token]:
                        score = getScore(prevPrevToken, prevToken, possibleToken, nextToken, nextNextToken, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
                           
                        if score > highestScore:
                            bestToken = possibleToken
                            highestScore = score
                        
                    tokensTrueCase.append(bestToken)
                    
                if tokenIdx == 0:
                    tokensTrueCase[0] = tokensTrueCase[0].title();
                    
            else: #Token out of vocabulary
                if outOfVocabularyTokenOption == 'title':
                    tokensTrueCase.append(token.title())
                elif outOfVocabularyTokenOption == 'lower':
                    tokensTrueCase.append(token.lower())
                else:
                    tokensTrueCase.append(token) 
    
    return tokensTrueCase


if __name__ == "__main__":
    print("Loading model...")
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
    sentence = input("Enter sentence: ")
    start = time.time()
    tokens = nltk.word_tokenize(sentence)
    tokensTrueCased = getTrueCase(tokens, 'title', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
    result = " ".join(tokensTrueCased)
    end = time.time()
    print("Processing time: {:.3f}s".format(end - start))
    print("=============== Result ===============")
    print(result)
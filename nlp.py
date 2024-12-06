#1
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

def pre_process(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower()  not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

text = "Neural Language Processing(NLP) is a field of artificial intellignece that focuses on"
preprocessed_text = pre_process(text)
print("Preprocessed Text is: ", preprocessed_text)

#2
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist, MLEProbDist
from nltk.tokenize import word_tokenize  # Corrected import

nltk.download('punkt')

sentence = "I Love programming in Python and I enjoy learning languages"

tokens = word_tokenize(sentence.lower())

unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

unigram_freq = FreqDist(unigrams)
bigram_freq = FreqDist(bigrams)
trigram_freq = FreqDist(trigrams)

unigram_prob_dist = MLEProbDist(unigram_freq)
bigram_prob_dist = MLEProbDist(bigram_freq)
trigram_prob_dist = MLEProbDist(trigram_freq)

def get_ngram_prob(ngram, prob_dist):
    if not isinstance(ngram, tuple):
        ngram = tuple(ngram)
    return prob_dist.prob(ngram)

print("\nUnigram Probabilities")
for unigram in unigram_freq:
    print(f"{unigram[0]}: {get_ngram_prob(unigram, unigram_prob_dist)}")

print("\nBigram Probabilities")
for bigram in bigram_freq:
    print(f"{bigram[0]}: {get_ngram_prob(bigram, bigram_prob_dist)}")

print("\nTrigram Probabilities")
for trigram in trigram_freq:
    print(f"{trigram[0]}: {get_ngram_prob(trigram, trigram_prob_dist)}")

#3
def min_edit_distance(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)
       
    dp = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]

    for i in range(len_str1 + 1):
        dp[i][0] = i 
    for j in range(len_str2 + 1):
        dp[0][j] = j 
        
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,  
                    dp[i - 1][j - 1] + 1 
                )
    
    return dp[len_str1][len_str2]

test_cases = [
    ("kitten", "sitting"),
    ("flaw", "lawn"), 
    ("intention", "execution"),
    ("horse","ros"),
    ("", "abc"),
    ("abc", "abc"),     
    ("abcd", "abcd"),      
    ("kitten", "kitten")     
]

for str1, str2 in test_cases:
    distance = min_edit_distance(str1, str2)
    print(f"min edit distance between ('{str1}' and '{str2}') : {distance}")


#7
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
def get_synonyms_antonyms(word):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    
    return set(synonyms), set(antonyms)

word = input('Enter the word to get antonym and synonym:').strip()
synonyms, antonyms = get_synonyms_antonyms(word)

print(f"Synonyms of '{word}': {synonyms}")
print(f"Antonyms of '{word}': {antonyms}")


#4
import nltk
from nltk import CFG
from nltk.tree import Tree
print(nltk.__version__)

grammar = CFG.fromstring("""
   S -> NP VP
   VP -> V NP|V NP PP
   PP -> P NP
   V -> "saw"|"ate"|"walked"
   NP -> "Rahil"|"Bob"|Det N|Det N PP
   Det ->"a"|"an"|"the"|"my"|"his"
   N -> "dog"|"cat"|"telescope"|"park"|"Moon"|"terrace"
   P -> "in" | "on" | "by" | "with" | "from"
""")

sentence = "Rahil saw the Moon with the telescope from his terrace".split()

print("Bottom up parsing: ")
bottom_up_parser = nltk.ChartParser(grammar)
bottom_up_trees=[]
for tree in bottom_up_parser.parse(sentence):
    print(tree)
    tree.pretty_print()
    bottom_up_trees.append(tree)
   
if bottom_up_trees:
    for tree in bottom_up_trees:
        tree.draw()

print("Top_Down Parsing")
top_down_parser = nltk.RecursiveDescentParser(grammar)  
top_down_trees = []
try:
    for tree in top_down_parser.parse(sentence): 
        print(tree)
        tree.pretty_print()
        top_down_trees.append(tree)
except ValueError as e:
    print(f"Error in parsing: {e}")

if top_down_trees:
    for tree in top_down_trees:
        tree.draw()

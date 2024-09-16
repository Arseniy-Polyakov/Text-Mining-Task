import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words("Russian")
lemmatizer = WordNetLemmatizer()

def jaccard_extraction(text_1, text_2):
    text_1 = [lemmatizer.lemmatize(item.lower()) for item in nltk.word_tokenize(text_1) if item not in stop_words]
    text_2 = [lemmatizer.lemmatize(item.lower()) for item in nltk.word_tokenize(text_2) if item not in stop_words]
    common = len(set(text_1).intersection(set(text_2)))
    unique = len(set(text_1).union(set(text_2)))
    jaccard_sum = common / unique
    return jaccard_sum 

def find_keys(dict, value):
    keys_list = [key for key, val in dict.items() if val == value]
    return keys_list


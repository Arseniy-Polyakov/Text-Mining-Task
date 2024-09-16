# Importing NLP and Data Analytics Libraries 
import re

import nltk
# from nltk.corpus import stopwords
from nltk.probability import FreqDist 
# stop_words = stopwords.words("Russian")
import pandas as pd
import numpy as np
import spacy 
nlp = spacy.load("ru_core_news_md")
from textblob import TextBlob

from autocorrect import spell
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf_model = TfidfVectorizer()

from natasha import NewsNERTagger, Doc, emb, NewsEmbedding, Segmenter, NewsMorphTagger, MorphVocab
emb = NewsEmbedding() 
segmenter = Segmenter() 
morph_vocab = MorphVocab()
morph_tagger = NewsMorphTagger(emb)
ner_tagger = NewsNERTagger(emb)


from dash import Dash, html, dcc, Input, Output
import plotly.express as px 
import plotly.figure_factory as ff
import plotly.graph_objects as go

from jaccard import jaccard_extraction, find_keys
 
# Opening the File and Mining the Text
with open("NovayaGazeta.txt", "rt", encoding="utf-8") as file:
    text_str = file.read()
    tokens = text_str.split()

with open("intro.txt", "rt", encoding="utf-8") as file:
    intro = file.read()
    intro_no_punct = re.sub(r"[^\w\n\- ]", "", intro)

with open("main.txt", "rt", encoding="utf-8") as file:
    main = file.read()
    main_no_punct = re.sub(r"[^\w\n\- ]", "", main)

with open("final.txt", "rt", encoding="utf-8") as file:
    final = file.read()
    final_no_punct = re.sub(r"[^\w\n\- ]", "", final)

with open("stopwords.txt", "rt", encoding="utf-8") as file:
    stop_words = file.read().split()

with open("NovayaGazetaEnglish.txt", "rt", encoding="utf-8") as file:
    english_text = file.read()

# Removing from Punctuation Marks
text_without_punct = re.sub(r"[^\w\n\- ]", "", text_str)

# AutoCorrection
text_without_punct_correct = spell(text_without_punct)

# Stopwords removing
text_without_punct_list = text_without_punct_correct.split()
tokens_without_stopwords = [text_without_punct_list[i].lower() for i in range(len(text_without_punct_list)) if text_without_punct_list[i] not in stop_words]

# Segmentation
sentences = nltk.sent_tokenize(text_str)
tokens_per_sentence = len(tokens) / len(sentences)
tokens_per_sentence_without_stopwords = len(tokens_without_stopwords) / len(sentences)

# Dividing into Introduction, the Main Body and the Conclusion
intro_sent = nltk.sent_tokenize(intro)
intro_tokens = nltk.word_tokenize(intro_no_punct)
intro_tokens_without_stopwords = [token.lower() for token in intro_tokens if token not in stop_words]

main_sent = nltk.sent_tokenize(main)
main_tokens = nltk.word_tokenize(main_no_punct)
main_tokens_without_stopwords = [token.lower() for token in main_tokens if token not in stop_words]

final_sent = nltk.sent_tokenize(final)
final_tokens = nltk.word_tokenize(final_no_punct)
final_tokens_without_stopwords = [token.lower() for token in final_tokens if token not in stop_words]

descriptive_pd = pd.DataFrame({"Sentences": len(sentences), 
                               "Tokens": len(tokens), 
                               "Tokens without Stopwords": len(tokens_without_stopwords),  
                               "Tokens per Sentence": tokens_per_sentence, 
                               "Tokens per Sentence without Stopwords": tokens_per_sentence_without_stopwords}, 
                               index=[0])

descriptive_intro_pd = pd.DataFrame({"Sentences": len(intro_sent), 
                                    "Tokens": len(intro_tokens), 
                                    "Tokens without Stopwords": len(intro_tokens_without_stopwords), 
                                    "Tokens per Sentence": len(intro_tokens) / len(intro_sent), 
                                    "Tokens per Sentence without Stopwords": len(intro_tokens_without_stopwords) / len(intro_sent)}, 
                                    index=[0])

descriptive_main_pd = pd.DataFrame({"Sentences": len(main_sent), 
                                    "Tokens": len(main_tokens), 
                                    "Tokens without Stopwords": len(main_tokens_without_stopwords), 
                                    "Tokens per Sentence": len(main_tokens) / len(main_sent), 
                                    "Tokens per Sentence without Stopwords": len(main_tokens_without_stopwords) / len(main_sent)}, 
                                    index=[0])

descriptive_final_pd = pd.DataFrame({"Sentences": len(final_sent), 
                                    "Tokens": len(final_tokens), 
                                    "Tokens without Stopwords": len(final_tokens_without_stopwords), 
                                    "Tokens per Sentence": len(final_tokens) / len(final_sent), 
                                    "Tokens per Sentence without Stopwords": len(final_tokens_without_stopwords) / len(final_sent)}, 
                                    index=[0])

descriptive_tokens_pd = pd.DataFrame({"Sentences": len(sentences), 
                               "Tokens": len(tokens), 
                               "Tokens without Stopwords": len(tokens_without_stopwords),  
                               "Tokens per Sentence": tokens_per_sentence, 
                               "Tokens per Sentence without Stopwords": tokens_per_sentence_without_stopwords}, 
                               index=[0])
# Lemmatization
doc = nlp(" ".join(tokens_without_stopwords))
lemmas = list(set([token.lemma_ for token in doc]))
text_lemmatized = " ".join(lemmas)

doc = nlp(" ".join(intro_tokens_without_stopwords))
lemmas = list(set([token.lemma_ for token in doc]))
intro = " ".join(lemmas)

doc = nlp(" ".join(main_tokens_without_stopwords))
lemmas = list(set([token.lemma_ for token in doc]))
main = " ".join(lemmas)

doc = nlp(" ".join(final_tokens_without_stopwords))
lemmas = list(set([token.lemma_ for token in doc]))
final = " ".join(lemmas)

# Finding Jaccard Similarity Coefficient 
jaccard_intro_main = jaccard_extraction(intro, main)
jaccard_main_final = jaccard_extraction(main, final)
jaccard_intro_final = jaccard_extraction(intro, final)

jaccard_pd = pd.DataFrame({"Jaccard: Intro + Main" : jaccard_intro_main, 
                           "Jaccard: Main + Final": jaccard_main_final, 
                           "Jaccard: Intro + Final": jaccard_intro_final}, 
                           index=[0])

# Finding Cosine Similarity Between the Introduction, the Main Body and the Conclusion
corpus = [intro, main, main, final, intro, final]
tfidf_results = tfidf_model.fit_transform(corpus).todense()

cosine_intro_main = cosine_similarity(np.asarray(tfidf_results[0]), np.asarray(tfidf_results[1]))
cosine_main_final = cosine_similarity(np.asarray(tfidf_results[2]), np.asarray(tfidf_results[3]))
cosine_intro_final = cosine_similarity(np.asarray(tfidf_results[4]), np.asarray(tfidf_results[5]))

cosine_intro_main_float = float(np.array2string(cosine_intro_main, precision=2, separator = ",", suppress_small=True)[2:6])
cosine_main_final_float = float(np.array2string(cosine_main_final, precision=2, separator = ",", suppress_small=True)[2:6])
cosine_intro_final_float = float(np.array2string(cosine_intro_final, precision=2, separator = ",", suppress_small=True)[2:6])

cosine_pd = pd.DataFrame({"Cosine: Intro + Main" : cosine_intro_main_float, 
                           "Cosine: Main + Final": cosine_main_final_float, 
                           "Cosine: Intro + Final": cosine_intro_final_float}, 
                           index=[0])

# Named Entities Recognition
text_ner = nlp(text_lemmatized)
intro_ner = nlp(intro)
main_ner = nlp(main)
final_ner = nlp(final)

text_ner_dict = {named_entity: named_entity.label_ for named_entity in text_ner.ents}
intro_ner_dict = {named_entity: named_entity.label_ for named_entity in intro_ner.ents}
main_ner_dict = {named_entity: named_entity.label_ for named_entity in main_ner.ents}
final_ner_dict = {named_entity: named_entity.label_ for named_entity in final_ner.ents}

text_ner_per = find_keys(text_ner_dict, "PER")
intro_ner_per = find_keys(intro_ner_dict, "PER")
main_ner_per = find_keys(main_ner_dict, "PER")
final_ner_per = find_keys(final_ner_dict, "PER")

text_ner_loc = find_keys(text_ner_dict, "LOC")
intro_ner_loc = find_keys(intro_ner_dict, "LOC")
main_ner_loc = find_keys(main_ner_dict, "LOC")
final_ner_loc = find_keys(final_ner_dict, "LOC")

text_ner_org = find_keys(text_ner_dict, "ORG")
intro_ner_org = find_keys(intro_ner_dict, "ORG")
main_ner_org = find_keys(main_ner_dict, "ORG")
final_ner_org = find_keys(final_ner_dict, "ORG")

ner_statistics = [
    [" ", "The Whole Text", "Intro", "Main", "Final"], 
    ["Number of Entities", len(text_ner_dict), len(intro_ner_dict), len(main_ner_dict), len(final_ner_dict)], 
    ["Persons", len(text_ner_per), len(intro_ner_per), len(main_ner_per), len(final_ner_per)],
    ["Organizations", len(text_ner_loc), len(intro_ner_loc), len(main_ner_loc), len(final_ner_loc)],
    ["Locations", len(text_ner_org), len(intro_ner_org), len(main_ner_org), len(final_ner_org)]
]

# Frequent Words
frequent_words = FreqDist(tokens_without_stopwords).most_common(15)
frequent_words_intro = FreqDist(intro_tokens_without_stopwords).most_common(5)
frequent_words_main = FreqDist(main_tokens_without_stopwords).most_common(20)
frequent_words_final = FreqDist(final_tokens_without_stopwords).most_common(5)

frequent_words = [word for word, count in frequent_words if len(word)>3]
frequent_words_intro = [word for word, count in frequent_words_intro if len(word)>3]
frequent_words_main = [word for word, count in frequent_words_main if len(word)>3]
frequent_words_final = [word for word, count in frequent_words_final if len(word)>3]

frequent_words_matrix = [[" ", "Key Words"],
                   ["The Whole Text", " ".join(frequent_words)], 
                   ["Intro", " ".join(frequent_words_intro)],
                   ["Main", " ".join(frequent_words_main[:5])],
                   [" ", " ".join(frequent_words_main[6:])],
                   ["Final", " ".join(frequent_words_final)]]

frequent_entities_per = [word for word, count in FreqDist(text_ner_per).most_common(4)]
frequent_entities_per_intro = [word for word, count in FreqDist(intro_ner_per).most_common(4)]
frequent_entities_per_main = [word for word, count in FreqDist(main_ner_per).most_common(4)]
frequent_entities_per_final = [word for word, count in FreqDist(final_ner_per).most_common(4)]

frequent_entities_loc = [word for word, count in FreqDist(text_ner_loc).most_common(5)]
frequent_entities_loc_intro = [word for word, count in FreqDist(intro_ner_loc).most_common(5)]
frequent_entities_loc_main = [word for word, count in FreqDist(main_ner_loc).most_common(5)]
frequent_entities_loc_final = [word for word, count in FreqDist(final_ner_loc).most_common(5)]

frequent_entities_org = [word for word, count in FreqDist(text_ner_org).most_common(5)]
frequent_entities_org_intro = [word for word, count in FreqDist(intro_ner_org).most_common(5)]
frequent_entities_org_main = [word for word, count in FreqDist(main_ner_org).most_common(5)]
frequent_entities_org_final = [word for word, count in FreqDist(final_ner_org).most_common(3)]

frequent_entities_matrix = [[" ", "Personal Names", "Organizations", "Locations"],
                   ["The Whole Text", frequent_entities_per, frequent_entities_org, frequent_entities_loc], 
                #    [" ", " ".join(frequent_entities_per[3:]), " ".join(frequent_entities_org[3:]), " ".join(frequent_entities_loc[3:])],
                   ["Intro", frequent_entities_per_intro, frequent_entities_org_intro, frequent_entities_loc_intro], 
                #    [" ", " ".join(frequent_entities_per_intro[3:]), " ".join(frequent_entities_org_intro[3:]), " ".join(frequent_entities_loc_intro[3:])],
                   ["Main", frequent_entities_per_main, frequent_entities_org_main, frequent_entities_loc_main], 
                #    [" ", " ".join(frequent_entities_per_main[3:]), " ".join(frequent_entities_org_main[3:]), " ".join(frequent_entities_loc_main[3:])],
                   ["Final", frequent_entities_per_final, frequent_entities_org_final, frequent_entities_loc_final]] 
                #    [" ", " ".join(frequent_entities_per_final[3:]), " ".join(frequent_entities_org_final[3:]), " ".join(frequent_entities_loc_final[3:])]]

# Sentiment Analysis 
english_texts = nltk.sent_tokenize(english_text)
sentiment_analysis = list([TextBlob(item).sentiment.polarity for item in english_texts])
sentiment = [[i for i in range(len(sentiment_analysis))], sentiment_analysis]

# Dash Visualization
app = Dash(__name__)

data_matrix_descriptive = [
    [" ", "Text Part", "Quantity"], 
    ["Sentences", "The Whole Text", len(sentences)], 
    ["Sentences", "Intro", len(intro_sent)], 
    ["Sentences", "Main", len(main_sent)], 
    ["Sentences", "Final", len(final_sent)], 
    ["Tokens", "The Whole Text", len(tokens)], 
    ["Tokens", "Intro", len(intro_tokens)], 
    ["Tokens", "Main", len(main_tokens)], 
    ["Tokens", "Final", len(final_tokens)],
    ["Tokens without Stopwords", "The Whole Text", len(tokens_without_stopwords)], 
    ["Tokens without Stopwords", "Intro", len(intro_tokens_without_stopwords)], 
    ["Tokens without Stopwords", "Main", len(main_tokens_without_stopwords)], 
    ["Tokens without Stopwords", "Final", len(final_tokens_without_stopwords)]]


text_parts = ["The Whole Text", "Intro", "Main", "Final"]
text_pairs = ["Intro-Main", "Main-Final", "Intro-Final"]

# Plotly Graphs
fig = ff.create_table(data_matrix_descriptive)

fig_keywords = ff.create_table(frequent_words_matrix)

fig_tokens = go.Figure(data=[
    go.Bar(name="Tokens", x=text_parts, y=[len(tokens), len(intro_tokens), len(main_tokens), len(final_tokens)]),
    go.Bar(name="Tokens without Stopwords", x=text_parts, y=[len(tokens_without_stopwords), len(intro_tokens_without_stopwords), len(main_tokens_without_stopwords), len(final_tokens_without_stopwords)])
])

fig.update_layout(barmode="group")

fig_keyentities = ff.create_table(frequent_entities_matrix)

fig_distance = go.Figure(data=[
    go.Bar(name="Jaccard Criterion", x=text_pairs, y=[jaccard_intro_main, jaccard_main_final, jaccard_intro_final]),
    go.Bar(name="Cosine Distance Criterion", x=text_pairs, y=[cosine_intro_main_float, cosine_main_final_float, cosine_intro_final_float])
])

fig.update_layout(barmode="group")

fig_ner = ff.create_table(ner_statistics)

fig_sentiment = go.Figure(data=[go.Scatter(x=sentiment[0], y=sentiment[1])])

# Dash Interpretation
app.layout = html.Div(children=[
    html.H1(children="Descriptive Statistics and Result of Data Mining and Feature Extraction Based on the Alexey Tarasov's Article"),
    
    html.H3(children="Key Words in the Article"),

    dcc.Graph(id="Key Words", 
              figure=fig_keywords),
    
    html.H3(children="Descriptive Statistics Based on the Text"), 
    
    dcc.Graph(id="Descriptive graph", 
              figure=fig),

    html.H3(children="The Average Number of Tokens in Different Parts of the Text"),

    dcc.Graph(id="tokens_2", 
              figure=fig_tokens),
    
    html.H3(children="Jaccard Similarity and Cosine Distance Creterion"),

    dcc.Graph(id="jaccard_and_cosine", 
              figure=fig_distance),

    html.H3(children="Key Entities in the Article"),

    dcc.Graph(id="Key Entities", 
              figure=fig_keyentities),

    html.H3(children="Named Entities in Different Text Parts"),

    dcc.Graph(id="ner", 
              figure=fig_ner), 
    
    html.H3(children="Sentiment Analysis (from Intro to Final)"),

    dcc.Graph(id="Sentiment", 
              figure=fig_sentiment)])

if __name__ == '__main__':
    app.run(debug=True)

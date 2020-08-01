import string
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Import Data; Change as deemed necessary
data_train = pd.read_csv('C:/Users/Sri/Documents/DonorsChooseData/train/train.csv')


# Functions for text analysis features:

# Prepare_text: prepares text for analysis, cleaning, tokenization, lemmatization, stop-word removal, etc.
def prepare_text(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    for char in string.punctuation:
        cleaned_text = cleaned_text.replace(char, ' ')

    # Using word_tokenize because it's faster than split()
    tokenized_words = word_tokenize(cleaned_text, "english")

    # Removing Stop Words
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)

    # Lemmatization - From plural to single + Base form of a word (example better-> good)
    lemma_words = []
    for word in final_words:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)
    return str(lemma_words)


# dominant emotion: returns the dominant emotion present in given text, based on emotions.txt
def dom_emotion(text):
    te = prepare_text(text)
    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(': ')
            if word in te:
                emotion_list.append(emotion)
    d = Counter(emotion_list)
    return d.most_common(1)[0][0] if d else "None"


# sentiment analysis: returns sentiment index; abs val is strength of sentiment
# + number is pos sentiment
# - number is neg sentiment
def sentiment_analyse(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    if score['neg'] > score['pos']:
        return -score['neg']
    elif score['neg'] < score['pos']:
        return score['pos']
    else:
        return score['pos']


# returns verb index, how action oriented the essay is?
def verb_eval(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    for char in string.punctuation:
        cleaned_text = cleaned_text.replace(char, ' ')

    tokenized_words = word_tokenize(cleaned_text, "english")
    tags = nltk.pos_tag(tokenized_words)
    counts = Counter(tag for word, tag in tags)
    total = sum(counts.values())
    if total == 0:
        return 0
    else:
        return (counts['VB'] + counts['VBD'] + counts['VBN'] + counts['VBG'] + counts['VBP'] + counts['VBZ']) / total


# returns descriptive index of text; how expressive given essay is>
def comp_desc_eval(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    for char in string.punctuation:
        cleaned_text = cleaned_text.replace(char, ' ')

    tokenized_words = word_tokenize(cleaned_text, "english")
    tags = nltk.pos_tag(tokenized_words)
    counts = Counter(tag for word, tag in tags)
    total = sum(counts.values())
    if total == 0:
        return 0
    else:
        return (counts['JJR'] + counts['RBR'] + counts['JJS'] + counts['RBS']) / total


# returns index of personal index of text; how many possessive or personal pronouns
def personal_eval(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    tokenized_words = word_tokenize(cleaned_text, "english")
    tags = nltk.pos_tag(tokenized_words)
    counts = Counter(tag for word, tag in tags)
    total = sum(counts.values())
    if total == 0:
        return 0
    else:
        return (counts['PRP'] + counts['PRP$']) / total


# Initialization of new data features
data_train['essay_2_emotion'] = data_train.apply(lambda x: dom_emotion(x['project_essay_2']), axis=1)
data_train['essay_2_sentiment'] = data_train.apply(lambda x: sentiment_analyse(x['project_essay_2']), axis=1)
data_train['essay_2_verb_ind'] = data_train.apply(lambda x: verb_eval(x['project_essay_2']), axis=1)
data_train['essay_2_desc_ind'] = data_train.apply(lambda x: comp_desc_eval(x['project_essay_2']), axis=1)
data_train['essay_2_pers_ind'] = data_train.apply(lambda x: personal_eval(x['project_essay_2']), axis=1)
data_train['essay_1_emotion'] = data_train.apply(lambda x: dom_emotion(x['project_essay_1']), axis=1)
data_train['essay_1_sentiment'] = data_train.apply(lambda x: sentiment_analyse(x['project_essay_1']), axis=1)
data_train['essay_1_verb_ind'] = data_train.apply(lambda x: verb_eval(x['project_essay_1']), axis=1)
data_train['essay_1_desc_ind'] = data_train.apply(lambda x: comp_desc_eval(x['project_essay_1']), axis=1)
data_train['essay_1_pers_ind'] = data_train.apply(lambda x: personal_eval(x['project_essay_1']), axis=1)

# list of features to be used in new data file
header = ["essay_2_emotion", "essay_2_sentiment", "essay_2_verb_ind",
          "essay_2_desc_ind", "essay_2_pers_ind", "essay_1_emotion",
          "essay_1_sentiment", "essay_1_verb_ind", "essay_1_desc_ind",
          "essay_1_pers_ind", "project_essay_2", "project_essay_1",
          "project_is_approved"]

# dataframe converted to CSV; change file directory/path as deemed necessary
data_train.to_csv(r'C:\Users\Sri\Desktop\text_train.csv', columns=header)

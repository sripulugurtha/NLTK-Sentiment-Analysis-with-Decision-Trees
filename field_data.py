import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# open up the console
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

col_names = ["essay_2_emotion", "essay_2_sentiment", "essay_2_verb_ind",
             "essay_2_desc_ind", "essay_2_pers_ind", "essay_1_emotion",
             "essay_1_sentiment", "essay_1_verb_ind", "essay_1_desc_ind",
             "essay_1_pers_ind", "project_essay_2", "project_essay_1",
             "project_is_approved"]

# import data from text analysis fields file (end: data --> csv)
# change file directory as deemed necessary
df = pd.read_csv('C:/Users/Sri/Desktop/text_train.csv', header=None, low_memory=False, names=col_names, skiprows=[0])


# convert data to usable format (strings)
def convert(data):
    number = preprocessing.LabelEncoder()
    data['essay_2_emotion'] = number.fit_transform(data.essay_2_emotion)
    data['essay_1_emotion'] = number.fit_transform(data.essay_1_emotion)
    data = data.fillna(-999)
    return data


# subset data for analyzing features
df = convert(df)
df = df[0:5000]


plt.figure(figsize=(10,7))
plt.scatter(df["essay_1_pers_ind"],df["project_is_approved"], marker='o');
plt.legend(loc='best')
plt.xlabel('feature values')
plt.ylabel('project approval')
plt.show()

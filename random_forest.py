import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sn

col_names = ["essay_2_emotion", "essay_2_sentiment", "essay_2_verb_ind",
             "essay_2_desc_ind", "essay_2_pers_ind", "essay_1_emotion",
             "essay_1_sentiment", "essay_1_verb_ind", "essay_1_desc_ind",
             "essay_1_pers_ind", "project_essay_2", "project_essay_1",
             "project_is_approved"]

df = pd.read_csv('C:/Users/Sri/Desktop/text_train.csv', header=None, low_memory=False, names=col_names, skiprows=[0])

# convert data (strings) to a usable format
def convert(data):
    number = preprocessing.LabelEncoder()
    data['essay_2_emotion'] = number.fit_transform(data.essay_2_emotion)
    data['essay_1_emotion'] = number.fit_transform(data.essay_1_emotion)
    data = data.fillna(-999)
    return data


convert(df)


# initialize forest based on given parameters (changed for plotting various hyperparameters)

feature_cols1 = ['essay_2_emotion', 'essay_2_sentiment', 'essay_2_verb_ind', 'essay_2_desc_ind', 'essay_2_pers_ind',
                 'essay_1_emotion', 'essay_1_sentiment', 'essay_1_verb_ind', 'essay_1_desc_ind', 'essay_1_pers_ind']

X = df[feature_cols1]
y = df.project_is_approved  # target var

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.92, random_state=1)

# make forest model
model1 = RandomForestClassifier(max_depth=50)
model1.fit(X_train, y_train)
y_predicted = model1.predict(X_test)

# make confusion matrix
cm = confusion_matrix(y_test, y_predicted)

# plot matrix
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')
plt.show()



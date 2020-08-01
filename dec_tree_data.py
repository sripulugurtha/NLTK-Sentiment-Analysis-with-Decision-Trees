import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
import matplotlib.pyplot as plt

# open up the console to large limits
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# features to be used in dec tree
col_names = ["essay_2_emotion", "essay_2_sentiment", "essay_2_verb_ind",
             "essay_2_desc_ind", "essay_2_pers_ind", "essay_1_emotion",
             "essay_1_sentiment", "essay_1_verb_ind", "essay_1_desc_ind",
             "essay_1_pers_ind", "project_essay_2", "project_essay_1",
             "project_is_approved"]

# imported data from text_analysis_fields file (at the end, data --> csv)
df = pd.read_csv('C:/Users/Sri/Desktop/text_train.csv', header=None, low_memory=False, names=col_names, skiprows=[0])


# usable data; no errors when processing
def convert(data):
    number = preprocessing.LabelEncoder()
    data['essay_2_emotion'] = number.fit_transform(data.essay_2_emotion)
    data['essay_1_emotion'] = number.fit_transform(data.essay_1_emotion)
    data = data.fillna(-999)
    return data


# convert data
convert(df)


# change parameters to see different graphs, with effect on tree accuracy
# add and change parameters as deemed necessary to optimize and explore decision tree
# potential features: min leaf, samples, train size, tree depth, criterion, splitter, etc.
def tree_accuracy(nodes, train):
    feature_cols1 = ['essay_2_emotion', 'essay_2_sentiment', 'essay_2_verb_ind', 'essay_2_desc_ind', 'essay_2_pers_ind',
                     'essay_1_emotion', 'essay_1_sentiment', 'essay_1_verb_ind', 'essay_1_desc_ind', 'essay_1_pers_ind']
    X = df[feature_cols1]
    y = df.project_is_approved
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=nodes / 100, random_state=1)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf = clf.fit(X_train, y_train)  # train the tree
    # return training accuracies
    if train:
        pred = clf.predict(X_train)
        return metrics.accuracy_score(y_train, pred)

    # return validation accuracies
    else:
        pred = clf.predict(X_test)
        return metrics.accuracy_score(y_test, pred)


# training and validation accuracy data; for LEARNING CURVE
# change data vals as deemed necessary to plot different features
clf_data = pd.DataFrame(
    [i, tree_accuracy(i, True)] for i in range(1, 100))
clf_data_val = pd.DataFrame(
    [i, tree_accuracy(i, False)] for i in range(1, 100))


# Plot learning curve; change as deemed necessary
plt.plot(clf_data[0], clf_data[1])
plt.plot(clf_data_val[0], clf_data_val[1])
plt.title("Decision Tree Accuracy Based on Train Data Size\nCriterion: Entropy\nMax Tree Depth: 3")
plt.xlabel('Training Data Size')
plt.ylabel('Decision Tree Accuracy (percentage)')
plt.show()

import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt

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


convert(df)

# Tested Decision Trees

# used features
feature_cols1 = ['essay_2_emotion', 'essay_2_sentiment', 'essay_2_verb_ind', 'essay_2_desc_ind', 'essay_2_pers_ind',
                 'essay_1_emotion', 'essay_1_sentiment', 'essay_1_verb_ind', 'essay_1_desc_ind', 'essay_1_pers_ind']

# split data into target and used variables
X = df[feature_cols1]
y = df.project_is_approved  # target var
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# feature_cols1, 70% test:
clf1 = DecisionTreeClassifier()  # 1 Accuracy: 0.7309058289396603
clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=3)  # 2 Accuracy: 0.8476127709431751
clf3 = DecisionTreeClassifier(criterion="gini", max_depth=3)  # 3 Accuracy: 0.8475944639718805
clf4 = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=3)  # 4 Accuracy: 0.8476127709431751
clf5 = DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=3)  # 5 Accuracy: 0.8476127709431751
clf6 = DecisionTreeClassifier(criterion="entropy", max_depth=3)  # 6 Accuracy: 0.8474480082015231
clf7 = DecisionTreeClassifier(criterion="entropy", max_depth=2)  # 7 Accuracy: 0.8476127709431751
clf8 = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=2)  # 7 Accuracy: 0.8476127709431751
# MOST OPTIMAL TREE: Accuracy: 0.8476127709431751, trees 2, 4, 5, 7

# feature_cols1, 80% train:
clf8 = DecisionTreeClassifier()  # 8 Accuracy: 0.7334413444639719
clf9 = DecisionTreeClassifier(criterion="entropy", max_depth=3)  # 9 Accuracy: 0.8471550966608085
clf10 = DecisionTreeClassifier(criterion="gini", max_depth=3)  # 10 Accuracy: 0.8471550966608085
clf11 = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=3)  # 11 Accuracy: 0.8471550966608085
clf12 = DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=3)  # 12 Accuracy: 0.8471001757469244
clf13 = DecisionTreeClassifier(criterion="entropy", max_depth=3)  # 13 Accuracy: 0.8471550966608085
clf14 = DecisionTreeClassifier(criterion="entropy", max_depth=2)  # 14 Accuracy: 0.8471550966608085

# change dec tree to test different types
clf = clf2.fit(X_train, y_train)  # train the tree

# plot decision tree
plt.figure(figsize=(14, 7))
tree.plot_tree(decision_tree=clf, feature_names=feature_cols1, rounded=True, fontsize=8)
plt.show()

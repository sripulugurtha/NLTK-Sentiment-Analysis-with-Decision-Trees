import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
import matplotlib.pyplot as plt


col_names = ["essay_2_emotion", "essay_2_sentiment", "essay_2_verb_ind",
             "essay_2_desc_ind", "essay_2_pers_ind", "essay_1_emotion",
             "essay_1_sentiment", "essay_1_verb_ind", "essay_1_desc_ind",
             "essay_1_pers_ind", "project_essay_2", "project_essay_1",
             "project_is_approved"]

# open up the console
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# balance data based on percentage of approved projects from given data set
def distribute_data(d, percent_new_approved_samples):
    # all values based on parameters of given data set (predetermined --> kaggle)
    data = d[0:182079]
    total_samples = 182080
    total_unapproved_samples = 27734
    total_approved_samples = 154346
    total_new_samples = total_unapproved_samples / (1 - (percent_new_approved_samples / 100))
    desired_approved_samples = total_new_samples * (percent_new_approved_samples / 100)
    removed_approved_samples = total_approved_samples - desired_approved_samples
    remove = []# list of indexes to remove
    remove_index = 0
    while removed_approved_samples > 0 and remove_index < total_samples:
        if data['project_is_approved'][remove_index] == 1:
            remove.append(remove_index)
            removed_approved_samples = removed_approved_samples - 1

        remove_index = remove_index + 1
    data = data.drop(remove)
    return convert(data)

# return accuracy of given data and train boolean (used for plotting)
def dec_tree_accuracy(data, train):
    feature_cols1 = ['essay_2_emotion', 'essay_2_sentiment', 'essay_2_verb_ind', 'essay_2_desc_ind', 'essay_2_pers_ind',
                     'essay_1_emotion', 'essay_1_sentiment', 'essay_1_verb_ind', 'essay_1_desc_ind', 'essay_1_pers_ind']
    X = data[feature_cols1]
    y = data.project_is_approved
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)  # train the tree
    # return training accuracies
    if train:
        prediction = clf.predict(X_train)
        return metrics.accuracy_score(y_train, prediction)

    # return validation accuracies
    else:
        prediction = clf.predict(X_test)
        return metrics.accuracy_score(y_test, prediction)


def convert(data):
    number = preprocessing.LabelEncoder()
    data['essay_2_emotion'] = number.fit_transform(data.essay_2_emotion)
    data['essay_1_emotion'] = number.fit_transform(data.essay_1_emotion)
    data = data.fillna(-999)
    return data


# original dataset
df = pd.read_csv('C:/Users/Sri/Desktop/text_train.csv', header=None, low_memory=False, names=col_names,
                 skiprows=[0])

# initialize datasets based on above functions
balance_data_train = pd.DataFrame(
    [i, dec_tree_accuracy(distribute_data(df, i), True)] for i in range(1, 100))

balance_data_val = pd.DataFrame(
    [i, dec_tree_accuracy(distribute_data(df, i), False)] for i in range(1, 100))

# Plot curves; change as deemed necessary
plt.plot(balance_data_train[0], balance_data_train[1], label="Training")
plt.plot(balance_data_val[0], balance_data_val[1], label="Validation")
plt.legend(loc='best')
plt.title("Decision Tree Accuracy based on Approved Sample Data Distribution")
plt.xlabel('Percent of Approved Samples')
plt.ylabel('Decision Tree Accuracy')
plt.show()


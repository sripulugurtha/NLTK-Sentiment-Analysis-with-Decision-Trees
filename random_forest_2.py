# Copy of random_forest, made to run more processes and plots.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

col_names = ["essay_2_emotion", "essay_2_sentiment", "essay_2_verb_ind",
             "essay_2_desc_ind", "essay_2_pers_ind", "essay_1_emotion",
             "essay_1_sentiment", "essay_1_verb_ind", "essay_1_desc_ind",
             "essay_1_pers_ind", "project_essay_2", "project_essay_1",
             "project_is_approved"]

df = pd.read_csv('C:/Users/Sri/Desktop/text_train.csv', header=None, low_memory=False, names=col_names, skiprows=[0])

# convert unusable data to usable format
def convert(data):
    number = preprocessing.LabelEncoder()
    data['essay_2_emotion'] = number.fit_transform(data.essay_2_emotion)
    data['essay_1_emotion'] = number.fit_transform(data.essay_1_emotion)
    data = data.fillna(-999)
    return data


convert(df)


# def make_forest(n, f1, t):
def make_forest(n):
    feature_cols1 = ['essay_2_emotion', 'essay_2_sentiment', 'essay_2_verb_ind', 'essay_2_desc_ind', 'essay_2_pers_ind',
                     'essay_1_emotion', 'essay_1_sentiment', 'essay_1_verb_ind', 'essay_1_desc_ind', 'essay_1_pers_ind']

    X = df[feature_cols1]
    y = df.project_is_approved  # target var

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.92, random_state=1)

    model1 = RandomForestClassifier(criterion='entropy', max_depth=n)
    model1.fit(X_train, y_train)
    pred = model1.predict(X_test)
    pred_train = model1.predict(X_train)
    print(metrics.accuracy_score(y_test, pred))
    print(metrics.f1_score(y_test, pred))


    # Plot forest accuracy data:
    # if not f1:  # accuracy score
    #     if not t:  # validation
    #         return metrics.accuracy_score(y_test, pred)
    #     else:
    #         return metrics.accuracy_score(y_train, pred_train)
    #
    # else:  # f1 score
    #     if t:  # training
    #         return metrics.f1_score(y_train, pred_train)
    #     else:
    #         return metrics.f1_score(y_test, pred)

#
# ss = pd.DataFrame(
#     [i, make_forest(i, True, True)] for i in range(2, 18, 2))
# ff = pd.DataFrame(
#     [i, make_forest(i, True, False)] for i in range(2, 18, 2))
#
# # Plot curves; change as deemed necessary
# plt.plot(ss[0], ss[1], label='Training, F1')
# plt.plot(ff[0], ff[1], label='Validation, F1')
#
# plt.legend(loc='best')
# plt.title("F1 random forest accuracy based on\nmax depth")
# plt.xlabel('max depth')
# plt.ylabel('forest accuracy')
# plt.show()
# print(max(ff))

# plot forest accuracies
make_forest(50)




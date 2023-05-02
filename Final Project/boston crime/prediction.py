import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
import warnings
warnings.filterwarnings('ignore')


def getData(category):
    crime_rate_df = pd.read_csv('dataset/boston_crime_2021-2022.csv', dtype=str)
    crime_dictionary = ['LARCENY', 'M/V ACCIDENT', 'LIQUOR', 'INCEST', 'MANSLAUGHTER', 'MISSING PERSON',
                        'PROPERTY - LOST', 'MURDER', 'FRAUD', 'PROSTITUTION', 'RAPE', 'ROBBERY', 'ASSAULT',
                        'SICK/INJURED/MEDICAL', 'TOWED MOTOR VEHICLE', 'TRESPASSING', 'VIOLATION', 'ANIMAL',
                        'AUTO THEFT', 'FIREARM/WEAPON', 'HUMAN TRAFFICKING', 'DRUGS', 'SEX OFFENSE', 'ARSON',
                        'VANDALISM', 'SEARCH WARRANT', 'KIDNAPPING', 'DEATH INVESTIGATION', 'CHILD ABUSE', 'HARASSMENT']

    crime_list = [0, 1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10, 11, 12,
                  13, 14, 15, 16, 17,
                  18, 19, 20, 21, 22, 23,
                  24, 25, 26, 27, 28, 29]
    for j in range(len(crime_list)):
        crime_rate_df.loc[crime_rate_df['OFFENSE_DESCRIPTION'].str.contains(crime_dictionary[j]), 'GROUP'] = crime_list[j]

    crime_rate_df = crime_rate_df.dropna()

    Weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i in range(len(Weekday)):
        crime_rate_df['DAY_OF_WEEK'] = np.where((crime_rate_df.DAY_OF_WEEK == Weekday[i]), i, crime_rate_df.DAY_OF_WEEK)
    X = crime_rate_df.drop([category, 'OFFENSE_DESCRIPTION', 'Location', 'DISTRICT', 'Dates', 'STREET', 'YEAR'], axis=1).values
    Y = crime_rate_df[[category]].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
    return x_train, y_train, x_test, y_test


def getTable(cm, i, all=False):
    TP = cm[i][0][0]
    FP = cm[i][0][1]
    FN = cm[i][1][0]
    TN = cm[i][1][1]
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    d = {'Accuracy': [ACC], 'True positive rate': [TPR], 'True negative rate': [TNR]}
    dfx = pd.DataFrame(data=d)
    if all:
        return TP, FP, FN, TN, TPR, TNR, ACC
    return dfx


def predictCrime():
    cm = []
    x_train, y_train, x_test, y_test = getData('GROUP')

    NB_classifier = GaussianNB().fit(x_train, y_train)
    accuracy = accuracy_score(y_test, NB_classifier.predict(x_test))

    print("\n1:")
    print('Implement a Naive Bayesian classifier:')
    print('The accuracy is', accuracy)

    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(x_train, y_train)
    accuracy = log_reg_classifier.score(x_train, y_train)

    print("\n2:")
    print('Implement a Logistic regression classifier:')
    print('The accuracy is', accuracy)

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)

    print("\n3:")
    print('Implement a Decision Tree:')
    print('The accuracy is', accuracy)

    # 5. Use Random Forest classifier
    error_rate = []
    random_forest_table = pd.DataFrame(columns=['n_estimators', 'max_depth', 'accuracy'])
    for i in range(1, 11):
        for j in range(1, 6):
            rf = RandomForestClassifier(n_estimators=i, max_depth=j)
            rf.fit(x_train, y_train)
            error_rate.append(1 - accuracy_score(y_test, rf.predict(x_test)))
            ACC = accuracy_score(y_test, rf.predict(x_test))
            random_forest_table.loc[len(random_forest_table.index)] = [i, j, ACC]

    best_n = error_rate.index(min(error_rate)) % 10 + 1
    best_max = error_rate.index(min(error_rate)) % 5 + 1

    print("\n4:")
    print('Implement a Random Forest classifier :')
    print("The best n_estimators and max_depth are", best_n, "and", best_max)

    rf = RandomForestClassifier(n_estimators=best_n, max_depth=best_max)
    rf.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(x_test))

    print('The accuracy is', accuracy)



predictCrime()

















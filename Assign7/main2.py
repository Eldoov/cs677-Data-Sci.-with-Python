# CS677 assignment 7
# by Zuowen Tang
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from seaborn import pairplot
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# Question 1
def Q1_Task():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # 1. load the data into Pandas dataframe. Extract two dataframes
    # with the above 4 features: df 0 for surviving patients (DEATH EVENT = 0) and df 1 for
    # deceased patients (DEATH EVENT = 1)
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    df = df[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets", "DEATH_EVENT"]]
    df0 = df[df["DEATH_EVENT"] == 0]
    df1 = df[df["DEATH_EVENT"] == 1]
    # print(df0.head())
    # print(df1.head())

    # 2. for each dataset, construct the visual representations of correponding correlation
    # matrices M0 (from df 0) and M1 (from df 1) and save the plots into two separate files
    figure = plt.figure()
    axes = figure.add_subplot(111)
    ax = axes.matshow(df0[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]].corr(),
                      interpolation='nearest')
    figure.colorbar(ax)
    #plt.show()
    df0[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]].corr()

    figure = plt.figure()
    axes = figure.add_subplot(111)
    ax = axes.matshow(df1[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]].corr(),
                      interpolation='nearest')
    figure.colorbar(ax)
    #plt.show()
    df1[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]].corr()

    # 3. examine your correlation matrix plots visually and answer the following:
    # (a) which features have the highest correlation for surviving patients?
    # (b) which features have the lowest correlation for surviving patients?
    # (c) which features have the highest correlation for deceased patients?
    # (d) which features have the lowest correlation for deceased patients?
    # (e) are results the same for both cases?

    # print("a) serum_sodium and creatinine_phosphokinase have the highest correlation for surviving patients.\n"
    #       "b) serum_sodium and serum_creatinine have the lowest correlation for surviving patients.\n"
    #       "c) serum_sodium and creatinine_phosphokinase have the highest correlation for deceased patients.\n"
    #       "d) serum_sodium and serum_creatinine have the lowest correlation for deceased patients.\n"
    #       "e) No they are slightly different.\n")

    # 4. for each class and for each feature f1, f2, f3, f4, compute its mean μ()
    # and standard deviation σ(). Summarize them in a table as shown below:
    classification = pd.DataFrame(
        columns=['Class', 'μ(f1)', 'σ(f1)', 'μ(f2)', 'σ(f2)', 'μ(f3)', 'σ(f3)', 'μ(f4)', 'σ(f4)'])
    name = ['0', '1', 'All']
    for i in range(3):
        if i == 0:
            dfx = df0
        elif i == 1:
            dfx = df1
        elif i == 2:
            dfx = df

        avg_f1 = dfx['creatinine_phosphokinase'].mean()
        std_f1 = dfx['creatinine_phosphokinase'].std()
        avg_f2 = dfx['serum_creatinine'].mean()
        std_f2 = dfx['serum_creatinine'].std()
        avg_f3 = dfx['serum_sodium'].mean()
        std_f3 = dfx['serum_sodium'].std()
        avg_f4 = dfx['platelets'].mean()
        std_f4 = dfx['platelets'].std()
        classification.loc[len(classification.index)] = [name[i], avg_f1, std_f1, avg_f2, std_f2, avg_f3,
                                                         std_f3, avg_f4, std_f4]
    print("\nQ1:")
    print(classification)


def getName():
    f1 = 'creatinine_phosphokinase'
    f2 = 'serum_creatinine'
    f3 = 'serum_sodium'
    f4 = 'platelets'
    return f1, f2, f3, f4


def getTable(FP, TN, TP, FN):
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    d = {'TR': [TP], 'FP': [FP], 'TN': [TN], 'FN': [FN], 'Accuracy': [ACC], 'TPR': [TPR], 'TNR': [TNR]}
    dfx = pd.DataFrame(data=d)
    return dfx


def getData():
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    x1 = df[["serum_creatinine", "serum_sodium", "platelets", "DEATH_EVENT"]]
    x2 = df[["creatinine_phosphokinase", "serum_sodium", "platelets", "DEATH_EVENT"]]
    x3 = df[["creatinine_phosphokinase", "serum_creatinine", "platelets", "DEATH_EVENT"]]
    x4 = df[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "DEATH_EVENT"]]

    x1_train, x1_test = train_test_split(x1, test_size=0.5, random_state=0)
    x2_train, x2_test = train_test_split(x2, test_size=0.5, random_state=0)
    x3_train, x3_test = train_test_split(x3, test_size=0.5, random_state=0)
    x4_train, x4_test = train_test_split(x4, test_size=0.5, random_state=0)
    X = [x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, x4_train, x4_test]

    y1_train, y1_test = x1_train['DEATH_EVENT'].values, x1_test['DEATH_EVENT'].values
    y2_train, y2_test = x2_train['DEATH_EVENT'].values, x2_test['DEATH_EVENT'].values
    y3_train, y3_test = x3_train['DEATH_EVENT'].values, x3_test['DEATH_EVENT'].values
    y4_train, y4_test = x4_train['DEATH_EVENT'].values, x4_test['DEATH_EVENT'].values
    Y = [y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test]

    return X, Y


# Question 2
def Q2_Task():
    # 1. split your dataset X into training Xtrain and Xtesting parts (50/50 split).
    #    Using ”pairplot” from seaborn package, plot pairwise relationships in Xtrain separately
    #    for class 0 and class 1. Save your results into 2 pdf files ”survived.pdf” and ”not-survived.pdf”
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    x = df[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets", "DEATH_EVENT"]]

    x_train, x_test = train_test_split(x, test_size=0.5, random_state=0)
    x_train0 = x_train[x_train["DEATH_EVENT"] == 0]
    x_train1 = x_train[x_train["DEATH_EVENT"] == 1]
    pairplot(x_train0)
    #plt.show()
    pairplot(x_train1)
    #plt.show()

    # 2. visually examine your results. Come up with three simple comparisons
    # that you think may be sufficient to predict a survival.
    # if (f_1 > 1100) and (f_3 > 125) and (f_4 < 400000):
    #    x = "survive"
    # else:
    #    x = "not_survive"
    # 3. apply your simple classifier to Xtest and compute predicted class labels
    f1, f2, f3, f4 = getName()
    x_train['Q2pred'] = np.where((x_train[f1] >= 1100) & (x_train[f3] >= 125) & (x_train[f4] < 400000), 1, 0)
    conditions = [
        (x_train['DEATH_EVENT'] == 1) & (x_train['Q2pred'] == 1),
        (x_train['DEATH_EVENT'] == 0) & (x_train['Q2pred'] == 1),
        (x_train['DEATH_EVENT'] == 0) & (x_train['Q2pred'] == 0),
        (x_train['DEATH_EVENT'] == 1) & (x_train['Q2pred'] == 0)
    ]
    values = ['TP', 'FP', 'TN', 'FN']
    x_train['compare'] = np.select(conditions, values)
    print("\nQ2:")
    print(x_train.head(), '\n')
    print(x_train['compare'].value_counts())

    dfx = getTable(9, 93, 6, 41)
    print(dfx)
    print("The accuracy is higher tha 50%.")


# Question 3 use k-NN classifier using sklearn library
def Q3_Task():
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    x = df[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets", "DEATH_EVENT"]]

    x_train, x_test = train_test_split(x, test_size=0.5, random_state=0)
    y_train, y_test = x_train['DEATH_EVENT'].values, x_test['DEATH_EVENT'].values

    error_rate = []
    for k in range(3, 8, 2):
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(x_train, y_train)
        pred_k = knn_classifier.predict(x_test)
        error_rate.append(np.mean(pred_k != y_test))
    plt.plot(range(3, 8, 2), error_rate, color='red', linestyle='dashed', marker='o', markerfacecolor='black',
             markersize=10)
    plt.title('Error Rate vs K')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
    print("\nQ3:")
    print('The optimal value of k from year 1 is 5.')

    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(x_train, y_train)
    predicted = knn_classifier.predict(x_test)
    actual = y_test

    cm = metrics.confusion_matrix(actual, predicted)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    display.plot()
    plt.show()

    dfx = getTable(40,80,9,21)
    print(dfx)
    print("The accuracy is higher tha 50% but not as good ass the former prediction.")


# Question 4 fine tune k-NN classifier using sklearn library
def Q4_Task():
    X, Y = getData()

    error_rate = []
    for i in range(0, 7, 2):
        x_train, x_test = X[i], X[i+1]
        y_train, y_test = Y[i], Y[i + 1]
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(x_train, y_train)
        pred_k = knn_classifier.predict(x_test)
        error_rate.append(np.mean(pred_k != y_test))
    print("\nQ4:")
    print(error_rate)

    # 2. did accuracy increase in any of the 4 cases compared with accuracy when all 4 features are used?
    print("Yes the accuracy increased.")
    # 3. which features, when removed, contributed the most to loss?
    print("When serum_creatinine and serum_sodium are removed, the accuracy decreases.")
    # 4. which features, when removed, contributed the least to loss of accuracy?
    print("When platelets is removed, the accuracy increases.")


def Q5_Task():
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    x = df[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets", "DEATH_EVENT"]]

    x_train, x_test = train_test_split(x, test_size=0.5, random_state=0)
    y_train, y_test = x_train['DEATH_EVENT'].values, x_test['DEATH_EVENT'].values

    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(x_train, y_train)

    predicted = log_reg_classifier.predict(x_test)
    accuracy = log_reg_classifier.score(x_test, y_test)
    print("\nQ5:")
    print("Accuracy is", accuracy)
    cm = metrics.confusion_matrix(y_test, predicted)
    print(cm)

    dfx = getTable(48, 99, 1, 2)
    print(dfx)
    print("The accuracy is higher tha 50% and is slightly better than the first prediction and the KNN prediction.")


def Q6_Task():
    X, Y = getData()

    error_rate = []
    for i in range(0, 7, 2):
        x_train, x_test = X[i], X[i + 1]
        y_train, y_test = Y[i], Y[i + 1]
        log_reg_classifier = LogisticRegression()
        log_reg_classifier.fit(x_train, y_train)
        pred = log_reg_classifier.predict(x_test)
        error_rate.append(np.mean(pred != y_test))
    print("\nQ6:")
    print(error_rate)
    # 2. did accuracy increase in any of the 4 cases compared with accuracy when all 4 features are used?
    print("Yes the accuracy increased.")
    # 3. which features, when removed, contributed the most to loss?
    print("When serum_sodium is removed, the accuracy decreases.")
    # 4. which features, when removed, contributed the least to loss of accuracy?
    print("When platelets is removed, the accuracy increases.")
    # 5. is the relative significance of features the same as you ob- tained using k-NN?
    print("The result is better than K-NN.")


Q1_Task()
Q2_Task()
Q3_Task()
Q4_Task()
Q5_Task()
Q6_Task()
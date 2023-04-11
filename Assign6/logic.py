# CS677 assignment 6
# by Zuowen Tang
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def getData(year):
    df = pd.read_csv("TMO_weekly_label.csv")
    df1 = df[df['Year'] == 2021]
    df2 = df[df['Year'] == 2022]

    if year == 2021:
        df = df1
    elif year == 2022:
        df = df2

    X = df[['Mean Return', 'Volatility']].values
    Y = df['Label'].values

    return X, Y


# 1. what is the equation for logistic regression that your classifier found from year 1 data?
def Task1():
    X, Y = getData(2021)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(X, Y)

    print("Task 1:")
    print(log_reg_classifier.coef_)
    print(log_reg_classifier.intercept_)
    print("the logistic regression is 1 / (1+e^-(0.06468441 - 1.30020309 *X1+0.35472611*X2))")


# 2. what is the accuracy for year 2?
def Task2_3_4_5():
    X1, Y1 = getData(2021)
    X2, Y2 = getData(2022)

    scaler = StandardScaler()
    scaler.fit(X1)
    X1 = scaler.transform(X1)

    new_x = scaler.transform(X2)

    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(X1, Y2)

    predicted = log_reg_classifier.predict(new_x)
    accuracy = log_reg_classifier.score(X1, Y1)
    print("\nTask 2:")
    print("Predicted is",predicted)
    print("Accuracy is", accuracy)

    # 3. compute the confusion matrix for year 2
    cm = confusion_matrix(Y2, predicted)
    print("\nTask 3:")
    print(cm)

    # 4. what is true positive rate (sensitivity or recall) and true negative rate (specificity) for year 2?
    FP, TN, TP, FN = 22, 26, 0, 4
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    sum = FP + TN + TP + FN
    d = {'TR': [TP], 'FP': [FP], 'TN': [TN], 'FN': [FN], 'Total Weeks': [sum], 'Accuracy': [ACC], 'TPR': [TPR],
         'TNR': [TNR]}
    df = pd.DataFrame(data=d)
    print("\nTask 4:")
    print(df)

    # 5. implement a trading strategy based on your labels for year 2 and compare the performance
    #   with the ”buy-and-hold” strategy. Which strategy results in a larger amount at the end of the year?
    df3 = pd.read_csv("TMO_weekly_label.csv")
    df3 = df3[df3['Year'] == 2022]
    meanReturn = df3['Mean Return']

    print("\nTask 5:")
    print('Money earned based on buy-and-hold strategy for Year2:')
    print("-2.2672499999999984")

    meanReturn = list(meanReturn)
    moneyEarned = 0
    for i in range(52):
        if predicted[i] == 'g':
            moneyEarned = moneyEarned + meanReturn[i]
    print('\nNew strategy: only buy when the predicted label is green.')
    print('Money earned based on this strategy for Year2:')
    print(moneyEarned)


Task1()
Task2_3_4_5()
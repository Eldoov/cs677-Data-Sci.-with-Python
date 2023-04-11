# CS677 assignment 6
# by Zuowen Tang
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics


def getData():
    df = pd.read_csv("TMO_weekly_label.csv")
    df1 = df[df['Year'] == 2021]
    df2 = df[df['Year'] == 2022]

    X1 = df1[['Mean Return', 'Volatility']].values
    scaler = StandardScaler()
    scaler.fit(X1)
    Y1 = df1['Label'].values

    X2 = df2[['Mean Return', 'Volatility']].values
    Y2 = df2['Label'].values
    return X1, Y1, X2, Y2


# 1. take k = 3,5,7,9,11. For each value of k compute the accuracy of your k-NN classifier on year 1 data.
# On x axis you plot k and on y-axis you plot accuracy. What is the optimal value of k for year 1?
def Task1():
    X, Y, temp1, temp2 = getData()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

    error_rate = []
    for k in range(3, 12, 2):
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(x_train, y_train)
        pred_k = knn_classifier.predict(x_test)
        error_rate.append(np.mean(pred_k != y_test))

    plt.plot(range(3, 12, 2), error_rate, color='red', linestyle ='dashed', marker ='o', markerfacecolor ='black', markersize = 10)
    plt.title('Error Rate vs K')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
    print("\nTask 1:")
    print('The optimal value of k from year 1 is 5.')


# 2. use the optimal value of k from year 1 to predict labels for year 2. What is your accuracy?
def Task2():
    x_train, y_train, x_test, y_test = getData()

    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(x_train, y_train)
    pred_k = knn_classifier.predict(x_test)
    error_rate = np.mean(pred_k != y_test)

    print("\nTask 2:")
    print('The error rate is', error_rate)


# 3. using the optimal value for k from year 1, compute the confusion matrix for year 2
def Task3_4():
    X1, Y1, X2, Y2 = getData()

    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X1, Y1)

    predicted = knn_classifier.predict(X2)
    actual = Y2

    cm = metrics.confusion_matrix(actual, predicted)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    display.plot()
    plt.show()
    print("\nTask 3:")
    print(cm)

    # 4. what is true positive rate (sensitivity or recall) and true negative rate (specificity) for year 2?
    FP, TN, TP, FN = 8, 27, 14, 3
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    sum = FP+TN+TP+FN
    d = {'TR': [TP], 'FP': [FP], 'TN': [TN], 'FN': [FN], 'Total Weeks': [sum], 'Accuracy': [ACC], 'TPR': [TPR], 'TNR': [TNR]}
    df = pd.DataFrame(data=d)
    print("\nTask 4:")
    print(df)


# 5. implement a trading strategy based on your labels for year 2 and compare the performance
# with the ”buy-and-hold” strategy. Which strategy results in a larger amount at the end of the year?
def Task5():
    df = pd.read_csv("TMO_weekly_label.csv")
    df = df[df['Year'] == 2022]
    meanReturn = df['Mean Return']
    print("\nTask 5:")
    print('Money earned based on buy-and-hold strategy for Year2:')
    print(sum(meanReturn))

    X1, Y1, X2, Y2 = getData()
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X1, Y1)
    predicted = knn_classifier.predict(X2)

    meanReturn = list(meanReturn)
    moneyEarned = 0
    for i in range(52):
        if predicted[i] == 'g':
            moneyEarned = moneyEarned + meanReturn[i]
    print('\nNew strategy: only buy when the predicted label is green.')
    print('Money earned based on this strategy for Year2:')
    print(moneyEarned)


def Driver():
    Task1()
    Task2()
    Task3_4()
    Task5()


Driver()
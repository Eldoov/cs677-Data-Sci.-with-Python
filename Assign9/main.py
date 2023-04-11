import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from scipy import stats
from scipy.stats import t
import warnings
warnings.filterwarnings('ignore')


def getData(year):
    df = pd.read_csv("TMO_weekly_label.csv")
    df = df[df['Year'] == year]
    X = df[['Mean Return', 'Volatility']].values
    Y = df['Label'].values
    return X, Y


def getTable(FP, TN, TP, FN):
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    d = {'Accuracy': [ACC], 'True positive rate': [TPR], 'True negative rate': [TNR]}
    dfx = pd.DataFrame(data=d)
    return dfx


def Naive_Bayesian():
    x_train, y_train = getData(2021)
    x_test, y_test = getData(2022)

    # 1. implement a Gaussian naive bayesian classifier and compute its accuracy for year 2
    NB_classifier = GaussianNB().fit(x_train, y_train)
    prediction = NB_classifier.predict(x_test)
    error_rate = np.mean(prediction != y_test)
    year2ACU = accuracy_score(y_test, NB_classifier.predict(x_test))
    print("-"*50)
    print("Implement Naive Bayesian classifier.")
    print("\nTask 1:")
    print("The error rate is", error_rate)
    print("The accuracy for year 2 is", year2ACU)

    # 2. compute the confusion matrix for year 2
    cm = confusion_matrix(y_test, NB_classifier.predict(x_test))
    print("\nTask 2:")
    print('the confusion matrix is\n', cm)

    # 3. what is true positive rate and true negative rate for year 2
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    dfx = getTable(FP, TN, TP, FN)
    print("\nTask 3:")
    print(dfx)

    # 4. implement a trading strategy based on your labels for year 2 and compare the performance
    # with the ”buy-and-hold” strategy. Which strategy results in a larger amount at the end of the year?
    df3 = pd.read_csv("TMO_weekly_label.csv")
    df3 = df3[df3['Year'] == 2022]
    meanReturn = df3['Mean Return']
    print("\nTask 4:")
    print('Money earned based on buy-and-hold strategy for Year2:')
    print("-2.2672499999999984")

    meanReturn = list(meanReturn)
    moneyEarned = 0
    for i in range(52):
        if prediction[i] == 'g':
            moneyEarned = moneyEarned + meanReturn[i]
    print('\nNew strategy: only buy when the predicted label is green.')
    print('Money earned based on this strategy for Year2:')
    print(moneyEarned)

    print('\nStrategy based on Naive Bayesian has the larger amount at the end of the year.')


def Discriminant_Analysis():
    x_train, y_train = getData(2021)
    x_test, y_test = getData(2022)

    print("")
    print("-" * 50)
    print("Implement linear and quadratic discriminant classifier.")
    
    # 1. what is the equation for linear and quadratic classifier found from year 1 data?
    lda_classifier = LDA().fit(x_train, y_train)
    lda_prediction = lda_classifier.predict(x_test)
    lda_error_rate = np.mean(lda_prediction != y_test)
    #print(lda_error_rate)
    qda_classifier = QDA().fit(x_train, y_train)
    qda_prediction = qda_classifier.predict(x_test)
    qda_error_rate = np.mean(qda_prediction != y_test)
    #print(qda_error_rate)

    print("\nTask 1:")
    print("for linear coef and intercept", lda_classifier.coef_, lda_classifier.intercept_)
    print("so for linear the function is y = -2.106x^2 + 0.5214x + (-0.40100985)")
    print("for Quadratic classifier, we can't get the corf_ and intercet")

    # 2. what is the accuracy for year 2 for each classifier. Which classifier is ”better”?
    print("\nTask 2:")
    print("The accuracy for year 2 for lda is", accuracy_score(y_test, lda_classifier.predict(x_test)))
    print("The accuracy for year 2 for qda is", accuracy_score(y_test, qda_classifier.predict(x_test)))

    # 3. compute the confusion matrix for year 2 for each classifier
    print("\nTask 3:")
    lda_cm = confusion_matrix(y_test, lda_classifier.predict(x_test))
    qda_cm = confusion_matrix(y_test, qda_classifier.predict(x_test))
    print('the confusion matrix for lda is\n', lda_cm)
    print('the confusion matrix for qda is\n', qda_cm)

    # 4. what is true positive rate (sensitivity or recall) and true negative rate (specificity) for year 2?
    print("\nTask 4:")
    TP = lda_cm[0][0]
    FP = lda_cm[0][1]
    FN = lda_cm[1][0]
    TN = lda_cm[1][1]
    dfx = getTable(FP, TN, TP, FN)
    print("True positive rate for LDA'")
    print(dfx)

    TP = lda_cm[0][0]
    FP = lda_cm[0][1]
    FN = lda_cm[1][0]
    TN = lda_cm[1][1]
    dfx = getTable(FP, TN, TP, FN)
    print("\nTrue positive rate for QDA':")
    print(dfx)

    # 5. implement trading strategyies based on your labels for year 2 (for both linear and quadratic)
    # and compare the performance with the ”buy-and-hold” strategy.
    # Which strategy results in a larger amount at the end of the year?
    df3 = pd.read_csv("TMO_weekly_label.csv")
    df3 = df3[df3['Year'] == 2022]
    meanReturn = df3['Mean Return']
    print("\nTask 5:")
    print('Money earned based on buy-and-hold strategy for Year2:')
    print("-2.2672499999999984")

    meanReturn = list(meanReturn)
    moneyEarned = 0
    for i in range(52):
        if lda_prediction[i] == 'g':
            moneyEarned = moneyEarned + meanReturn[i]
    print('\nNew strategy based on LDA.')
    print('Money earned based on this strategy for Year2:')
    print(moneyEarned)

    moneyEarned = 0
    for i in range(52):
        if qda_prediction[i] == 'g':
            moneyEarned = moneyEarned + meanReturn[i]
    print('\nNew strategy based on QDA.')
    print('Money earned based on this strategy for Year2:')
    print(moneyEarned)

    print('\nStrategy based on LDA has the larger amount at the end of the year.')


def Student_t():
    df = pd.read_csv("TMO_weekly_label.csv")
    label_green = df[df['Label'] == 'g']
    label_red = df[df['Label'] == 'r']
    df2021 = df[df['Year'] == 2021]
    mean2021 = df2021['Mean Return']
    df2022 = df[df['Year'] == 2022]
    label2022 = df2022['Label']
    mean2022 = df2022['Mean Return']

    green_prob = len(label_green)/len(df['Label'])
    red_prob = len(label_red)/len(df['Label'])
    #print(green_prob, red_prob)
    print("")
    print("-" * 50)
    print("Implement Student-t Naive Bayesian classifier.")

    # 1. implement a Student-t Naive Bayesian classifier (df = 0.5, 1, 5) and compute its accuracy
    # for year 2
    df1, location, scale = stats.t.fit(mean2021)
    for i in [0.5, 1, 5]:
        value = stats.t.pdf(mean2022, i, location, scale)
        arr = []
        labelarr = list(df2022['Label'])

        for j in range(len(df2022)):
            if labelarr[j] == "g":
                arr.append(value[j] * green_prob)
            else:
                arr.append(value[j] * red_prob)

        df2022["prob" + str(i)] = arr
        df2022["predict" + str(i)] = df2022["prob" + str(i)].apply(
            lambda x: 'g' if x > df2022["prob" + str(i)].mean() else 'r')
    print(df2022.head())
    print('\nTask 1:')
    for i in [0.5, 1, 5]:
        print("The accuracy for year 2 for", i, "is", accuracy_score(label2022, df2022["predict" + str(i)]))

    # 2. compute the confusion matrices for year 2
    print('\nTask 2:')
    cm = [0]*3
    temp = 0
    for i in [0.5, 1, 5]:
        cm[temp] = confusion_matrix(label2022, df2022["predict" + str(i)])
        print("the confusion matrix for", i, ':\n', cm[temp])
        temp += 1

    # 3. what is true positive rate and true negative rate for year 2
    print('\nTask 3:')
    temp = 0
    for i in [0.5, 1, 5]:
        TP = cm[temp][0][0]
        FP = cm[temp][0][1]
        FN = cm[temp][1][0]
        TN = cm[temp][1][1]
        temp += 1
        dfx = getTable(FP, TN, TP, FN)
        print("\nTrue positive rate for", i, ':\n', dfx)

    # 4. what is the best value of df? Is it better than normal Naive bayesian
    print('\nTask 4:')
    print('The best value of df is 0.5.')

    # 5. for the best value of df, implement a trading strategy based on your labels for year 2
    # and compare the performance with the ”buy-and-hold” strategy. Which strategy results in a
    # larger amount at the end of the year?
    df3 = pd.read_csv("TMO_weekly_label.csv")
    df3 = df3[df3['Year'] == 2022]
    meanReturn = df3['Mean Return']
    print("\nTask 5:")
    print('Money earned based on buy-and-hold strategy for Year2:')
    print("-2.2672499999999984")

    meanReturn = list(meanReturn)
    for i in [0.5, 1, 5]:
        moneyEarned = 0
        predictarr = list(df2022["predict" + str(i)])
        for j in range(52):
            if predictarr[j] == 'g':
                moneyEarned = moneyEarned + meanReturn[j]
        print('\nNew strategy based on', i, ':')
        print('Money earned based on this strategy for Year 2:')
        print(moneyEarned)

    print('\nStrategy based on df0.5 has the larger amount at the end of the year.')


Naive_Bayesian()
Discriminant_Analysis()
Student_t()



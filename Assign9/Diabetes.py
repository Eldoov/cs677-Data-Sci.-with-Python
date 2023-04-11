import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def getName():
    f1 = 'Glucose'
    f2 = 'BloodPressure'
    f3 = 'SkinThickness'
    f4 = 'Insulin'
    return f1, f2, f3, f4


# Question 1
def Q1():
    df = pd.read_csv("Diabetes dataset.csv")
    f1, f2, f3, f4 = getName()
    df = df[[f1, f2, f3, f4, 'Outcome']]
    df0 = df[df['Outcome'] == 0]
    df1 = df[df['Outcome'] == 1]

    # 1. load the data into Pandas dataframe. Extract two dataframes with the above 4 features:
    # df 0 for healthy patients (OUTCOME = 0) and df 1 for unhealthy patients (OUTCOME = 1)
    print('\nTask 1:')
    print(df0.head())
    print(df1.head())

    # 2. for each dataset, construct the visual representations of corresponding correlation
    # matrices M0 (from df 0) and M1 (from df 1) and save the plots into two separate files
    figure = plt.figure()
    axes = figure.add_subplot(111)
    ax = axes.matshow(df0[[f1, f2, f3, f4]].corr(),
                      interpolation='nearest')
    figure.colorbar(ax)
    plt.savefig('df0.png')
    plt.show()
    df0[[f1, f2, f3, f4]].corr()

    figure = plt.figure()
    axes = figure.add_subplot(111)
    ax = axes.matshow(df1[[f1, f2, f3, f4]].corr(),
                      interpolation='nearest')
    figure.colorbar(ax)
    plt.savefig('df1.png')
    plt.show()
    df1[[f1, f2, f3, f4]].corr()

    # 3. examine your correlation matrix plots visually and answer the following:
    print('\nTask 3:')
    print('(a) which features have the highest correlation for healthy patients?')
    print('BloodPressure and Insulin have the highest correlation for healthy patients')
    print('\n(b) which features have the lowest correlation for healthy patients?')
    print('Glucose and SkinThickness have the lowest correlation for healthy patients')
    print('\n(c) which features have the highest correlation for unhealthy patients?')
    print('BloodPressure and Insulin have the highest correlation for unhealthy patients')
    print('\n(d) which features have the lowest correlation for unhealthy patients?')
    print('Glucose and SkinThickness have the lowest correlation for unhealthy patients')
    print('\n(e) are results the same for both cases?')
    print('\nThey are very similar.')

    # 4. for each class and for each feature f1, f2, f3, f4, compute its mean μ() and standard
    # deviation σ(). Round the results to 2 decimal places and summarize them in a table as shown below:
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

        avg_f1 = round(dfx[f1].mean(), 2)
        std_f1 = round(dfx[f1].std(), 2)
        avg_f2 = round(dfx[f2].mean(), 2)
        std_f2 = round(dfx[f2].std(), 2)
        avg_f3 = round(dfx[f3].mean(), 2)
        std_f3 = round(dfx[f3].std(), 2)
        avg_f4 = round(dfx[f4].mean(), 2)
        std_f4 = round(dfx[f4].std(), 2)
        classification.loc[len(classification.index)] = [name[i], avg_f1, std_f1, avg_f2, std_f2, avg_f3,
                                                         std_f3, avg_f4, std_f4]
    print("\nTask 4:")
    print(classification)

    # 5. examine your table. Are there any obvious patterns in the distribution of each class
    print('\nTask 5:')
    print('The results values of unhealthy patients in every class are always higher.')


# Question 2 (Question 3?)
def Q2():
    df = pd.read_csv("Diabetes dataset.csv")
    f1, f2, f3, f4 = getName()
    X = df[[f1, f2, f3, f4]]
    Y = df['Outcome']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)
    cm = [0] * 7

    # 1. apply Logistic regression
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(x_train, y_train)
    cm[0] = confusion_matrix(y_test, log_reg_classifier.predict(x_test))

    # 2. apply k-NN (k = 1, 3, 5)
    temp = 1
    for k in range(1, 6, 2):
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(x_train, y_train)
        cm[temp] = confusion_matrix(y_test, knn_classifier.predict(x_test))
        temp += 1


    # 3. apply Naive-Bayesian classifier
    NB_classifier = GaussianNB().fit(x_train, y_train)
    cm[4] = confusion_matrix(y_test, NB_classifier.predict(x_test))

    # 4. apply Linear Discriminant
    lda_classifier = LDA().fit(x_train, y_train)
    cm[5] = confusion_matrix(y_test, lda_classifier.predict(x_test))

    # 5. apply Quadratic Discriminant
    qda_classifier = QDA().fit(x_train, y_train)
    cm[6] = confusion_matrix(y_test, qda_classifier.predict(x_test))

    # 6. compute its confusion matrices and summarize results in table below:
    result_table = pd.DataFrame(columns=['Method', 'TP', 'FP', 'FN', 'TN', 'Accuracy', 'TPR', 'TNR'])
    method = ['Logistic Reg.', 'k-NN (k = 1)', 'k-NN (k = 3)', 'k-NN (k = 5)',
                 'Naive Bayesian NB', 'Linear Discr.', 'Quadr. Discr.']
    for i in range(7):
        TP = cm[i][0][0]
        FP = cm[i][0][1]
        FN = cm[i][1][0]
        TN = cm[i][1][1]
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        result_table.loc[len(result_table.index)] = [method[i], TP, FP, FN, TN, ACC, TPR, TNR]

    print('\n\nQuestion 2:')
    print(result_table)

    # 7. examine your results and correlation matrices. Any conclusions?
    print('The Logistic Reg. has the highest TPR, yet the Naive Bayesian has the highest TNR.')
    print('The k-NN (k = 1) has the lowest TPR and lowest TNR.')


Q1()
Q2()
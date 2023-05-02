import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import warnings
warnings.filterwarnings('ignore')


def changeSample(size):
    df = pd.read_csv('Airlines.csv')
    samp = df.sample(size)
    samp.to_csv('samp.csv')
    print('Sample size has changed to', size)


def getData():
    df = pd.read_csv('samp.csv')

    X1 = df[['Airline', 'AirportFrom', 'AirportTo']]
    X2 = df.drop(['Airline', 'AirportFrom', 'AirportTo', 'Delay'], axis=1)
    X1 = pd.get_dummies(X1, drop_first=True, dtype=int)

    scaler = StandardScaler()
    scaler.fit(X2)
    X = scaler.transform(X2)
    X = pd.DataFrame(X, index=X2.index, columns=X2.columns)
    X = pd.concat([X, X1], axis=1)
    X.to_csv('X.csv')
    Y = df['Delay'].values
    return X, Y


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
        ACC = "%.2f%%" % (ACC * 100)
        TPR = "%.2f%%" % (TPR * 100)
        TNR = "%.2f%%" % (TNR * 100)
        return TP, FP, FN, TN, TPR, TNR, ACC
    return dfx


def predictDelay1():
    changeSample(30000)
    cm = []
    X, Y = getData()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
    result_table = pd.DataFrame(columns=['Method', 'TP', 'FP', 'FN', 'TN', 'Accuracy', 'TPR', 'TNR'])
    method = ['Logistic Reg.', 'k-NN (k = 1)', 'k-NN (k = 3)', 'k-NN (k = 5)', 'Naive Bayesian', 'Linear Discr.',
              'Quadr. Discr.', 'Decision Tree', 'Random Forest']

    # 0. apply Logistic regression
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(x_train, y_train)
    cm.append(confusion_matrix(y_test, log_reg_classifier.predict(x_test)))
    print('Finishing Logistic regression...')

    # 1-3. apply k-NN (k = 1, 3, 5)
    temp = 1
    for k in range(1, 6, 2):
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(x_train, y_train)
        cm.append(confusion_matrix(y_test, knn_classifier.predict(x_test)))
        print('Finishing k-NN (k =', k, ')...')

    # 4. apply Naive-Bayesian classifier
    NB_classifier = GaussianNB().fit(x_train, y_train)
    cm.append(confusion_matrix(y_test, NB_classifier.predict(x_test)))
    print('Finishing Naive-Bayesian...')

    # 5. apply Linear Discriminant
    lda_classifier = LDA().fit(x_train, y_train)
    cm.append(confusion_matrix(y_test, lda_classifier.predict(x_test)))
    print('Finishing Linear Discriminant...')

    # 6. apply Quadratic Discriminant
    qda_classifier = QDA().fit(x_train, y_train)
    cm.append(confusion_matrix(y_test, qda_classifier.predict(x_test)))
    print('Finishing Quadratic Discriminant...')

    # 7. Use Decision Tree
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    cm.append(confusion_matrix(y_test, prediction))
    print('Finishing Decision Tree...')

    # 8. Use Random Forest classifier
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

    rf = RandomForestClassifier(n_estimators=best_n, max_depth=best_max)
    rf.fit(x_train, y_train)
    cm.append(confusion_matrix(y_test, rf.predict(x_test)))
    print('Finishing Random Forest...')

    for i in range(9):
        TP, FP, FN, TN, ACC, TPR, TNR = getTable(cm, i, True)
        result_table.loc[len(result_table.index)] = [method[i], TP, FP, FN, TN, ACC, TPR, TNR]

    print('\nOverall:')
    print(result_table)


def predictDelay2():
    changeSample(8000)
    cm = []
    X, Y = getData()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

    for i in range(3):
        if i == 0:
            # linear SVM.
            svm_classifier = svm.SVC(kernel='linear')
            kern = 'linear'
        elif i == 1:
            # Gaussian SVM
            svm_classifier = svm.SVC(kernel='rbf')
            kern = 'Gaussian'
        else:
            # SVM degree 2
            svm_classifier = svm.SVC(kernel='poly', degree=2)
            kern = 'polynomial'

        svm_classifier.fit(x_train, y_train)
        predicted = svm_classifier.predict(x_test)
        accuracy = svm_classifier.score(x_test, y_test)
        cm.append(confusion_matrix(y_test, predicted))
        dfx = getTable(cm, i)

        print('\n', i+1, ':')
        print('Implement a', kern, 'SVM:')
        print('The accuracy is', accuracy)
        print('Confusion matrix is:')
        print(cm[i])
        print(dfx)


predictDelay1()
predictDelay2()
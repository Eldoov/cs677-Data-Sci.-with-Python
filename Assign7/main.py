# CS677 assignment 7
# by Zuowen Tang
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# Question 1
def Q1_Task():
    # 1. load the data into Pandas dataframe. Extract two dataframes
    # with the above 4 features: df 0 for surviving patients (DEATH EVENT = 0) and df 1 for
    # deceased patients (DEATH EVENT = 1)
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    df = df[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets", "DEATH_EVENT"]]
    df0 = df[df["DEATH_EVENT"] == 0]
    df1 = df[df["DEATH_EVENT"] == 1]
    print(df0.head())
    print(df1.head())

    # 2. for each dataset, construct the visual representations of correponding correlation
    # matrices M0 (from df 0) and M1 (from df 1) and save the plots into two separate files
    figure = plt.figure()
    axes = figure.add_subplot(111)
    ax = axes.matshow(df0[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]].corr(),
                      interpolation='nearest')
    figure.colorbar(ax)
    plt.show()
    df0[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]].corr()

    figure = plt.figure()
    axes = figure.add_subplot(111)
    ax = axes.matshow(df1[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]].corr(),
                      interpolation='nearest')
    figure.colorbar(ax)
    plt.show()
    df1[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]].corr()

    # 3. examine your correlation matrix plots visually and answer the following:
    # (a) which features have the highest correlation for surviving patients?
    # (b) which features have the lowest correlation for surviving patients?
    # (c) which features have the highest correlation for deceased patients?
    # (d) which features have the lowest correlation for deceased patients?
    # (e) are results the same for both cases?

    print("a) serum_sodium and creatinine_phosphokinase have the highest correlation for surviving patients.\n"
          "b) serum_sodium and serum_creatinine have the lowest correlation for surviving patients.\n"
          "c) serum_sodium and creatinine_phosphokinase have the highest correlation for deceased patients.\n"
          "d) serum_sodium and serum_creatinine have the lowest correlation for deceased patients.\n"
          "e) No they are slightly different.\n")


def sse(X, Y, degree):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)
    weights = np.polyfit(X_train, y_train, degree)
    print('(b) wights:', weights)
    model = np.poly1d(weights)
    prodict = model(X_test)
    print('(e) the corresponding loss function', mean_squared_error(y_test, prodict) * len(y_test))
    return mean_squared_error(y_test, prodict) * len(y_test)


def Q2Q3_Task():
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    df = df[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets", "DEATH_EVENT"]]
    df0 = df[df["DEATH_EVENT"] == 0]
    df1 = df[df["DEATH_EVENT"] == 1]

    X_0 = df0['platelets']
    Y_0 = df0['serum_creatinine']
    X_1 = df1['platelets']
    Y_1 = df1['serum_creatinine']

    print('1. y = ax + b (simple linear regression)')
    print('when the dgree =1, the suriving patients :')
    sse_0_1 = sse(X_0, Y_0, 1)
    print('when the dgree =1, the deceased patients :')
    sse_1_1 = sse(X_1, Y_1, 1)
    print('\n')

    print('2. y = ax2 + bx + c (quadratic)')
    print('when the dgree =2, the suriving patients :')
    sse_0_2 = sse(X_0, Y_0, 2)
    print('when the dgree =2, the deceased patients :')
    sse_1_2 = sse(X_1, Y_1, 2)
    print('\n')

    print('3. y = ax3 + bx2 + cx + d (cubic spline)')
    print('when the dgree =3, the suriving patients :')
    sse_0_3 = sse(X_0, Y_0, 3)
    print('when the dgree =3, the deceased patients :')
    sse_1_3 = sse(X_1, Y_1, 3)
    print('\n')

    print('4. y = a log x + b (GLM - generalized linear model)')
    print('when the dgree =1, the suriving patients :')
    sse_0_log_x = sse(np.log(X_0), Y_0, 1)
    print('when the dgree =1, the deceased patients :')
    sse_1_log_x = sse(np.log(X_1), Y_1, 1)
    print('\n')

    print('5. log y = a log x + b (GLM - generalized linear model))')
    print('when the dgree =1, the suriving patients :')
    sse_0_log_xy = sse(np.log(X_0), np.log(Y_0), 1)
    print('when the dgree =1, the deceased patients :')
    sse_1_log_xy = sse(np.log(X_1), np.log(Y_1), 1)
    print('\n')

    Q3 = pd.DataFrame({
        'Model': ["y = ax + b", "y = ax2 + bx + c", "y = ax3 + bx2 + cx + d", 'y = a log x + b', 'log y = a log x + b'],
        'SSE (death event=0)': [sse_0_1, sse_0_2, sse_0_3, sse_0_log_x, sse_0_log_xy],
        '(death event=1)': [sse_1_1, sse_1_2, sse_1_3, sse_1_log_x, sse_1_log_xy]
    })
    print(Q3)


Q1_Task()
Q2Q3_Task()
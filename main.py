import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import log10, log
import math


def getPrice():
    df = pd.read_csv('online_retail.csv')
    df.drop(df[df['UnitPrice'] < 1].index, inplace=True)
    df2 = df['UnitPrice'].astype(str).str[:1]
    df2 = df2.astype(int)
    return df2


def benford(n):
    return log10(n+1) - log10(n)


def model1(p):
    temp = int(len(p) / 9)
    res = []
    for i in range(1, 10):
        res = res + [i] * temp
    return res


def model2(p):
    x = [i for i in range(1, 10)]
    num = [benford(i) for i in x]
    res = []
    for i in range(1, 10):
        temp = len(p) * num[i - 1]
        res = res + [i] * int(temp)
    return res


# 1. plot 3 histograms for the frequencies for real distribution,
#    equal-weight and Bernford (for each digit)
def Task1():
    # Real Distribution
    p = getPrice()
    p = list(p)

    # Equal-weight Distribution
    # Benford Distribution
    data = [0]*3
    data[0], data[1], data[2] = p, model1(p), model2(p)

    colors = ['purple', 'limegreen', 'tomato']
    labels = ['Real Distribution', 'Equal-weight', 'Benford']
    plt.hist(data, bins=np.arange(0.5, 10.5), histtype='bar', color=colors, label=labels)
    plt.legend(prop={'size': 12})
    plt.xlim(0.5, 9.5)
    plt.title('Histogram for Task 1')
    plt.xlabel('Digits')
    plt.ylabel('Frequency')
    plt.savefig('hist1.png')
    plt.show()


# 2. plot 3 histograms for the relative errors for Models 1 and 2 (for each digit)
def Task2():
    p = getPrice()
    measured_val = list(p.value_counts())
    p0 = list(p)

    # Models1
    p2 = int(len(p0)/9)
    true_val = [p2]*9

    abs_err, relative_err = [0]*9, [0]*9
    for i in range(9):
        abs_err[i] = abs(true_val[i] - measured_val[i])
        relative_err[i] = abs_err[i]/true_val[i]

    data = [0]*2
    data[0] = relative_err

    true_val, abs_err, relative_err = [0] * 9, [0] * 9, [0] * 9
    # Models 2
    p3 = model2(p)
    for i in range(1, 10):
        true_val[i-1] = p3.count(i)

    for i in range(9):
        abs_err[i] = abs(true_val[i] - measured_val[i])
        relative_err[i] = abs_err[i] / true_val[i]
    data[1] = relative_err


    labels = [i for i in range(1, 10)]
    x = np.arange(len(labels))
    width = 0.25
    plt.bar(x - width / 2, data[0], width, label='Model 1')
    plt.bar(x + width / 2, data[1], width, label='Model 2')
    plt.xlabel('Digit')
    plt.ylabel('Relative Errors')
    plt.title('Histogram for Task 2')
    plt.xticks(x, labels=labels)
    plt.legend()
    plt.savefig('hist2.png')
    plt.show()


# 3. compute RMSE (root mean squared error) for model 1, 2.
# Which model is closer to the real distribution?
def Task3():
    p = getPrice()
    actual_val = list(p.value_counts())
    p0 = list(p)
    predicted_val = [0]*2

    p2 = int(len(p0) / 9)
    predicted_val[0] = [p2] * 9

    p3 = model2(p)
    predicted_val[1] = [0] * 9
    for i in range(1, 10):
        predicted_val[1][i - 1] = p3.count(i)

    for i in range(2):
        MSE = np.square(np.subtract(actual_val, predicted_val[i])).mean()
        RMSE = math.sqrt(MSE)
        print('RMSE of Model ' + str(i+1) + ' is', RMSE)


# 4. take 3 countires of your choice:
# one from Asia, one from Europe and one from the Middle East.
# For each of these countries do the following:
def Task4():
    df = pd.read_csv('online_retail.csv')
    df.drop(df[df['UnitPrice'] < 1].index, inplace=True)
    df['UnitPrice'] = df['UnitPrice'].astype(str).str[:1]
    df['UnitPrice'] = df['UnitPrice'].astype(int)
    Israel = df[df['Country'] == 'Israel']
    Japan = df[df['Country'] == 'Japan']
    Germany = df[df['Country'] == 'Germany']
    Israel = Israel['UnitPrice']
    Japan = Japan['UnitPrice']
    Germany = Germany['UnitPrice']

    data, data_Israel, data_Japan, data_Germany = [0]*3, [0]*3, [0]*3, [0]*3
    #(a) compute F, P and π
    data_Israel[0], data_Israel[1], data_Israel[2] = list(Israel), model1(Israel), model2(Israel)
    data_Japan[0], data_Japan[1], data_Japan[2] = list(Japan), model1(Japan), model2(Japan)
    data_Germany[0], data_Germany[1], data_Germany[2] = list(Germany), model1(Germany), model2(Germany)
    data[0], data[1], data[2] = data_Israel, data_Japan, data_Germany
    f, p, pi = [0]*9,[0]*9,[0]*9
    for i in range(3):
        if i == 0:
            country = 'Israel'
        if i == 1:
            country = 'Japan'
        if i == 2:
            country = 'Germany'
        for j in range(9):
            f[j] = data[i][0].count(j + 1)
            p[j] = data[i][1].count(j + 1)
            pi[j] = data[i][2].count(j + 1)

        print('F, P, and pi for ' + country + ' is (from 1 - 9)')
        print('F:', f)
        print('P:', p)
        print('pi:', pi)

        # (b) using RMSE as a ”distance” metric, for which of these chosen three countries
        #    is the distribution ”closest” to equal weight P?
        MSE = np.square(np.subtract(f, p)).mean()
        RMSE = math.sqrt(MSE)
        print('RMSE of equal-weight in ' + country + ' is', RMSE)
        print('*' * 30, '\n')



    for i in range(3):
        colors = ['purple', 'limegreen', 'tomato']
        labels = ['Real Distribution', 'Equal-weight', 'Benford']
        plt.hist(data[i], bins=np.arange(0.5, 10.5), histtype='bar', color=colors, label=labels)
        plt.legend(prop={'size': 12})
        plt.xlim(0.5, 9.5)

        if i == 0:
            country = 'Israel'
        if i == 1:
            country = 'Japan'
        if i == 2:
            country = 'Germany'

        plt.title('Histogram for ' + country)
        plt.xlabel('Digits')
        plt.ylabel('Frequency')
        plt.savefig(str(i) + '.png')
        plt.show()


def Driver():
    Task1()
    Task2()
    Task3()
    Task4()


Driver()
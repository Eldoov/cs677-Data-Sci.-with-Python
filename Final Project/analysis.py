import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def numAnalysis():
    df = pd.read_csv('Airlines.csv')
    df_num = df[['Flight', 'DayOfWeek', 'Time', 'Length', 'Delay']]

    sns.heatmap(df_num.corr(), annot=True)
    plt.savefig('corr.png')
    plt.show()

    sns.histplot(data=df_num, x='Flight', hue='Delay', kde=True)
    plt.savefig('flightDelay.png')
    plt.show()

    sns.histplot(data=df_num, x='Time', hue='Delay', kde=True)
    plt.savefig('timeDelay.png')
    plt.show()

    sns.histplot(data=df_num, x='Length', hue='Delay', kde=True)
    plt.savefig('lengthDelay.png')
    plt.show()


def textAnalysis():
    df = pd.read_csv('Airlines.csv')

    sns.countplot(data=df, x='Airline', hue='Delay')
    plt.savefig('airlineDelay.png')
    plt.show()

    sns.countplot(data=df, x='DayOfWeek', hue='Delay')
    plt.savefig('weekdayDelay.png')
    plt.show()


textAnalysis()





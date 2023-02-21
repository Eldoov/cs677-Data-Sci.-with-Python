import pandas as pd
import matplotlib.pyplot as plt


def plot(yearNum):
    df = pd.read_csv('TMO_weekly_label.csv')
    year = df[df['Year'] == yearNum]
    green = year[year['Label'] == 'g']
    red = year[year['Label'] == 'r']

    x = green['Mean Return']
    y = green['Volatility']
    plt.scatter(x, y, c='green')

    x = red['Mean Return']
    y = red['Volatility']
    plt.scatter(x, y, c='red')

    plt.title(str(yearNum))
    plt.xlabel('Avg. of Daily Returns')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.savefig(str(yearNum)+'.png')
    plt.show()


plot(2021)
plot(2022)

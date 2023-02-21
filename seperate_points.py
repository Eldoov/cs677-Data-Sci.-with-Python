import pandas as pd
import matplotlib.pyplot as plt

# 1. take year 1 and examine the plot of your labels.
# Construct a reduced dataset by removing some green and red points
# so that you can draw a line separating the points.
# Compute the equation of such a line (many solutiuons are possible)
def Task1(yearNum):
    df = pd.read_csv('TMO_weekly_label.csv')
    year = df[df['Year'] == yearNum]
    green = year[year['Label'] == 'g']
    red = year[year['Label'] == 'r']

    if yearNum == 2021:
        filter = green[green['Mean Return'] >= 0]
        y = filter['Volatility']
        x = filter['Mean Return']
        plt.scatter(x, y, c='green')

        filter = red[red['Mean Return'] <= 0]
        y = filter['Volatility']
        x = filter['Mean Return']
        plt.scatter(x, y, c='red')

    if yearNum == 2022:
        x = green['Mean Return']
        y = green['Volatility']
        plt.scatter(x, y, c='green')

        x = red['Mean Return']
        y = red['Volatility']
        plt.scatter(x, y, c='red')

    x1, y1 = [0.05, 0.05], [0.3, 3]
    plt.plot(x1, y1, marker='o')

    plt.title(str(yearNum))
    plt.xlabel('Avg. of Daily Returns')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.savefig(str(yearNum)+'*.png')
    plt.show()


# 2. Take this line and use it to assign labels for year 2
def Task2():
    df = pd.read_csv('TMO_weekly_label.csv')
    year = df[df['Year'] == 2022]
    filter = year[year['Mean Return'] >= 0]
    filter = filter.replace('r','g')

    filter2 = year[year['Mean Return'] <= 0]
    filter2 = filter2.replace('g', 'r')

    x = filter['Mean Return']
    y = filter['Volatility']
    plt.scatter(x, y, c='green')

    x = filter2['Mean Return']
    y = filter2['Volatility']
    plt.scatter(x, y, c='red')
    plt.title('Year 2 (Assign labels based on last year)')
    plt.xlabel('Avg. of Daily Returns')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.savefig('2022_new.png')
    plt.show()


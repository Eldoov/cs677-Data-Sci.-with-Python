import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import warnings
warnings.filterwarnings('ignore')


# Set option as display all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
# Load data by Pandas
crime_rate_df = pd.read_csv('dataset/boston_crime_2021-2022.csv', dtype=str)
crime_list = ['LARCENY', 'M/V ACCIDENT', 'LIQUOR', 'INCEST', 'MANSLAUGHTER', 'MISSING PERSON',
                  'PROPERTY - LOST', 'MURDER', 'FRAUD', 'PROSTITUTION', 'RAPE', 'ROBBERY', 'ASSAULT',
                  'SICK/INJURED/MEDICAL', 'TOWED MOTOR VEHICLE', 'TRESPASSING', 'VIOLATION', 'ANIMAL',
                  'AUTO THEFT', 'FIREARM/WEAPON', 'HUMAN TRAFFICKING', 'DRUGS', 'SEX OFFENSE', 'ARSON',
                  'VANDALISM', 'SEARCH WARRANT', 'KIDNAPPING', 'DEATH INVESTIGATION', 'CHILD ABUSE', 'HARASSMENT']


def getData():
    df_list = []
    for i in range(2021, 2023):
        df_list.append(crime_rate_df[crime_rate_df['YEAR'] == str(i)])

    for i in range(len(df_list)):
        for j in range(len(crime_list)):
            df_list[i].loc[df_list[i]['OFFENSE_DESCRIPTION'].str.contains(crime_list[j]), 'GROUP'] = crime_list[j]
    return df_list


# What sorts of crime happened most often during 2021 - 2022?
def Part1():
    for j in range(len(crime_list)):
        crime_rate_df.loc[crime_rate_df['OFFENSE_DESCRIPTION'].str.contains(crime_list[j]), 'GROUP'] = crime_list[j]
    bar = crime_rate_df.GROUP.value_counts()
    x = list(bar.index)
    y = list(bar)

    fig, ax = plt.subplots(figsize=(15, 10))
    width = 0.75  # the width of the bars
    ind = np.arange(len(y))  # the x locations for the groups
    ax.barh(ind, y, width)
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(x, fontsize='small', minor=False)
    for i, v in enumerate(y):
        ax.text(v + .25, i + .25, str(v), fontweight='bold')  # add value labels into bar
    plt.title('Crimes happened most often in Boston during 2021-2022')
    plt.xlabel('Crime Rate')
    plt.tight_layout()
    plt.savefig('all.png')
    plt.show()


# What sorts of crimes happened most often in each year?
def Part2():
    df_list = getData()
    for i in range(len(df_list)):
        year = i + 2021
        bar = df_list[i].GROUP.value_counts()
        x = list(bar.index)
        y = list(bar)

        fig, ax = plt.subplots(figsize=(15, 10))
        width = 0.75  # the width of the bars
        ind = np.arange(len(y))  # the x locations for the groups
        ax.barh(ind, y, width)
        ax.set_yticks(ind + width / 2)
        ax.set_yticklabels(x, fontsize='small', minor=False)
        for i, v in enumerate(y):
            ax.text(v + .25, i + .25, str(v), fontweight='bold')  # add value labels into bar
        plt.title('Crimes happened most often in Boston in year '+str(year))
        plt.xlabel('Crime Rate')
        plt.tight_layout()
        plt.savefig(str(year) + '.png')
        plt.show()


# What is the crime rate for each year?
# Does the crime rate increase or decrease every year?
# Or there's no obvious change?
def Part3():
    df_list = getData()
    crime_rate = []
    for i in range(len(df_list)):
        crime_rate.append(len(df_list[i].index))
    print(crime_rate)

    Incidents = [101338, 98888, 87184, 70894, 71721, 73852]
    Year = ["2017", "2018", "2019", "2020", "2021", "2022"]
    position = range(len(Incidents))

    plt.bar(position, Incidents, color=('steelblue', 'steelblue', 'steelblue',
                                        'steelblue', 'orange', 'orange'))
    plt.xticks(position, Year)
    plt.title("Crime rate from 2017 to 2021")
    plt.xlabel("")
    plt.ylabel("Incidents")
    plt.tight_layout()
    plt.savefig('2017-2022.png')
    plt.show()


# In one week, which day does the crime happen the most?
# How about where does the crime happen most often?
def Part4():
    #print(crime_rate_df.head())
    #print(crime_rate_df.DAY_OF_WEEK.value_counts())
    #print(crime_rate_df.STREET.value_counts())

    Incidents = [21052, 20609, 21286, 21171, 22229, 20850, 18376]
    Weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    position = range(len(Incidents))

    plt.bar(position, Incidents, color=('steelblue', 'steelblue', 'steelblue',
                                       'steelblue', 'darkred', 'steelblue', 'forestgreen'))
    plt.xticks(position, Weekday)
    plt.title("Crime rate in different weekday")
    plt.xlabel("")
    plt.ylabel("Incidents")
    plt.tight_layout()
    #plt.savefig('weekday.png')
    plt.show()

    labels = ['SUDBURY ST', 'BOYLSTON ST', 'W BROADWAY', 'HYDE PARK AVE',
             'CENTRE ST', 'BLUE HILL AVE', 'HARRISON AVE', 'GIBSON ST', 'WASHINGTON ST']
    share = [1686, 1914, 2321, 2567, 3105, 4446, 4446, 4487, 11291]
    position = range(len(share))

    plt.bar(position, share, color=('forestgreen', 'steelblue', 'steelblue', 'steelblue',
                                    'steelblue', 'steelblue', 'steelblue', 'steelblue', 'darkred'))
    plt.xticks(position, labels, rotation = 60)
    plt.title("Crime rate in different street")
    plt.xlabel("")
    plt.ylabel("Incidents")
    plt.tight_layout()
    #plt.savefig('street.png')
    plt.show()

    street = list(crime_rate_df.STREET.value_counts())
    print(crime_rate_df.STREET.value_counts())
    print(144893 - 36263)
    print((11291 + 4487 + 4446 + 4446 + 3105 + 2567 + 2321 + 1914 + 1686))
    count = 0
    for i in range(len(street)):
        if count >= (144893 * 0.2497):
            count = 0
            print(i)
        count = count + street[i]

    labels = ['SUDBURY ST', 'BOYLSTON ST', 'W BROADWAY', 'HYDE PARK AVE',
             'CENTRE ST', 'BLUE HILL AVE', 'HARRISON AVE', 'GIBSON ST', 'WASHINGTON ST']
    share = [1686, 1914, 2321, 2567, 3105, 4446, 4446, 4487, 11291]
    explode = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    plt.style.use('ggplot')
    colors = ['#cbffff', '#b1f3fa', '#97e6f6', '#7ed9f4',
              '#66cbf3', '#52bdf1', '#43adee', '#3f9eea', '#3f7fcc']
    plt.pie(x=share, explode=explode, labels=labels, autopct='%.2f%%', colors=colors, shadow=True, startangle=90)
    plt.axis('equal')
    plt.title("Percentage of Where Crime Happens")
    plt.savefig('pie-chart-street.png')
    plt.tight_layout()
    plt.show()

    labels = ['Others', 'MEDICAL ASSIST', 'FRAUD', 'PROPERTY - LOST', 'VANDALISM',
             'TOWED VEHICLE', 'ASSAULT', 'M/V ACCIDENT', 'LARCENY']
    share = [18918, 4131, 4326, 4417, 6189, 6405, 8919, 11307, 17046]
    explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
    plt.style.use('ggplot')
    colors = ['#ebebeb', '#ec997c', '#f2b680', '#f2d38d', '#ecf0a5',
              '#bbdfa0', '#8dcca1', '#66b7a1', '#47a19f']
    plt.pie(x=share, explode=explode, labels=labels, autopct='%.2f%%', colors=colors, shadow=True, startangle=90)
    plt.axis('equal')
    plt.title("Percentage of Crime Incidents")
    #plt.savefig('pie-chart.png')
    #plt.show()


# How many crime incidents involve guns?
def Part5():
    #print(crime_rate_df.head())
    shooting = list(crime_rate_df.SHOOTING.value_counts())
    shooting[1] = shooting[1] + shooting[2]
    shooting.remove(859)
    print(shooting)

    labels = ['Without Shooting', 'With Shooting']

    plt.pie(shooting, labels=labels, autopct='%1.2f%%')
    plt.title("Crime with shooting from 2017 to 2021.")
    plt.savefig('shooting.png')
    plt.show()


Part4()




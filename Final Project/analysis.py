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
crime_rate_df = pd.read_csv('dataset/boston_crime_rate_2017-2022.csv', dtype=str)
crime_rate_df = crime_rate_df[crime_rate_df['YEAR'] != "2022"]
shooting_df = pd.read_csv('dataset/shooting_incident.csv')
homicide_weapon_df = pd.read_csv('dataset/Homicide Weapon.csv')

crime_list = ['LARCENY', 'M/V ACCIDENT', 'LIQUOR', 'INCEST', 'MANSLAUGHTER', 'MISSING PERSON',
                  'PROPERTY - LOST', 'MURDER', 'FRAUD', 'PROSTITUTION', 'RAPE', 'ROBBERY', 'ASSAULT',
                  'SICK/INJURED/MEDICAL', 'TOWED MOTOR VEHICLE', 'TRESPASSING', 'VIOLATION', 'ANIMAL',
                  'AUTO THEFT', 'FIREARM/WEAPON', 'HUMAN TRAFFICKING', 'DRUGS', 'SEX OFFENSE', 'ARSON',
                  'VANDALISM', 'SEARCH WARRANT', 'KIDNAPPING', 'DEATH INVESTIGATION', 'CHILD ABUSE', 'HARASSMENT']


def getData():
    df_list = []
    for i in range(2017, 2022):
        df_list.append(crime_rate_df[crime_rate_df['YEAR'] == str(i)])

    for i in range(len(df_list)):
        for j in range(len(crime_list)):
            df_list[i].loc[df_list[i]['OFFENSE_DESCRIPTION'].str.contains(crime_list[j]), 'GROUP'] = crime_list[j]
    return df_list


# What sorts of crime happened most often during 2017 - 2021?
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
    plt.title('Crimes happened most often in Boston during 2017-2021')
    plt.xlabel('Crime Rate')
    plt.tight_layout()
    plt.savefig('all.png')
    plt.show()


# What sorts of crimes happened most often in each year?
def Part2():
    df_list = getData()
    for i in range(len(df_list)):
        year = i + 2017
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
    #print(crime_rate)
    crime_rate_year = pd.DataFrame({"Incidents": [101338, 98888, 87184, 70894, 71721]},
                                index=["2017", "2018", "2019", "2020", "2021"])

    crime_rate_year.plot(kind='bar', figsize=(10, 8))
    plt.title("Crime rate from 2017 to 2021")
    plt.xlabel("")
    plt.ylabel("Incidents")
    plt.show()


# In one week, which day, month, and hour does the crime happen the most?
# How about where does the crime happen most often?
def Part4():
    #print(crime_rate_df.head())
    print(crime_rate_df.MONTH.value_counts())
    print(crime_rate_df.DAY_OF_WEEK.value_counts())
    print(crime_rate_df.HOUR.value_counts())
    print(crime_rate_df.STREET.value_counts())

    wordcloud = WordCloud(background_color="white").generate(str(crime_rate_df['STREET']))
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('Street_wordcloud.png')
    plt.show()

    X = crime_rate_df.DAY_OF_WEEK.value_counts()
    labels = ['Friday', 'Wednesday', 'Thursday', 'Tuesday', 'Monday', 'Saturday', 'Sunday']
    print(list(X))

    plt.pie(X, labels=labels, autopct='%1.2f%%')
    plt.title("Pie chart")
    plt.show()


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







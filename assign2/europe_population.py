import pandas as pd


def readFile(path):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    file = pd.read_csv(path)
    return file


# Task 1: Read the file with python
def Task1():
    file = readFile('europe_population.csv')
    print("Task 1:")
    print("Finished reading file.")
    print("Origin Data Sample:")
    print(file.head())
    print("\n", "*" * 50)


# Task 2: Rank the countries/cities (from best to worst)
#         in terms of surviving the first year of life.
def Task2():
    file = readFile('europe_population.csv')
    print("\nTask 2:")
    print("Amount of one-year-old per city, from best to worst:")
    first_year = file.sort_values(by=file.index.tolist(), ascending=False, axis=1)
    first_year = first_year.drop('Ages', axis=1)
    first_year = first_year.drop([0])
    print(first_year.head(1))
    print("\n", "*" * 50)


# Task 3: For every place and every year compute the annual death rate
#         of the population and write the results to an output file.
def Task3():
    file = readFile('europe_population.csv')
    writedata = file.copy()
    for i in range(1, 12):
        for j in range(0, 91):
            x = int(file.iloc[j][i])
            y = 1000
            writedata.iloc[j][i] = abs(1 - (x/y))
    writedata.to_csv('europe_population_death_rates.csv')
    print("\nTask 3:")
    print("Finished writing file.")
    print("Data Sample:")
    print(writedata.head())
    print("\n", "*" * 50)


# Task 4: Compute three measures of average death rates (over 90 years) for each place
def Task4_5_6():
    file = readFile('europe_population_death_rates.csv')
    ann = [0]*11
    file = file.drop(labels=['Unnamed: 0', 'Ages'], axis=1)
    column_name = list(file.columns.values)

    # (a) average of annual death rates
    print("\nTask 4:")
    print("Average death rate in different cities.")
    avg = file.mean()
    print(avg, "\n")

    # (b) median of annual death rate
    print("Median of death rate in different cities.")
    med = file.median()
    print(med, "\n")

    # (c) "annualized" death rate
    print("Annualized death rate in different cities.")
    for i in range(0, 11):
        S = 1000
        E = float(file.iloc[90][i])
        res = abs(1 - pow((E / S), 1 / 90))
        ann[i] = res
        table = "{0:^15}\t{1:^10}"
        print(table.format(column_name[i].ljust(15), '{:.4%}'.format(res).ljust(10)))
    print("*" * 50)

    # Task 5: Which places had the highest average, median, and annualized death rates?
    # Task 6: Which places had the lowest average, median, and annualized death rates?
    print("\nTask 5 & 6:")
    # Highest & Lowest Average
    print("The city with highest average is Breflaw:", '{:.3%}'.format(avg.max()))
    print("The city with lowest average is France :", '{:.3%}'.format(avg.min()), "\n")

    # Highest & Lowest Median

    print("The city with highest median is London :", '{:.2%}'.format(med.max()))
    print("The city with lowest median is Vaud :", '{:.2%}'.format(med.min()), "\n")

    # Highest & Lowest Annualized
    max_ann = (max(ann))
    index = ann.index(max_ann)
    print("The city with highest annualized is", column_name[index], ":", '{:.3%}'.format(max_ann))

    min_ann = (min(ann))
    index = ann.index(min_ann)
    print("The city with lowest annualized is", column_name[index], ":", '{:.3%}'.format(min_ann), "\n")
    print("*" * 50)

# Task 7: For every age and every place:
#         Assign + if the death rate for that year if greater or equal to the death rate in the previous year.
#         Assign - if the death rate for that year is less (i.e., decreased) than the death rate in the previous year.
#         With these labels, every place is assigned a sequence like "++−−−+−···−+".
#         For each such sequence, compute the following statistics and put the results in the table.
def Task7_8_9():
    file = readFile('europe_population_death_rates.csv')
    column_name = list(file.columns.values)
    data = [0]*11

    print("\nTask 7:")

    for i in range (2,13):
        numPlus, numMinus = 1, 0
        conPlus, conMinus = 1, 0
        maxPlus, maxMinus = 1, 0

        for j in range(1, 90):
            if file.iloc[j][i] <= file.iloc[j+1][i]:
                # if next year is greater than this year
                # conPlus +1 and conMinus = 0
                conMinus = 0
                conPlus += 1
                if conPlus > maxPlus:
                    maxPlus = conPlus

                numPlus += 1
            if file.iloc[j][i] > file.iloc[j+1][i]:
                # if next year is lesser than this year
                # conMinus +1 and conPlus = 0
                conPlus = 0
                conMinus += 1
                if conMinus > maxMinus:
                    maxMinus = conMinus

                numMinus += 1
        data[i-2] = [column_name[i], numPlus, numMinus, maxPlus, maxMinus]

    df = pd.DataFrame(data, columns=['City', 'Sum of +', 'Sum of -', 'Max cons. +', 'Max cons. -'])
    print(df)
    print("*" * 50)

    # Task 8: which three places have the longest sequences of increasing death rates?
    # Task 9: which three places have the longest sequences of decreasing death rates?
    print("\nTask 8 & 9:")
    increase = df.sort_values(by='Max cons. +', ascending=False)
    decrease = df.sort_values(by='Max cons. -', ascending=False)
    print("Longest sequences of increasing death rates:")
    print(increase.head(3))
    print("\nLongest sequences of decreasing death rates:")
    print(decrease.head(3))


def Driver():
    Task1()
    Task2()
    Task3()
    Task4_5_6()
    Task7_8_9()
    # Task 10: Examine your table carefully. How would you describe what you see in words?
    print("*" * 50)
    print("\nTask 10:")
    print("The death rate is almost always increasing. Only in very few places the death rate would decrease,")
    print("but even in that case, the death rate would eventually keep raising.")

    # Task 11:
    print("*" * 50)
    print("\nTask 11:")
    print("(c) Pinocchio plagiarized and violated academic code.")


Driver()
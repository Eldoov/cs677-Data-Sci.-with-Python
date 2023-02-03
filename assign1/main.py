# Zuowen Tang
import csv
import statistics
import tools


# Task 1: Read the file with python
def readFile(path):
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        data = list(reader)
        return data


# Task 2: Rank the countries/cities (from best to worst)
#         in terms of surviving the first year of life.
def Task2():
    data = readFile("europe_population.csv")
    fstYear = data[2]
    cityName = data[0]

    fstYear, cityName = minmax(fstYear, cityName)

    print("\n***************************************************")
    print("Task 2:\n")
    print("Amount of one-year-old per city, from best to worst:")
    for i in range(11):
        print(i + 1, cityName[i], '-', fstYear[i])


# Task 3: For every place and every year compute the annual death rate
#         of the population and write the results to an output file.
def Task3():
    data = readFile("europe_population.csv")
    writeData = data
    for i in range(1, len(data)-1):
        for j in range(1, 12):
            x = int(data[i][j])
            y = 1000
            r = abs(1 - (x/y))
            writeData[i][j] = '{:.1%}'.format(r)

    writeFile(writeData, 'europe_population_death_rates.csv')
    print("\n\n***************************************************")
    print("Task 3:\n")
    print("Finished writing file.\n")


# Task 4: Compute three measures of average death rates (over 90 years) for each place
def Task4():
    data = readFile('europe_population_death_rates.csv')
    listColumn = [0] * 91
    avg, med, annual = [0] * 12, [0] * 12, [0] * 12
    cityName = data[0]

    # (a) average of annual death rates
    print("\n***************************************************")
    print("Task 4:\n")
    print("Average death rate in different cities.")
    for i in range(1, 12):
        for j in range(1, len(data)-1):
            temp = float(data[j][i].strip('%')) / 100.0
        temp = temp / 90
        avg[i] = temp
        table = "{0:^15}\t{1:^10}"
        print(table.format(data[0][i].ljust(15), '{:.3%}'.format(temp)))

    # (b) median of annual death rate
    print("\nMedian of death rate in different cities.")
    for i in range(1, 12):
        for j in range(1, len(data)-1):
            listColumn[j-1] = float(data[j][i].strip('%')) / 100
        r = statistics.median(listColumn)
        med[i] = r
        table = "{0:^15}\t{1:^10}"
        print(table.format(data[0][i].ljust(15), '{:.2%}'.format(r)))

    # (c) "annualized" death rate
    print("\nAnnualized death rate in different cities.")
    for i in range(1, 12):
        S = 1000
        E = float(data[91][i].strip('%')) / 100
        r = abs(1 - pow((E/S), 1 / 90))
        annual[i] = r
        table = "{0:^15}\t{1:^10}"
        print(table.format(data[0][i].ljust(15), '{:.3%}'.format(r)))

    return avg, med, annual, cityName


# Task 5: Which places had the highest average, median, and annualized death rates?
# Task 6: Which places had the lowest average, median, and annualized death rates?
def Task5_6():
    avg, med, annual, cityName = Task4()
    city1, city2 = cityName.copy(), cityName.copy()

    print("\n\n***************************************************")
    print("Task 5 & 6:\n")
    # Highest & Lowest Average
    avg, cityName = minmax(avg, cityName)
    print("The city with highest average is", cityName[0], ":", '{:.3%}'.format(avg[0]))
    print("The city with lowest average is", cityName[10], ":", '{:.3%}'.format(avg[10]), "\n")

    # Highest & Lowest Median
    med, cityName = minmax(med, city1)
    print("The city with highest median is", cityName[0], ":", '{:.2%}'.format(med[0]))
    print("The city with lowest median is", cityName[10], ":", '{:.2%}'.format(med[10]), "\n")

    # Highest & Lowest Annualized
    annual, cityName = minmax(annual, city2)
    print("The city with highest annualized is", cityName[0], ":", '{:.3%}'.format(annual[0]))
    print("The city with lowest annualized is", cityName[10], ":", '{:.3%}'.format(annual[10]), "\n")


# Task 7: For every age and every place:
#         Assign + if the death rate for that year if greater or equal to the death rate in the previous year.
#         Assign - if the death rate for that year is less (i.e., decreased) than the death rate in the previous year.
#         With these labels, every place is assigned a sequence like "++−−−+−···−+".
#         For each such sequence, compute the following statistics and put the results in the table.
def Task7_8_9():
    data = readFile('europe_population_death_rates.csv')
    max, min = [0] * 12, [0] * 12
    cityName = data[0]
    cityName1 = cityName.copy()

    print("\n***************************************************")
    print("Task 7:\n")
    print("CityName     Num of +     Num of -     Max cons. +     Max cons. -")
    print("------------------------------------------------------------------")
    for i in range(1, 12):
        numPlus, numMinus = 0, 0
        arr = [0] * 90
        arrPlus, arrMinus, conPlus, conMinus = 1, 1, 1, 0

        #print("Sequence :", end='')
        for j in range(2, len(data)-1):
            thisYear = float(data[j][i].strip('%')) / 100.0
            lastYear = float(data[j-1][i].strip('%')) / 100.0
            if thisYear >= lastYear:
                #print("+",end='')
                arr[j - 2] = 1
                numPlus += 1
            if thisYear < lastYear:
                #print("-", end='')
                arr[j - 2] = 0
                numMinus += 1

        for k in range(len(arr)-1):

            if arr[k] == arr[k+1] and arr[k] == 1:
                arrPlus += 1
            if arr[k-1] != arr[k]:
                if arrPlus > conPlus:
                    conPlus = arrPlus
                arrPlus = 1

            if arr[k] == arr[k+1] and arr[k] == 0:
                arrMinus += 1
            if arr[k-1] != arr[k]:
                if arrMinus > conMinus:
                    conMinus = arrMinus
                arrMinus = 1

        if arrPlus > conPlus: conPlus = arrPlus
        if arrMinus == 0 : conPlus += 1

        max[i] = conPlus
        min[i] = conMinus

        table = "{0:^10}\t{1:^10}\t{2:^10}\t{3:^15}\t{4:^15}"
        print(table.format(data[0][i].ljust(10), numPlus, numMinus, conPlus, conMinus))

    # Task 8: which three places have the longest sequences of increasing death rates?
    # Task 9: which three places have the longest sequences of decreasing death rates?
    max, cityName = minmax(max, cityName)
    min, cityName1 = minmax(min, cityName1)

    print("\n***************************************************")
    print("Task 8 & 9:\n")
    print("Longest sequences of increasing death rates:")
    table = "{0:^10}\t{1:^15}\t{2:^15}"
    print(table.format(cityName[0],cityName[1],cityName[2]))
    print(table.format(max[0],max[1],max[2]))

    print("\nLongest sequences of decreasing death rates:")
    print(table.format(cityName1[0], cityName1[1], cityName1[2]))
    print(table.format(min[0], min[1], min[2]))


# Tools
def writeFile(writeData, path):
    with open(path, 'w') as file:
        writer = csv.writer(file)
        for i in range(len(writeData)):
            writer.writerow(writeData[i])


def minmax(myList, cityName):
    j = 0
    while j < len(myList) - 1:
        i = 0
        while i < len(myList) - (j + 1):
            if myList[i] < myList[i + 1]:
                a = myList[i]
                myList[i] = myList[i + 1]
                myList[i + 1] = a
                b = cityName[i]
                cityName[i] = cityName[i + 1]
                cityName[i + 1] = b
            i += 1
        j += 1
    return myList, cityName


# Driver
def Driver():
    print("Task 1:\n")
    print("Finished reading file.\n")

    Task2()
    Task3()
    Task5_6()
    Task7_8_9()

    # Task 10: Examine your table carefully. How would you describe what you see in words?
    print("\n***************************************************")
    print("Task 10:\n")
    print("The death rate is almost always increasing. Only in very few places the death rate would decrease,")
    print("but even in that case, the death rate would eventually keep raising.")

    # Task 11:
    print("\n***************************************************")
    print("Task 11:\n")
    print("(c) Pinocchio plagiarized and violated academic code.")


Driver()
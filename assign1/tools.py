import csv


# Task 1: Read the file with python
def readFile(path):
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        data = list(reader)
        return data


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

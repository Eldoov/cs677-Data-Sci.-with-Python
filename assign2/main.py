import pandas as pd


def readFile(path):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.options.mode.chained_assignment = None
    file = pd.read_csv(path)
    return file


# 1. load both datasets into two pandas frames
def Task1():
    dog_breed = readFile('AKC Breed Info.csv')
    dog_intellience = readFile('dog_intelligence.csv')
    print("Task 1:")
    print("Finished reading files.")
    print("Origin Data Sample:\n")
    print("Dog Breed:")
    print(dog_breed.head(), "\n")
    print("Dog Intellience:")
    print(dog_intellience.head())
    print("\n", "*" * 50)

# 2. datasets could have unknown values or outlier values.
#   • remove them
#   • impute these values by assign some new values
def Task2():
    dog_breed = readFile('AKC Breed Info.csv')
    dog_intellience = readFile('dog_intelligence.csv')
    column = [0]*10
    count = 0

    maxIndex = dog_breed.iloc[-1][0]
    for i in range(0, 6):
        for j in range(0, maxIndex):
            if dog_breed.iloc[j][i] == "na" or dog_breed.iloc[j][i] == "not found":
                dog_breed.iloc[j, i] = 0
                column[count] = j
                count += 1

    convert_dict = {'height_low_inches': float,
                    'height_high_inches': float,
                    'weight_low_lbs': float,
                    'weight_high_lbs': float
                    }
    dog_breed = dog_breed.astype(convert_dict)
    avg = dog_breed.mean(numeric_only=True)

    for i in range(0, 2):
        if column[i] == 0:
            break
        c = column[i]
        dog_breed.iloc[c, 2] = 17.72
        dog_breed.iloc[c, 3] = 20.67
        dog_breed.iloc[c, 4] = 42.39
        dog_breed.iloc[c, 5] = 56.97

    dog_intellience = dog_intellience.fillna("40%")
    print("\nTask 2:")
    print(dog_breed.iloc[18], "\n")
    print(dog_breed.iloc[111], "\n")
    print(dog_intellience.tail())
    dog_breed.to_csv('AKC Breed Info_filed.csv')
    dog_intellience.to_csv('dog_intelligence_filed.csv')


# 3. read about the ”join” command for Pandas frames. Construct a new frame by ”join”
# and write the results into a file ”dog info combined.csv”
def Task3():
    dog_breed = readFile('AKC Breed Info_filed.csv')
    dog_intellience = readFile('dog_intelligence_filed.csv')
    pd.set_option('display.width', 100)
    dog_intellience = dog_intellience.drop(['index', 'Unnamed: 0'], axis=1)
    joinFile = dog_breed.join(dog_intellience.set_index('Breed'), on='Breed')
    joinFile = joinFile.drop('Unnamed: 0', axis=1)

    joinFile.to_csv('dog info combined.csv')

    filledFile = joinFile.copy()

    filledFile['obey'] = filledFile['obey'].fillna(value='40%')
    filledFile['reps_lower'] = filledFile['reps_lower'].fillna(value=26)
    filledFile['reps_upper'] = filledFile['reps_upper'].fillna(value=40)
    filledFile['Classification'] = filledFile['Classification'].fillna(value='Normal Dogs')

    filledFile.to_csv('dog info combined_filled.csv')
    print("\nTask 3:")
    print("Finished writing file.")
    print("Data Sample (origin):")
    print(joinFile.head())
    print("Data Sample (filled):")
    print(filledFile.head())
    print("\n", "*" * 50)


# How many different classification values are there?
# Which classification contains most breeds and the least number of breeds
def Task4():
    dogFile = readFile('dog info combined_filled.csv')
    print("Task 4:")
    print("There total seven different classifications.")
    print(dogFile['Classification'].value_counts())
    print("\n", "*" * 50)


# 5. for each breed, compute its average height and average weight.
# Which 5 breeds are the tallest, the shortest, the heaviest, and the lightest?
def Task5():
    dogFile = readFile('dog info combined_filled.csv')
    dogFile = dogFile.drop('Unnamed: 0', axis=1)
    pd.set_option('display.max_columns', 5)

    dogFile['average height'] = (dogFile['height_low_inches'] + dogFile['height_high_inches']) / 2
    dogFile['average weight'] = (dogFile['weight_low_lbs'] + dogFile['weight_high_lbs']) / 2

    print("Task 5:")
    print("Top 5 Tallest Dogs:")
    tall = dogFile.sort_values(by='average height', ascending=False)
    print(tall.head())
    print("\nTop 5 Shortest Dogs:")
    short = dogFile.sort_values(by='average height', ascending=True)
    print(short.head())
    print("\nTop 5 Heaviest Dogs:")
    heavy = dogFile.sort_values(by='average weight', ascending=False)
    print(heavy.head())
    print("\nTop 5 Lightest Dogs:")
    light = dogFile.sort_values(by='average weight', ascending=True)
    print(light.head())
    print("\n", "*" * 50)

    dogFile.to_csv('dog info combined_filled.csv')


# 6. Compute the variability of height and weight for each breed
#   and write this as additional columns in your combined data frame
# 7. compute the variability of repetitions for each breed and write this as an additional column
# 8. which 5 breeds have the highest and lowest ranking by vari- ability for height, weight and repetitions (separately).
#   Are any breeds that are in more than one group?
def Task6_7_8():
    dogFile = readFile('dog info combined_filled.csv')
    dogFile = dogFile.drop('Unnamed: 0', axis=1)
    pd.set_option('display.max_columns', 6)

    # Task 6
    dogFile['height variability'] = (dogFile['height_high_inches'] - dogFile['height_low_inches']) / (dogFile['height_high_inches'] + dogFile['height_low_inches'])
    dogFile['weight variability'] = (dogFile['weight_high_lbs'] - dogFile['weight_low_lbs']) / (dogFile['weight_high_lbs'] + dogFile['weight_low_lbs'])

    # Taks 7
    dogFile['repetitions variability'] = (dogFile['reps_upper'] - dogFile['reps_lower']) / (dogFile['reps_upper'] + dogFile['reps_lower'])

    # Task 8
    print("Task 8:")
    print("Variability for height - highest")
    v_height_a = dogFile.sort_values(by='height variability', ascending=False)
    print(v_height_a.head())
    print("Variability for height - lowest")
    v_height_d = dogFile.sort_values(by='height variability', ascending=True)
    print(v_height_d.head())
    print("\nVariability for weight - highest")
    v_weight_a = dogFile.sort_values(by='weight variability', ascending=False)
    print(v_weight_a.head())
    print("Variability for weight - lowest")
    v_weight_d = dogFile.sort_values(by='weight variability', ascending=True)
    print(v_weight_d.head())
    print("\nVariability for reps - highest")
    v_reps_a = dogFile.sort_values(by='repetitions variability', ascending=False)
    print(v_reps_a.head())
    print("Variability for reps - lowest")
    v_reps_d = dogFile.sort_values(by='repetitions variability', ascending=True)
    print(v_reps_d.head())

    print("\nSome breeds shows up multiple times: "
          "\nPoodle Standard, Yorkshire Terrier"
          "\nBorder Collie, and Chihuahua")
    print("\n", "*" * 50)

    dogFile.to_csv('dog info combined_filled.csv')


# 9. compute the average μ and standard deviation σ for obey probabilities by classification group.
def Task9():
    dogFile = readFile('dog info combined_filled.csv')
    df = dogFile.drop('Unnamed: 0', axis=1)
    X = [0]*8

    X[0] = filterData(df, 'Brightest')
    X[1] = filterData(df, 'Excellent Working')
    X[2] = filterData(df, 'Above Average')
    X[3] = filterData(df, 'Average Working/')
    X[4] = filterData(df, 'Fair Working/')
    X[5] = filterData(df, 'Normal')
    X[6] = filterData(df, 'Lowest')
    X[7] = filterData(df, '')

    classification = pd.DataFrame(columns=['Classification', 'μ(obey)', 'σ(obey)', 'μ(r l)', 'σ(r l)', 'μ(r u)', 'σ(r u)'])
    name = ['Brightest', 'Excellent Working', 'Above Average', 'Average Working', 'Fair Working', 'Normal', 'Lowest', 'All Breed']
    for i in range(8):
        avg_obey = X[i]['obey'].mean()
        std_obey = X[i]['obey'].std()
        avg_resl = X[i]['reps_lower'].mean()
        std_resl = X[i]['reps_lower'].std()
        avg_resu = X[i]['reps_upper'].mean()
        std_resu = X[i]['reps_upper'].std()
        classification.loc[len(classification.index)] = [name[i], avg_obey, std_obey, avg_resl, std_resl, avg_resu, std_resu]

    print("Task 8:")
    print(classification)
    print("Discuss: The σ(r_l) adn σ(r_u) are almost always zero, because the lack of data variation.")
    print("\n", "*" * 50)

    df.to_csv('dog info combined_filled.csv')


def filterData(df, key):
    res = df.loc[df['Classification'].str.contains(key)]
    res['obey'] = res['obey'].str.rstrip('%')
    convert_dict = {'obey': float}
    res = res.astype(convert_dict)
    res['obey'] = res['obey'] / 100
    return res


# 10. compute the average number of repetations
# 11. which 5 breeds require most repetitions to learn a new command
def Task10_11():
    dogFile = readFile('dog info combined_filled.csv')
    df = dogFile.drop('Unnamed: 0', axis=1)
    pd.set_option('display.max_columns', 4)

    # Task 10:
    df['average repetations'] = (df['reps_lower'] + df['reps_upper']) / 2

    # Task 11:
    print('Task 11:')
    print(df.head())
    print("\n", "*" * 50)
    df.to_csv('dog_analysis.csv')


def Task12_14():
    dogFile = readFile('dog info combined_filled.csv')
    df = dogFile.drop('Unnamed: 0', axis=1)
    # 12. if you measure intelligence by the obey probability,
    #     does it depend on size (height or weight). Suggest very simple ways to address this?
    print("Task 12:")
    print("We can list some intelligent dogs as samples and check their average height and weight. "
          "\n(See following table)")
    df = filterData(df, '')
    df = df.sort_values(by='obey', ascending=False)
    cluster = df[['Breed', 'obey', 'average height', 'average weight']]
    print(cluster.head(10))
    print("\nFrom above we can see, that the range of height and weight is quite wide based on different breeds."
          "\nThis means we cannot determinate that the obey probability is depend on the size,"
          "\nat least not based on the sample we have.")
    print("\n", "*" * 50)

    #14. consider a ”linear” density: divide the average weight by average height.
    #    Do you think that it relates to intelligence?
    print('Task 14:')
    cluster['linear'] = cluster['average weight'] / cluster['average height']
    print(cluster.head(10))
    print("\n From above we can draw some conclusion: most of the linear density is around 2 to 4."
          "\n I think it might has some relation with the intelligence, but I will need more samples"
          "\n to determinate whether it's true or just a coincidence. ")

    df['linear'] = cluster['linear']
    df.to_csv('dog_analysis.csv')


def Driver():
    Task1()
    Task2()
    Task3()
    Task4()
    Task5()
    Task6_7_8()
    Task9()
    Task10_11()
    Task12_14()


Driver()
## README

*By: Zuowen Tang - CS622 Assignment 3*

### <u>To run the program:</u> 

Load the src folder in your IDE and run main.java.

<u>Please ensure that the file.txt is in the same folder as the java files.</u> 



---

This program has two functions: (1)**Search a given keyword in the target file**, (2)**Display the search history**.

By entering numbers, you should be able to choose which function you wish to use. The program will throw out an error message if the input is not valid.



In the **searching function**, the user can:

1. Input a keyword, which can be words or numbers. The program will start to search for this keyword in the target file. If the keyword is found, it will output the results of the title that contains the given keyword and other related information, such as funds raised percent, close date, and the tagline. If not, the program will output a message telling the user "target not found."
2. Customize how many results they want to see. The default number is one.
3. If the given keyword does not exit the file, the program will output an error message.



The search history will automatically be stored in the memory; the user can check the memory using the **history display function**. The program will output a table that contains the following information: 

1. Keywords that are being searched. 
2. How many results have shown. 
3. How many times the user had searched the target keyword. 
4. The latest timestamp when the user searches the target keyword. 
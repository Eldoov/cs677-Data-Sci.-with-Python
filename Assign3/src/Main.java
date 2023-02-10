/**
 * Name: Zuowen Tang
 * Date: 2/9/2023
 * Course: CS-622 Spring 2023
 * Assignment 3
 * Description: This file contains a keyword searching engine for a certain file.
 *              The program will ask the user to enter a keyword and searching for it in the file.
 *              The result shows the title that contains the given keyword and information of related
 *              funds raised percent, close date, and tag line.
 *              update: This program now can also show and store the searching history.
 */

import java.sql.Timestamp;
public class Main {
    public static void main(String[] args) throws Exception {
        UserInterface.Start();// start the program
    }
    static void test(){ // A quick test method to store and show the history
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());

        // use History.Log() method to store the searching history
        History.Log("Hello", 10, timestamp);
        History.Log("you", 5, timestamp);
        History.Log("Albert", 100, timestamp);
        History.Log("NoWay", 1, timestamp);
        History.Log("Hello", 32, timestamp);
        History.Log("fine", 55, timestamp);
        History.Log("Robot", 9, timestamp);

        // use History.Display() method to display the history
        History.Display();
    }
}

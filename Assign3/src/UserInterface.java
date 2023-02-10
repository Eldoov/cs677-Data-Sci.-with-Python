/**
 * Name: Zuowen Tang
 * Date: 2/9/2023
 * Course: CS-622 Spring 2023
 * Assignment 3
 */

import java.util.Scanner;
import java.util.InputMismatchException;
import java.sql.Timestamp;

public class UserInterface {
    public static void Start() throws Exception {

        // Start the UI that asks user to input orders
        while (true){
            System.out.println("******************************************************************************");
            System.out.println("* Welcome to Company Searching Program. Please choose a function from below. *");
            System.out.println("******************************************************************************");
            System.out.println("1. Search for a Company.    2. Show Searching History.     0. Quit the Program.");

            int choice;
            while (true){ // ask user to input choice
                try {
                    Scanner inputChoice = new Scanner(System.in);
                    choice = inputChoice.nextInt();
                    break;
                }catch (InputMismatchException e){ // If user enters a non-integer, ask them to do it again.
                    System.out.println("Please only enters numbers.");
                }
            }

            switch (choice) {
                case 1 -> SearchCompany(); // call searching UI and ask user for further information
                case 2 -> History.Display(); // call history display function
                case 0 -> System.exit(0);
                default -> System.out.println("Please enter a valid integer."); // if user input other numbers
            }

            System.out.println();
        }

    }

    public static void SearchCompany() throws Exception {
        System.out.println("Enter the keyword you want to search:");
        Scanner inputWord = new Scanner(System.in);
        String keyWord = inputWord.nextLine();
        int resultNum = 1;

        // capture current time for timestamp
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());

        // Ask user to enter an integer for how many results they want to see.
        while (true){
            try {
                Scanner inputInt = new Scanner(System.in);
                System.out.println("How many results you want to shown?");
                resultNum = inputInt.nextInt();
                break;
            }catch (InputMismatchException e){ // If user enters a non-integer, ask them to do it again.
                System.out.println("Please only enters numbers.");
            }
        }

        // call searching function
        SearchEngine.Search(keyWord, resultNum, timestamp);
    }

}

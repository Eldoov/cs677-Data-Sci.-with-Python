/**
 * Name: Zuowen Tang
 * Date: 2/9/2023
 * Course: CS-622 Spring 2023
 * Assignment 3
 */

import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.sql.Timestamp;

public class SearchEngine {
    static Scanner Read(){ // A method to read target file
        String dir = System.getProperty("user.dir"); // Get current folder's path
        Scanner file = null;

        try{
            File f = new File(dir+"/src/file.txt"); // Read the given file.
            file = new Scanner(f);
        }catch (IOException e){
            System.out.println("File not found. Exiting...");
            System.exit(0);
        }

        return file;
    }

    static void Search(String keyword, int resultNum, Timestamp timestamp){
        Scanner file = Read();
        int totalSearch = 0; // Initiate the total number of how many times have searched.
        boolean isFound = false; // Initiate the boolean value to see if the given keyword is found in file.

        // Start searching the given keyword.
        while (file.hasNextLine()){
            final String lineFromFile = file.nextLine();

            // The program only searches the title column for given keyword.
            Matcher matcher = Pattern.compile("\"title\":\"(.*?)"+ keyword +"(.*?)\"", Pattern.CASE_INSENSITIVE).matcher(lineFromFile);

            if(matcher.find()) { // A match has found!
                isFound = true;
                resultNum--; // Keep tracking how many results user wants.
                totalSearch++;

                // Prepare to output the result, start from the title:
                String title = matcher.group(1) + keyword + matcher.group(2);
                System.out.println("Result found:\s\s\s\s\s\s\s\s\s\s\s\s" + title);

                // Searching for the close_date, the fund_raised_percent, and the tagline in the same line.
                Matcher closeDate = Pattern.compile("\"close_date\":\"(.*?)\"", Pattern.CASE_INSENSITIVE).matcher(lineFromFile);
                Matcher funds = Pattern.compile("\"funds_raised_percent\":(.*?),", Pattern.CASE_INSENSITIVE).matcher(lineFromFile);
                Matcher tagLine = Pattern.compile("\"tagline\":\"(.*?)\",", Pattern.CASE_INSENSITIVE).matcher(lineFromFile);
                funds.find();
                closeDate.find();
                tagLine.find();

                // In case the related data is missing, initiate exceptions.
                // Add spaces so the results look better.
                try{
                    System.out.println("Funds Raised Percent:\s\s\s\s"+funds.group(1));

                }catch (IllegalStateException e){
                    System.out.println("Funds Raised Percent:\s\s\s\sData Missing.");
                }

                try{
                    System.out.println("Close Date:\s\s\s\s\s\s\s\s\s\s\s\s\s\s"+closeDate.group(1));
                }catch (IllegalStateException e){
                    System.out.println("Close Date:\s\s\s\s\s\s\s\s\s\s\s\s\s\sData Missing.");
                }

                try{
                    System.out.println("Tag Line:\s\s\s\s\s\s\s\s\s\s\s\s\s\s\s\s"+tagLine.group(1));
                }catch (IllegalStateException e){
                    System.out.println("Tag Line:\s\s\s\s\s\s\s\s\s\s\s\s\s\s\s\sData Missing.");
                }

                System.out.println();

            }

            // If there has no match for the given keyword at all, output following message.
            if (!matcher.find() && !file.hasNextLine() && !isFound){
                System.out.println("No such match. Please enter a different keyword.");
            }

            // If the program outputs enough results user wants, break the loop.
            if (resultNum == 0){
                break;
            }
        }

        System.out.println("Total " +totalSearch+ " results have shown.");

        // Store searching history to the memory
        History.Log(keyword, totalSearch, timestamp);

    }
}

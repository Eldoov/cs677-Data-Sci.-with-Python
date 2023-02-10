/**
 * Name: Zuowen Tang
 * Date: 2/9/2023
 * Course: CS-622 Spring 2023
 * Assignment 3
 */

import java.sql.Timestamp;
import java.util.LinkedList;

public class History {
    // initiate four lists for storage
    static LinkedList<String> keywordList = new LinkedList<>();
    static LinkedList<Integer> searchList = new LinkedList<>();
    static LinkedList<Timestamp> timeList = new LinkedList<>();
    static LinkedList<Integer> countList = new LinkedList<>();

    static void Display(){ // a function to show all the searching history

        if (keywordList.isEmpty()){ // check if the history is empty
            System.out.println("You do not have any searching history yet!");
        }else{
            System.out.println(); // output the history table
            System.out.printf("--------------------------------------------------------------------------------------%n");
            System.out.printf("                               Your Searching History                                %n");
            System.out.printf("--------------------------------------------------------------------------------------%n");
            System.out.printf("| %-16s | %-15s | %-18s | %-24s |%n", "Keyword", "Results Shown", "Search Frequency", "Newest Timestamp");
            System.out.printf("--------------------------------------------------------------------------------------%n");
            for(int i = 0; i<keywordList.size(); i++){
                System.out.printf("| %-16s | %-15s | %-18s | %-24s |%n", keywordList.get(i), searchList.get(i), countList.get(i), timeList.get(i));
            }
            System.out.printf("--------------------------------------------------------------------------------------%n");
        }

    }

    static void Log(String keyword, int totalSearch, Timestamp timeStamp){

        if (keywordList.contains(keyword)){ // check if there's duplicates
            int indexNum = keywordList.indexOf(keyword); // if so, get the index number and get ready to update information
            int count = countList.get(indexNum);
            count++; // add searching frequency

            countList.set(indexNum, count); // update the searching frequency
            timeList.set(indexNum, timeStamp); // update the newest timestamp
            searchList.set(indexNum, totalSearch); // update all results shown
        }

        if (!keywordList.contains(keyword)){ // if target keyword is new to the list, add the related info to lists
            keywordList.add(keyword);
            searchList.add(totalSearch);
            timeList.add(timeStamp);
            countList.add(1);
        }
    }
}

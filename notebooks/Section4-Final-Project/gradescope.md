# How to use the autograder

Go to https://www.gradescope.com/ and enroll in CSE 255 (please sign in with your student email and use enrollment code 7GVWK2)

Go to HW 5 and submit your files. You will be re-routed to a page where the autograder results will appear, as soon as they finish processing.

## There are three files to submit

results.csv — your predictions on the random test set 

results_country.csv — your predictions on the country test set 

code.zip — your code in a zip file. If we have any doubt about your results, we will use your code to validate the results.

## Each csv file should have the following columns

filename — e.g. image13724.npz

urban — 1 when urban, 0 when not urban

pred_with_abstention  — predictions of -1, 1, and 0 when I don’t know

pred_wo_abstention - predictions of -1, 1 

## Evaluation 
You will be evaluated on different test cases, that will also appear on a class leaderboard. You can choose an anonymous name for the leaderboard. Please note that the asymmetric loss can be a value between -2 and 1, and appears in the leaderboard that way.

## Number of submissions per day
If you made too many submissions within the past 24 hours, your next submission will have 0 score. However, you don’t loose your best score — just go to your dashboard and you will find a button called submission history on the bottom. Select your the submission you want to have counted. This will also make this submission appear to the leaderboard.

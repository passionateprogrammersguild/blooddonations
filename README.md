# Warm Up: Predict Blood Donations
This project is from the [Warm Up: Predict Blood Donations](https://www.drivendata.org/competitions/2/warm-up-predict-blood-donations/page/5/) on the drivendata.org web site.

## Goal
The goal is to predict whether the person donated blood in March 2007

## Features
* Months since Last Donation: this is the number of monthis since this donor's most recent donation.
* Number of Donations: this is the total number of donations that the donor has made.
* Total Volume Donated: this is the total amound of blood that the donor has donated in cubuc centimeters.
* Months since First Donation: this is the number of months since the donor's first donation.

## Submission format

This competitions uses log loss as its evaluation metric, so the predictions you submit are the probability that a donor made a donation in March 2007.

The submission format is a csv with the following columns:

|               | Made Donation in March 2007|
| ------------- |:-------------------------- |
| 659           | 0.5                        |
| 276           | 0.5                        |
| 273           | 0.5                        |

To be explicit, you need to submit a file like the following with predictions for every ID in the Test Set we provide:

```
,Made Donation in March 2007
659,0.5
276,0.5
263,0.5
303,0.5
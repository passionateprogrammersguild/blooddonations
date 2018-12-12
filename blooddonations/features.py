import numpy as np
import math

class Features:
    
    @staticmethod
    def compute(df):
        def timesince(row):
            return ((row["Months since First Donation"])   / (row["Number of Donations"] * 1.0)) **2

        def timedifference(row):
            return (((row["Months since Last Donation"] * 1.0) - (row["TimeSince"] * 1.0)) **2)
        
        df["TimeSince"] = df.apply(timesince, axis=1)
        df["TimeSinceDifference"] = df.apply(timedifference, axis=1)
        
        tsdmedian = np.median(df["TimeSinceDifference"].values)
        tsdmin    = np.min(df["TimeSinceDifference"].values)
        tsdmax    = np.max(df["TimeSinceDifference"].values)

        df["TimeSinceMedianDifference"] =  df["TimeSinceDifference"].apply(lambda x: x - tsdmedian)        
        df["TimeSinceMinDifference"]    =  df["TimeSinceDifference"].apply(lambda x: x - tsdmin)
        df["TimeSinceMaxDifference"]    =  df["TimeSinceDifference"].apply(lambda x: x - tsdmax)

        return df

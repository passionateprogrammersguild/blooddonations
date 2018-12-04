class Features:
    
    @staticmethod
    def compute(df):
        def lastdonationrange(val):
            if val > 0 and val < 4:
                return "1-3Months"
            
            if val > 3 and val < 7:
                return "4-6Months"
            
            if val > 6 and val < 10:
                return "7-9Months"

            if val > 9 and val < 13:
                return "10-12Months"
            
            return ">12Months"

        df["LastDonationRange"] = df["Months since Last Donation"].apply(lastdonationrange)

        return df

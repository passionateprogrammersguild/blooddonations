import sys, os
import pandas as pd

class Transform:
    
    def __init__(self, datadir):
        self.datadir = datadir

    def dumpstats(self):
        df = pd.read_csv(os.path.join(datadir, 'train.csv'))

        print(df.columns.values)

        for col in df.columns.values:
            print("")
            print("############################################# START #############################################")
            print(df[col].describe())
            print("############################################# END #############################################") 
            print("")

    def transform(self):
        df = pd.read_csv(os.path.join(datadir, 'blooddonations.csv'))

        #TODO start to compute features



if __name__ == '__main__':
    datadir = sys.argv[1]

    t = Transform(datadir)
    t.dumpstats()
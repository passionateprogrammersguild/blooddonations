import sys, os, time, random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np

from features import Features

#initialize variables
maxtrainingtime = 1200 #20 * 60 = 20 minutes
maxnumberofiterationloss = 100
roundlevel = 4
metrics = []

# reproducable script
random.seed(21)
np.random.seed(21)

datadir = sys.argv[1]

classificationcols = [
    #"LastDonationRange"
]

floatcols = [
    'Months since Last Donation',
    'Number of Donations',
    'Total Volume Donated (c.c.)',
    'Months since First Donation',
    "TimeSince",
    "TimeSinceDifference",
    "TimeSinceMedianDifference",
    "TimeSinceMinDifference",
    "TimeSinceMaxDifference"
]

metacols = [
]

ycol = 'Made Donation in March 2007'

df_train = Features.compute(pd.read_csv(os.path.join(datadir, 'train.csv')))
df_test  = Features.compute(pd.read_csv(os.path.join(datadir, 'test.csv')))

print("df_train.shape", df_train.shape, "df_test.shape", df_test.shape)
print("columns", ','.join(df_train.columns.values))
traincols = floatcols + classificationcols

X_tr = df_train.loc[:, traincols]
Y_tr = df_train.loc[:, ycol]

X_xv = df_test.loc[:, traincols]

#label encode the classification columns
labelencoders = {}
for colname in classificationcols:
    
    le = LabelEncoder()
    vals = df_train[colname].unique()
    le.fit(vals)
    
    print("col:",colname,"values:",','.join(map(str,vals)))
    X_tr[colname]  = le.transform(X_tr[colname])
    X_xv[colname]  = le.transform(X_xv[colname])
    
    labelencoders[colname]     = le


rfc = RandomForestClassifier(
    max_depth=8,
    criterion="gini",
    min_samples_leaf=8,
    #n_estimators=50,
    #n_jobs=-1
    n_estimators=0,
    warm_start=True
)

#fit the model to the data ie train the model with a simple learning loop
n_iters = 0
starttime = time.time()
#execute the training loop
while (True):
    allworse = False
    n_iters +=1

    rfc.n_estimators = n_iters
    
    rfc.fit(X_tr, Y_tr)

    #compute metric
    tr_yhat_proba = rfc.predict_proba(X_tr)
    tr_logloss = round(log_loss(Y_tr, tr_yhat_proba), roundlevel)
    
    print("iter {0} TRAIN LOG LOSS {1} PREV LOG LOSS {2}".format(n_iters, tr_logloss, (metrics[-1][1] - 0.0) if len(metrics) > 1 else 0.0))

    #TODO: have we gotton worse in the last n iterations
    if len(metrics) > maxnumberofiterationloss and tr_logloss >= metrics[-1][1]:        
        # we have gotton worse compared to the previous iteration so lets see how bad we have gotton over time        
        lastN = maxnumberofiterationloss * -1
        allworse = all(tr_logloss >= x[1] for x in metrics[lastN:])
        if allworse:
            print("breaking out since last", maxnumberofiterationloss,"terations are worse or remained the same")

    metrics.append((n_iters, tr_logloss))

    exceededmaximumtimelimit = (time.time() - starttime) > maxtrainingtime
    if allworse or exceededmaximumtimelimit:
        print("exiting out of the learning loop due to early stopping")
        break

#get the best iteration of the model by sorting on the xv_rmse and n_iters    
sorted_metrics = sorted(metrics, key=lambda x: (x[1], x[0]))
best_iter, best_logloss = sorted_metrics[0][0], sorted_metrics[0][1]
rfc.estimators_ = rfc.estimators_[:best_iter]

tr_yhat_proba = rfc.predict_proba(X_tr)
tr_logloss = round(log_loss(Y_tr, tr_yhat_proba), roundlevel)
    
assert(tr_logloss == best_logloss)
print("BEST ITER {0} LOG LOSS {1}".format(best_iter, best_logloss))

#print out the feature importance
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(traincols)):
    print("%d. feature %s (%f)" % (f + 1, traincols[f], importances[indices[f]]))
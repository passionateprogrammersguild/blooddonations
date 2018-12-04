import sys, os, time, random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pandas as pd
import numpy as np

from features import Features


# reproducable script
random.seed(1)
np.random.seed(1)

datadir = sys.argv[1]

classificationcols = [
    #"LastDonationRange"
]

floatcols = [
    'Months since Last Donation',
    'Number of Donations',
    'Total Volume Donated (c.c.)',
    'Months since First Donation'
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
    max_depth=None,
    criterion="gini",
    min_samples_leaf=30,
    n_estimators=50,
    n_jobs=-1
)

#fit the model to the data ie train the model with a simple learning loop
rfc.fit(X_tr, Y_tr)

#eval on the training set
start = time.time()
xv_yhat_proba = rfc.predict_proba(X_xv)
tr_yhat_proba = rfc.predict_proba(X_tr)
end = time.time()
print("elapsed time of predict in seconds", end - start)

#compute the auc over the prediction
fpr, tpr, thresholds = metrics.roc_curve(Y_tr, tr_yhat_proba[:,1], pos_label=1)
tr_auc = metrics.auc(fpr, tpr)
print("TRAIN AUC {0}".format(tr_auc))

#print out the feature importance
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(traincols)):
    print("%d. feature %s (%f)" % (f + 1, traincols[f], importances[indices[f]]))
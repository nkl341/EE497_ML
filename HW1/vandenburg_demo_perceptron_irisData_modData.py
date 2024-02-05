# program		vandenburg_demo_perceptron_irisData_v1.py
# purpose	    Simple imputing of missing data, and simple perceptron
# usage         script
# notes         (1) Imputes only values encoded as NaN
#               (2) Imputes with median method
#               (3) Encoded lables seem to change the accuracy of the perceptron to a lower value than unencoded labels.
#               (4) The perceptron seems to be more accurate with standardized data than non-standardized data.
#               (5) Major graph changes
#               (") Two graphs instead of four, scatter length vs width, with polyfit line.
#               (6) Removed win32api as I work on a mac and it does not work on my computer.
# date			1/16/2024
# programmer    Colton Vandenburg

import datetime          # used for getting the date
import os                # used for getting the basic file name (returns lower case)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

import sklearn.impute as imp
import numpy as np
# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.basename(__file__)   # case insensitive file name

ix = str.find(programName_c,'.')

fileName_c = 'irisData_modified.csv'
programMsg_c = programName_c + ' (' + date_c + ')'
authorName_c = 'Colton Vandenburg'

figName_c = programName_c[:ix]+'_fig.png'



# ==== get data frame, split, preprocess train ==============
irisData_df = pd.read_csv((fileName_c))

irisDataT_df, irisDataV_df = train_test_split(irisData_df,test_size = 0.3, random_state=24) # Variations to the random state seem to change the unscaled data accuracy, but the scaled data accuracy as much.
irisDataT_num_df = irisDataT_df.drop("type", axis = 1)    # numerical data only
irisDataT_labels = irisDataT_df["type"]
irisDataT_labels_en, irisDataT_labelNames = irisDataT_labels.factorize()

# which is better - standardize or normalize [0,1]?
sc = StandardScaler()       # this method standardizes to 0-mean, sigma=1
#sc = MinMaxScaler()       # this method normalizes [0,1]
sc.fit(irisDataT_num_df)    # do the fit step on the training data

irisDataTs_num_df = sc.transform(irisDataT_num_df)  # apply transform on train data


# ================= preprocess verify ==================
irisDataV_num_df = irisDataV_df.drop("type", axis = 1)    # numerical data only
irisDataV_labels = irisDataV_df["type"]
irisDataV_labels_en, irisDataV_labelNames = irisDataV_labels.factorize()

irisDataVs_num_df = sc.transform(irisDataV_num_df)  # do same transform on verify data


#========================Impute missing data=========================
irisDataT_num_df = irisDataT_num_df.fillna(irisDataT_num_df.median()) # find nan values and impute with median in train data
irisDataV_num_df = irisDataV_num_df.fillna(irisDataV_num_df.median())  # find nan values and impute with median in verify data
# Scaled Data as well
irisDataTs_num_df = sc.transform(irisDataT_num_df)
irisDataVs_num_df = sc.transform(irisDataV_num_df)

# ================ perceptron, non-scaled ===================
ppn_T = Perceptron(eta0=0.05, random_state=1)    # what is best value for eta?
ppn_T.fit(irisDataT_num_df, irisDataT_labels)

predict_T = ppn_T.predict(irisDataT_num_df)         # try on test data (should be better than verify)
accuracy_T = accuracy_score(irisDataT_labels, predict_T)

predict_V = ppn_T.predict(irisDataV_num_df)         # now do on verify data
accuracy_V = accuracy_score(irisDataV_labels, predict_V)

print('accuracy_T = ',accuracy_T)
print('accuracy_V = ',accuracy_V)

# ================ perceptron, standardized ===================
ppn_T.fit(irisDataTs_num_df, irisDataT_labels)

predict_Ts = ppn_T.predict(irisDataTs_num_df)         # try on test data (should be better than verify)
accuracy_Ts = accuracy_score(irisDataT_labels, predict_Ts)

predict_Vs = ppn_T.predict(irisDataVs_num_df)         # now do on verify data
accuracy_Vs = accuracy_score(irisDataV_labels, predict_Vs)

print('accuracy_Ts = ',accuracy_Ts)
print('accuracy_Vs = ',accuracy_Vs)

# ===================== PLOT RESULTS ===================
plt.figure(num=1, figsize=(11.2, 5.2))        # for the hist method, this does not work
plt.rcParams.update({'font.size': 8})           # 8-point fits a little better but still overlaps

featureNames = list(irisData_df.drop("type", axis=1).columns)

kp = 0
plt.subplot(121) 
plt.scatter(irisDataV_num_df[featureNames[kp]], irisDataV_num_df[featureNames[kp+1]])
plt.title('Scatter Plot: ' + featureNames[kp] + ' vs ' + featureNames[kp+1])
plt.xlabel(featureNames[kp])
plt.ylabel(featureNames[kp+1])
x = np.unique(irisDataV_num_df[featureNames[kp]])
y = np.poly1d(np.polyfit(irisDataV_num_df[featureNames[kp]], irisDataV_num_df[featureNames[kp+1]], 1))(x)
plt.plot(x, y, color='red')

kp += 2
plt.subplot(122) 
plt.scatter(irisDataV_num_df[featureNames[kp]], irisDataV_num_df[featureNames[kp+1]])
plt.title('Scatter Plot: ' + featureNames[kp] + ' vs ' + featureNames[kp+1])
plt.xlabel(featureNames[kp])
plt.ylabel(featureNames[kp+1])
x = np.unique(irisDataV_num_df[featureNames[kp]])
y = np.poly1d(np.polyfit(irisDataV_num_df[featureNames[kp]], irisDataV_num_df[featureNames[kp+1]], 1))(x)
plt.plot(x, y, color='red')

# Tried to plot the length of the petal vs the width of the petal, and added a line of best fit to see if the data would have any sort of correlation.
# ================= label plot edges ==================
plt.subplot(position=[0.0500,    0.94,    0.02500,    0.02500]) # U-left
plt.axis('off')
plt.text(0,.5, programMsg_c, fontsize=8)

plt.subplot(position=[0.550,    0.94,    0.02500,    0.02500]) # U-right
plt.axis('off')
plt.text(0,.5, authorName_c, fontsize=8)

plt.subplot(position=[0.0500,    0.02,    0.02500,    0.02500]) # L-left
plt.axis('off')
plt.text(0,.5, fileName_c, fontsize=8)

# plt.subplot(position=[0.5500,    0.02,    0.02500,    0.02500]) # L-right
# plt.axis('off')
# plt.text(0,.5, msg_plot_c, fontsize=8)


plt.savefig(figName_c)

plt.show()

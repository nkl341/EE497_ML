# program		vandenburg_demo_tsne_irisData_v1.py
# purpose	    Demonstrate perceptron with iris data
# usage         script
# notes         (1)
# date			2/14/2024
# programmer   Colton Vandenburg

import datetime          # used for getting the date
import os                # used for getting the basic file name (returns lower case)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing    # Import label encoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import seaborn as sns

# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.basename(__file__)                     # case insensitive file name


ix = str.find(programName_c,'.')

fileName_c = 'irisData_modified.csv'
programMsg_c = programName_c + ' (' + date_c + ')'
authorName_c = 'G.L. Fudge'

figName_c = programName_c[:ix]+'_fig.png'


# ========== get data frame, split, get numerical data ==============
irisData_df = pd.read_csv((fileName_c))
irisData_featureNames_cv = irisData_df.columns
irisData_featureNames_cv = irisData_featureNames_cv[:4]     # remove type column

tts_ts = 0.3
tts_rs = 24     #26 performs poorly - why?
irisDataT_df, irisDataV_df = train_test_split(irisData_df,test_size = tts_ts, random_state=tts_rs) 
irisDataT_num_df = irisDataT_df.drop("type", axis = 1)    # numerical data only

# ============== get labels, preprocess train data except for scaling  ======================
irisDataT_labels = irisDataT_df["type"]
label_encoder = preprocessing.LabelEncoder()  
irisDataT_labels_v = label_encoder.fit_transform(irisDataT_labels)

imputer = SimpleImputer(strategy = "median")
imputer.fit(irisDataT_num_df)
irisDataT_num_df = imputer.transform(irisDataT_num_df)

# ================= get labels, preprocess verify data ==================
irisDataV_num_df = irisDataV_df.drop("type", axis = 1)    # numerical data only
irisDataV_labels = irisDataV_df["type"]
irisDataV_labels_v = label_encoder.fit_transform(irisDataV_labels) 

irisDataV_num_df = imputer.transform(irisDataV_num_df)

# ================ perceptron, non-scaled ===================
ppn_eta = .01
ppn_rs = 1
ppn_T = Perceptron(eta0 = ppn_eta, random_state = ppn_rs)    # what is best value for eta?
ppn_T.fit(irisDataT_num_df, irisDataT_labels_v)

predict_T = ppn_T.predict(irisDataT_num_df)         # try on test data (should be better than verify)
accuracy_T = accuracy_score(irisDataT_labels_v, predict_T)
predict_V = ppn_T.predict(irisDataV_num_df)         # now do on verify data
accuracy_V = accuracy_score(irisDataV_labels_v, predict_V)

print('accuracy_T = ',accuracy_T)
print('accuracy_V = ',accuracy_V)

# ================ perceptron, standardized ===================
sc = StandardScaler()       # 0-mean, sigma=1; performs better than minMaxScaler(), which does [0,1]
sc.fit(irisDataT_num_df)    # do the fit step on the training data
irisDataTs_num_df = sc.transform(irisDataT_num_df)  # apply transform on train data
irisDataVs_num_df = sc.transform(irisDataV_num_df)  # do same transform on verify data

ppn_T.fit(irisDataTs_num_df, irisDataT_labels_v)

predict_Ts = ppn_T.predict(irisDataTs_num_df)         # try on test data (should be better than verify)
accuracy_Ts = accuracy_score(irisDataT_labels_v, predict_Ts)
predict_Vs = ppn_T.predict(irisDataVs_num_df)         # now do on verify data
accuracy_Vs = accuracy_score(irisDataV_labels_v, predict_Vs)

print('accuracy_Ts = ',accuracy_Ts)
print('accuracy_Vs = ',accuracy_Vs)

# ================= perceptron, normalized ==================
scn = MinMaxScaler()        # does not do as well as StandardScaler
scn.fit(irisDataT_num_df)    # do the fit step on the training data
irisDataTn_num_df = scn.transform(irisDataT_num_df)  # apply transform on train data
irisDataVn_num_df = scn.transform(irisDataV_num_df)  # do same transform on verify data

ppn_T.fit(irisDataTn_num_df, irisDataT_labels_v)

predict_Tn = ppn_T.predict(irisDataTn_num_df)         # try on test data (should be better than verify)
accuracy_Tn = accuracy_score(irisDataT_labels_v, predict_Tn)
predict_Vn = ppn_T.predict(irisDataVn_num_df)         # now do on verify data
accuracy_Vn = accuracy_score(irisDataV_labels_v, predict_Vn)

print('accuracy_Tn = ',accuracy_Tn)
print('accuracy_Vn = ',accuracy_Vn)

# ===================== prepare to plot results ===================
msg_tts_c = "test_train_split: tts_ts = " + "%1.2f" %(tts_ts) + "; tts_rs = " + str(tts_rs)
msg_ppn_c = "perceptron: ppn_eta = " + "%1.2f" %(ppn_eta) + "; ppn_rs = " + str(ppn_rs)
msg_plot_c = msg_tts_c + '; ' + msg_ppn_c

msg_acc1_c = "Accuracy: T = " + "%1.2f" %(accuracy_T) + "; V = " + "%1.2f" %(accuracy_V)
msg_acc2_c = "Ts = " + "%1.2f" %(accuracy_Ts) + "; Vs = " + "%1.2f" %(accuracy_Vs)
msg_acc3_c = "Tn = " + "%1.2f" %(accuracy_Tn) + "; Vs = " + "%1.2f" %(accuracy_Vn)

msg_acc_c = msg_acc1_c + '; ' + msg_acc2_c + '; ' + msg_acc3_c

# ======================== PLOT RESULTS ======================
#plt.figure(num=1, figsize=(11.2, 5.2))        # not consistent with boxplot
plt.rcParams.update({'font.size': 9})           # 8-point fits a little better but still overlaps
fig, axes = plt.subplots(2, 2, figsize=(11.2, 5.2))
fig.suptitle(msg_acc_c,fontsize=9)
sns.set(style='whitegrid')

sns.set(font_scale = 0.7)   # make fonts a little smaller on the box plot
for kp, feature in enumerate(irisData_featureNames_cv):
    sns.boxplot(x='type', y=feature, hue='type', data=irisData_df, ax=axes[kp//2, kp%2], palette='pastel')


# ============= Make the subplots look a little nicer ================= 
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.3)

# ================= label plot edges ==================
plt.subplot(position=[0.0500,    0.93,    0.02500,    0.02500]) # U-left
plt.axis('off')
plt.text(0,.5, programMsg_c, fontsize=8)

plt.subplot(position=[0.550,    0.93,    0.02500,    0.02500]) # U-right
plt.axis('off')
plt.text(0,.5, authorName_c, fontsize=8)

plt.subplot(position=[0.0500,    0.02,    0.02500,    0.02500]) # L-left
plt.axis('off')
plt.text(0,.5, fileName_c, fontsize=8)

plt.subplot(position=[0.3500,    0.02,    0.02500,    0.02500]) # L-right
plt.axis('off')
plt.text(0,.5, msg_plot_c, fontsize=8)


plt.savefig(figName_c)

plt.show()

# program		vandenburg_demo_decisionTree_irisData_v2.py
# purpose	    Demonstrate perceptron with iris data
# usage         script
# notes         (1) tts_rs = 24 raises accuracy from 0.8 to 0.9 compared to 26
#               (2) Scaled data does not improve accuracy in decision trees, but does in perceptron?
#               (3) Pruned decision tree is more accurate than unpruned
# date			1/25/2024
# programmer    Colton Vandenburg

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.basename(__file__)                                          # case insensitive file name


ix = str.find(programName_c,'.')

fileName_c = '/Volumes/ESD-USB/Spring 2024/EE 497 ML/HW2/irisData_modified.csv'
programMsg_c = programName_c + ' (' + date_c + ')'
authorName_c = 'Colton Vandenburg'

figName_c = programName_c[:ix]+'_fig.png'


# ========== get data frame, split, get numerical data ==============
irisData_df = pd.read_csv((fileName_c))
irisData_featureNames_cv = irisData_df.columns
irisData_featureNames_cv = irisData_featureNames_cv[:4]     # remove type column

tts_ts = 0.3
tts_rs = 24   #26 performs poorly - why?
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

irisData_featureNames_cv = list(irisData_featureNames_cv)
# ================== Decision Tree, Pruned =====================
dt_pruned = DecisionTreeClassifier(max_depth=3, min_samples_leaf=9)
dt_pruned.fit(irisDataT_num_df, irisDataT_labels_v)

# Calculate accuracy score for testing data
predict_T = dt_pruned.predict(irisDataT_num_df)         # try on test data (should be better than verify)
accuracy_T = accuracy_score(irisDataT_labels_v, predict_T)
predict_V = dt_pruned.predict(irisDataV_num_df)         # now do on verify data
accuracy_V = accuracy_score(irisDataV_labels_v, predict_V)

print('accuracy_T = ',accuracy_T)
print('accuracy_V = ',accuracy_V)

# Plot pruned decision tree
plt.figure(figsize=(12, 10))  # Increase the figure size to 12x10
plot_tree(dt_pruned, feature_names=None , class_names=None, filled=True)
plt.title("Pruned Decision Tree")
plt.text(0.05, 0.05, fileName_c, transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='left')
plt.text(0.05, 0.95, programName_c, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
plt.text(0.95, 0.95, authorName_c, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)  # Adjust the plot space
plt.show()


# ================== Decision Tree, Unpruned ===================
dt_unpruned = DecisionTreeClassifier()
dt_unpruned.fit(irisDataT_num_df, irisDataT_labels_v)

# Calculate accuracy score for testing data
redict_T = dt_unpruned.predict(irisDataT_num_df)         # try on test data (should be better than verify)
accuracy_T = accuracy_score(irisDataT_labels_v, predict_T)
predict_V = dt_unpruned.predict(irisDataV_num_df)         # now do on verify data
accuracy_V = accuracy_score(irisDataV_labels_v, predict_V)

print('accuracy_T = ',accuracy_T)
print('accuracy_V = ',accuracy_V)

# Plot unpruned decision tree
plt.figure(figsize=(12, 10))  # Increase the figure size to 12x10
plot_tree(dt_unpruned, feature_names=None , class_names=None, filled=True)
plt.title("Unpruned Decision Tree")
plt.text(0.05, 0.05, fileName_c, transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='left')
plt.text(0.05, 0.95, programName_c, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
plt.text(0.95, 0.95, authorName_c, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)  # Adjust the plot space
plt.show()

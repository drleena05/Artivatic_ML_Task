# -*- coding: utf-8 -*-
"""
Created on Fri May 14 18:49:28 2021

@author: drlee
"""

#Importing the necessary libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Loading the train dataset
df = pd.read_csv("C:/Users/drlee/Downloads/ML_Artivatic_dataset/ML_Artivatic_dataset/train_indessa.csv")

#Checking shape of the dataset(no. of rows an col)
df.shape

#Dropping columns not related to our target variable
df.drop(['loan_amnt','funded_amnt','batch_enrolled','sub_grade','emp_title','desc','title','zip_code','addr_state','mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog','verification_status_joint'], axis=1, inplace=True)
print(df)

##Data transformation , Converting categorical values to numerical, stripping off texts
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
#Label encoding in columns 2,4,9,19,25
df.iloc[:, 2] = encoder.fit_transform(df.iloc[:, 2])
df.iloc[:, 4] = encoder.fit_transform(df.iloc[:, 4])
df.iloc[:, 9] = encoder.fit_transform(df.iloc[:, 9])
df.iloc[:, 19] = encoder.fit_transform(df.iloc[:, 19])
df.iloc[:, 25] = encoder.fit_transform(df.iloc[:, 25])


purpose_summary = df.groupby('purpose').agg({'member_id':'count','loan_status':'sum'})
purpose_summary['defaultrate'] = purpose_summary['loan_status']/purpose_summary['member_id']

#Grouping 'Purpose' column into P1, P2, P3 based on default rate and assigning the groups accordingly
#P1= >70%  , P2= >30% <70% , P3= <30%
df['purpose'] = df['purpose'].str.replace('educational','P1')
df['purpose'] = df['purpose'].str.replace('wedding','P1')
df['purpose'] = df['purpose'].str.replace('car','P2')
df['purpose'] = df['purpose'].str.replace('moving','P2')
df['purpose'] = df['purpose'].str.replace('major_purchase','P2')
df['purpose'] = df['purpose'].str.replace('renewable_energy','P2')
df['purpose'] = df['purpose'].str.replace('small_business','P2')
df['purpose'] = df['purpose'].str.replace('house','P2')
df['purpose'] = df['purpose'].str.replace('debt_consolidation','P3')
df['purpose'] = df['purpose'].str.replace('home_improvement','P3')
df['purpose'] = df['purpose'].str.replace('credit_card','P3')
df['purpose'] = df['purpose'].str.replace('other','P3')
df['purpose'] = df['purpose'].str.replace('vacation','P3')
df['purpose'] = df['purpose'].str.replace('medical','P3')
df['purpose'] = df['purpose'].str.replace('credit_P2d','P3')
#Check for replaced values
df['purpose'].unique()

df['last_week_pay'].unique()
#Stripping of texts in "last week pay" column
df['last_week_pay'] = df['last_week_pay'].str.replace('th week','')
df['last_week_pay'] = df['last_week_pay'].str.replace('NA','')

df['emp_length'].unique()
#Replacing years with blank space;<1 year with 0; 1year with 1 and + with blank space
df['emp_length'] = df['emp_length'].str.replace(' years','')
df['emp_length'] = df['emp_length'].str.replace('< 1 year','0')
df['emp_length'] = df['emp_length'].str.replace('1 year','1')
df['emp_length'] = df['emp_length'].str.replace('+','')
#Checking if above has been replaced
df['emp_length'].unique()

##Making Dummy variables for 'home_ownership','verifying_status','purpose'
df = pd.get_dummies(df,columns = ['home_ownership','verification_status','purpose'])
print("Current shape of dataset:",df.shape)

#Replacing blank spaces with nan
df = df.replace('',np.nan)
#Converting type of dataset to float
df = df.astype(float)

#Checking for missing values
df.isnull().sum()

#Imputation of missing value with Mean in columns with missing values
df = df.apply(lambda x:x.fillna(x.mean()),axis=0)
#Checking for missing values after imputation
df.isnull().sum()

#Identifying correlation between variables 
corr = df.corr()

X = df.drop('loan_status',axis = 1)
Y = df['loan_status']
## feature selection using information gain(Filter Method)  
from sklearn.feature_selection import mutual_info_classif

important = mutual_info_classif(X,Y)
feature_importance = pd.Series(important, df.columns[0:len(df.columns)-1])
feature_importance.plot(kind= 'barh', color = 'teal' )
plt.show()
#Based on correlation and feature selection dropping more columns
X.drop(['grade','collection_recovery_fee','total_rev_hi_lim','home_ownership_RENT','purpose_P2'], axis=1, inplace=True)

#checking count of 0(non-defaulters) and 1(Defaulters)
df['loan_status'].value_counts()

#Splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

##Logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred_lr = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))
confusion_matrix_lr = confusion_matrix(Y_test, Y_pred_lr)
print(confusion_matrix_lr)
print(classification_report(Y_test, Y_pred_lr))

sns.heatmap(confusion_matrix_lr , annot=True)
plt.show()

#get the probability distribution -log regression
prob_lr = logreg.predict_proba(X_test)
print(prob_lr)

##Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
clf = RandomForestClassifier(max_depth=6, random_state=0)
clf.fit(X_train, Y_train)

Y_pred_rf = clf.predict(X_test)
print('Accuracy of Random forest classifier on test set: {:.2f}'.format(clf.score(X_test, Y_test)))

confusion_matrix_rf = confusion_matrix(Y_test, Y_pred_rf)
print(confusion_matrix_rf)
print(classification_report(Y_test, Y_pred_rf))

sns.heatmap(confusion_matrix_rf , annot=True)
plt.show()

# get the probability distribution- RFC
prob_rf = clf.predict_proba(X_test)


##Changing the threshold to 0.4 to improve recall (TPR)
#Revised probability(log regression)
Y_pred_1 = prob_lr[:,1]
Y_pred_1 = np.where(Y_pred_1>0.4,1,0)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
confusion_matrix_1 = confusion_matrix(Y_test, Y_pred_1)
print(confusion_matrix_1)
print(classification_report(Y_test, Y_pred_1))
sns.heatmap(confusion_matrix_1 , annot=True)
plt.show()

#Revised probability(RFC)
Y_pred_2 = prob_rf[:,1]
Y_pred_2 = np.where(Y_pred_2>0.4,1,0)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
confusion_matrix_2 = confusion_matrix(Y_test, Y_pred_2)
print(confusion_matrix_2)
print(classification_report(Y_test, Y_pred_2))
sns.heatmap(confusion_matrix_2 , annot=True)
plt.show()

#Plotting ROC curve for both logistic regression and RFC
from sklearn.metrics import roc_curve
prob_lr = logreg.predict_proba(X_test)[:,1]
fpr1 , tpr1, thresholds1 = roc_curve(Y_test, prob_lr)

prob_rf = clf.predict_proba(X_test)[:,1]
fpr2 , tpr2, thresholds2 = roc_curve(Y_test, prob_rf)

plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr1, tpr1, label= "Log reg")
plt.plot(fpr2, tpr2, label= "RF")

plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('Receiver Operating Characteristic')
plt.show()

#AUC score for logistic regression model
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(Y_test, prob_lr)
print(auc_score)
#AUC score for Random forest classifier model
auc_score = roc_auc_score(Y_test, prob_rf)
print(auc_score)


###TEST DATA####
##Testing the trained model 
df1 = pd.read_csv('C:/Users/drlee/Downloads/ML_Artivatic_dataset/ML_Artivatic_dataset/test_indessa.csv')

#Dropping columns not related to target variable
df1.drop(['loan_amnt','funded_amnt','batch_enrolled','sub_grade','emp_title','desc','title','zip_code','addr_state','mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog','verification_status_joint'], axis=1, inplace=True)
print(df1)


##Converting categorical values to numerical
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df1.iloc[:, 2] = encoder.fit_transform(df1.iloc[:, 2])
df1.iloc[:, 4] = encoder.fit_transform(df1.iloc[:, 4])
df1.iloc[:, 9] = encoder.fit_transform(df1.iloc[:, 9])
df1.iloc[:, 19] = encoder.fit_transform(df1.iloc[:, 19])
df1.iloc[:, 25] = encoder.fit_transform(df1.iloc[:, 25])

##Grouping purpose as P1, P2,P3
df1['purpose'] = df1['purpose'].str.replace('educational','P1')
df1['purpose'] = df1['purpose'].str.replace('wedding','P1')
df1['purpose'] = df1['purpose'].str.replace('car','P2')
df1['purpose'] = df1['purpose'].str.replace('moving','P2')
df1['purpose'] = df1['purpose'].str.replace('major_purchase','P2')
df1['purpose'] = df1['purpose'].str.replace('renewable_energy','P2')
df1['purpose'] = df1['purpose'].str.replace('small_business','P2')
df1['purpose'] = df1['purpose'].str.replace('house','P2')
df1['purpose'] = df1['purpose'].str.replace('debt_consolidation','P3')
df1['purpose'] = df1['purpose'].str.replace('home_improvement','P3')
df1['purpose'] = df1['purpose'].str.replace('credit_card','P3')
df1['purpose'] = df1['purpose'].str.replace('other','P3')
df1['purpose'] = df1['purpose'].str.replace('vacation','P3')
df1['purpose'] = df1['purpose'].str.replace('medical','P3')
df1['purpose'] = df1['purpose'].str.replace('credit_P2d','P3')
#Checking Replaced value
df1['purpose'].unique()

###Stripping of texts in "last week pay" column
df1['last_week_pay'].unique()
df1['last_week_pay'] = df1['last_week_pay'].str.replace('th week','')
df1['last_week_pay'] = df1['last_week_pay'].str.replace('NA','')

df1['emp_length'].unique()
#Replacing years with blank space;<1 year with 0; 1year with 1 and + with blank space
df1['emp_length'] = df1['emp_length'].str.replace(' years','')
df1['emp_length'] = df1['emp_length'].str.replace('< 1 year','0')
df1['emp_length'] = df1['emp_length'].str.replace('1 year','1')
df1['emp_length'] = df1['emp_length'].str.replace('+','')
df1['emp_length'].unique()

###Making Dummy variables for 'home_ownership','verifying_status','purpose'
df1 = pd.get_dummies(df1,columns = ['home_ownership','verification_status','purpose'])
print(df1)
##Replacing blank spaces with nan
df1 = df1.replace('',np.nan)
#Converting type of dataset to float
df1 = df1.astype(float)
##Checking for missing values
df1.isnull().sum()
##Imputation of missing value with Mean in columns with missing values
df1 = df1.apply(lambda x:x.fillna(x.mean()),axis=0)
#Checking after imputation
df1.isnull().sum()

#Dropping More columns(after checking correlation and feature selection in train dataset)
df1.drop(['grade','collection_recovery_fee','total_rev_hi_lim','home_ownership_RENT','purpose_P2'], axis=1, inplace=True)

df1['home_ownership_ANY'] = 0

print("Current Shape of df1 is:",df1.shape)

#Probability distribution
prob_1 = clf.predict_proba(df1)

final_pred = df1['member_id']

final_pred['probability'] = prob_1[:,1]
#Converting final_pred into dataframe
final_pred = pd.DataFrame(df1['member_id'])
#Adding probability column
final_pred['probability'] = prob_1[:,1]



# Diabetetes-Prediction-using-Machine-Learning-in-python
step1: Import the depndencies
#import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
step2: data collection and Analysis
#Load the pima indian datset
df=pd.read_csv(r"C:\Users\HP\Downloads\diabetes.csv")
step 3:Exploratory Data Analysis
3.1) Undertsanding Your Variables
3.1.1) Head of the dataset
#Display the first five records of dataset
df.head()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
0	6	148	72	35	0	33.6	0.627	50	1
1	1	85	66	29	0	26.6	0.351	31	0
2	8	183	64	0	0	23.3	0.672	32	1
3	1	89	66	23	94	28.1	0.167	21	0
4	0	137	40	35	168	43.1	2.288	33	1
#Displsy the last five records of dataset
df.tail()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
763	10	101	76	48	180	32.9	0.171	63	0
764	2	122	70	27	0	36.8	0.340	27	0
765	5	121	72	23	112	26.2	0.245	30	0
766	1	126	60	0	0	30.1	0.349	47	1
767	1	93	70	31	0	30.4	0.315	23	0
#Display randomly any numbers of records of dataset
df.sample(5)
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
581	6	109	60	27	0	25.0	0.206	27	0
325	1	157	72	21	168	25.6	0.123	24	0
610	3	106	54	21	158	30.9	0.292	24	0
276	7	106	60	24	0	26.5	0.296	29	1
225	1	87	78	27	32	34.6	0.101	22	0
3.1.2 The shape of the dataset
#numbers of rows and column
df.shape
(768, 9)
3.1.3 List type of all columns
#List type of all columns
df.dtypes
Pregnancies                   int64
Glucose                       int64
BloodPressure                 int64
SkinThickness                 int64
Insulin                       int64
BMI                         float64
DiabetesPedigreeFunction    float64
Age                           int64
Outcome                       int64
dtype: object
3.1.4) Info of dataset
#finding out if the  dataset conatain any null values
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
3.1.5 ) Summary of the datadset
#Statistical summary
df.describe()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
count	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000
mean	3.845052	120.894531	69.105469	20.536458	79.799479	31.992578	0.471876	33.240885	0.348958
std	3.369578	31.972618	19.355807	15.952218	115.244002	7.884160	0.331329	11.760232	0.476951
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.078000	21.000000	0.000000
25%	1.000000	99.000000	62.000000	0.000000	0.000000	27.300000	0.243750	24.000000	0.000000
50%	3.000000	117.000000	72.000000	23.000000	30.500000	32.000000	0.372500	29.000000	0.000000
75%	6.000000	140.250000	80.000000	32.000000	127.250000	36.600000	0.626250	41.000000	1.000000
max	17.000000	199.000000	122.000000	99.000000	846.000000	67.100000	2.420000	81.000000	1.000000
3.2) Data cleaning
3.2.1)drop the Duplicates
#check the shape before the duplicates 
df.shape
(768, 9)
df=df.drop_duplicates()
#chck the shape after the duplicate
df.shape
(768, 9)
3.2.2)check the null value
#Count of null  values
#check the missing values in any column
#display number of null values in every column in dataset
df.isnull().sum()
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64
#there is no null values in datasdet
df.columns
Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')
check the no. of Zero values in given datset
print("No.of zero values in Glucose",df[df['Glucose']==0].shape[0])
No.of zero values in Glucose 5
print("No.of zero values in BloodPressure",df[df['BloodPressure']==0].shape[0])
No.of zero values in BloodPressure 35
print("No.of zero values in SkinThickness",df[df['SkinThickness']==0].shape[0])
No.of zero values in SkinThickness 227
print("No.of zero values in Insulin",df[df['Insulin']==0].shape[0])
No.of zero values in Insulin 374
print("No.of zero values in BMI",df[df['BMI']==0].shape[0])
No.of zero values in BMI 11
Replace no of zero values with mean of the columns
df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
print("No.of zero values in Glucose",df[df['Glucose']==0].shape[0])
No.of zero values in Glucose 0
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
print("No.of zero values in BloodPressure",df[df['BloodPressure']==0].shape[0])
No.of zero values in BloodPressure 0
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())
print("No.of zero values in SkinThickness",df[df['SkinThickness']==0].shape[0])
No.of zero values in SkinThickness 0
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
print("No.of zero values in Insulin",df[df['Insulin']==0].shape[0])
No.of zero values in Insulin 0
df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
print("No.of zero values in BMI",df[df['BMI']==0].shape[0])
​
No.of zero values in BMI 0
df.describe()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
count	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000
mean	3.845052	121.681605	72.254807	26.606479	118.660163	32.450805	0.471876	33.240885	0.348958
std	3.369578	30.436016	12.115932	9.631241	93.080358	6.875374	0.331329	11.760232	0.476951
min	0.000000	44.000000	24.000000	7.000000	14.000000	18.200000	0.078000	21.000000	0.000000
25%	1.000000	99.750000	64.000000	20.536458	79.799479	27.500000	0.243750	24.000000	0.000000
50%	3.000000	117.000000	72.000000	23.000000	79.799479	32.000000	0.372500	29.000000	0.000000
75%	6.000000	140.250000	80.000000	32.000000	127.250000	36.600000	0.626250	41.000000	1.000000
max	17.000000	199.000000	122.000000	99.000000	846.000000	67.100000	2.420000	81.000000	1.000000
4)data visualization
4.1)Count plot
#outcome count plot
import seaborn as sns
sns.countplot(df['Outcome'],label="Count")
<AxesSubplot:xlabel='Outcome', ylabel='count'>

4.2)Histograms
# Histogram of each feature
df.hist(bins=10,figsize=(10,10))
plt.show()

4.3)Scatter plot
#Scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(20,20));

4.4)Pairplot
#Pairplot
sns.pairplot(data=df,hue="Outcome")
plt.show()

4.5)Analyzing relationships between variables
correlationa Analysis
import seaborn as sns
#get correlation of each datset feature in dataset
corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(10,10))
#plot heatmap
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

5)split the dataframe into X & Y
target_name ="Outcome"
#separate object for target feature
Y=df[target_name]
​
#separate object for input featur
X=df.drop(target_name,axis=1)
​
​
X.head()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age
0	6	148.0	72.0	35.000000	79.799479	33.6	0.627	50
1	1	85.0	66.0	29.000000	79.799479	26.6	0.351	31
2	8	183.0	64.0	20.536458	79.799479	23.3	0.672	32
3	1	89.0	66.0	23.000000	94.000000	28.1	0.167	21
4	0	137.0	40.0	35.000000	168.000000	43.1	2.288	33
Y.head()
0    1
1    0
2    1
3    0
4    1
Name: Outcome, dtype: int64
6)Apply Feature Scalling
# apply the standard scalar
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
SSX=scaler.transform(X)
7) TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SSX,Y,test_size=0.2,random_state=4)
X_train.shape,Y_train.shape
((614, 8), (614,))
X_test.shape,Y_test.shape
((154, 8), (154,))
8)Bulid the CLASSIFICATION Algorithms
8.1)Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train,Y_train)
LogisticRegression(multi_class='ovr', solver='liblinear')
8.2)KNeighboursClassifier() (KNN)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
KNeighborsClassifier()
8.3) Naive_Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,Y_train)
GaussianNB()
8.4) Support Vector Machine (SVM)
from sklearn.svm import SVC
sv=SVC()
sv.fit(X_train,Y_train)
SVC()
8.5) Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)
DecisionTreeClassifier()
8.6) Random Forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(criterion='entropy')
rf.fit(X_train,Y_train)
RandomForestClassifier(criterion='entropy')
9) Making Prediction
9.1) Making Prediction on test by using Logistic Rregression
X_test.shape
(154, 8)
# Making Prediction on test datset
lr_pred=lr.predict(X_test)
lr_pred.shape
(154,)
9.2) Making Prediction on test by using KNN
#Making predition on test dataset 
knn_pred=knn.predict(X_test)
knn_pred.shape
(154,)
9.3) Making prediction on test by using Naivie Bayes
# Making peditions on test dataset
nb_pred=nb.predict(X_test)
nb_pred.shape
(154,)
9.4) Making prediction on test by using SVM
# Making peditions on test dataset
sv_pred=sv.predict(X_test)
sv_pred.shape
(154,)
9.5) Making prediction on test by using Decision Tree
# Making peditions on test dataset
dt_pred=dt.predict(X_test)
dt_pred.shape
(154,)
9.6) Making prediction on test by using Random Forest
# Making peditions on test dataset
rf_pred=rf.predict(X_test)
rf_pred.shape
(154,)
10) Model Evaluation
10.1)Train Score & Test Score
#Train Score & Test Score of Logistic  Regression
from sklearn.metrics import accuracy_score
print("Train Accuracy of Logistic Regression",lr.score(X_train,Y_train)*100)
print("Accuracy (Test) score of Logistic Regression",lr.score(X_test,Y_test)*100)
print("Accuracy score of Logistic Regression",accuracy_score(Y_test,lr_pred)*100)
Train Accuracy of Logistic Regression 76.71009771986971
Accuracy (Test) score of Logistic Regression 80.51948051948052
Accuracy score of Logistic Regression 80.51948051948052
#Train Score & Test Score of KNN
from sklearn.metrics import accuracy_score
print("Train Accuracy of KNN",knn.score(X_train,Y_train)*100)
print("Accuracy (Test) score of KNN",knn.score(X_test,Y_test)*100)
print("Accuracy score of KNN",accuracy_score(Y_test,knn_pred)*100)
Train Accuracy of KNN 83.71335504885994
Accuracy (Test) score of KNN 72.72727272727273
Accuracy score of KNN 72.72727272727273
#Train Score & Test Score of Naivie Bayes
from sklearn.metrics import accuracy_score
print("Train Accuracy of Naivie Bayes",nb.score(X_train,Y_train)*100)
print("Accuracy (Test) score of Naivie Bayes",nb.score(X_test,Y_test)*100)
print("Accuracy score of Naivie Bayes",accuracy_score(Y_test,nb_pred)*100)
Train Accuracy of Naivie Bayes 74.5928338762215
Accuracy (Test) score of Naivie Bayes 74.67532467532467
Accuracy score of Naivie Bayes 74.67532467532467
#Train Score & Test Score of SVM
from sklearn.metrics import accuracy_score
print("Train Accuracy of SVM",sv.score(X_train,Y_train)*100)
print("Accuracy (Test) score of SVM",sv.score(X_test,Y_test)*100)
print("Accuracy score of SVM",accuracy_score(Y_test,sv_pred)*100)
Train Accuracy of SVM 82.89902280130293
Accuracy (Test) score of SVM 76.62337662337663
Accuracy score of SVM 76.62337662337663
#Train Score & Test Score of Decision Tree
from sklearn.metrics import accuracy_score
print("Train Accuracy of Decision Tree",df.score(X_train,Y_train)*100)
print("Accuracy (Test) score of Decision Tree",dt.score(X_test,Y_test)*100)
print("Accuracy score of Decision Tree",accuracy_score(Y_test,dt_pred)*100)
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-73-1b440103addb> in <module>
      1 #Train Score & Test Score of Decision Tree
      2 from sklearn.metrics import accuracy_score
----> 3 print("Train Accuracy of Decision Tree",df.score(X_train,Y_train)*100)
      4 print("Accuracy (Test) score of Decision Tree",dt.score(X_test,Y_test)*100)
      5 print("Accuracy score of Decision Tree",accuracy_score(Y_test,dt_pred)*100)

~\anaconda3\lib\site-packages\pandas\core\generic.py in __getattr__(self, name)
   5463             if self._info_axis._can_hold_identifiers_and_holds_name(name):
   5464                 return self[name]
-> 5465             return object.__getattribute__(self, name)
   5466 
   5467     def __setattr__(self, name: str, value) -> None:

AttributeError: 'DataFrame' object has no attribute 'score'

#Train Score & Test Score of Random Forest
from sklearn.metrics import accuracy_score
print("Train Accuracy of Random Forest",rf.score(X_train,Y_train)*100)
print("Accuracy (Test) score of Random Forest",rf.score(X_test,Y_test)*100)
print("Accuracy score of Random Forest",accuracy_score(Y_test,rf_pred)*100)
10.2) Confusion Martix
10.2.1)Confusion matrix of "Logistic Regression"
from sklearn.metrics import classification_report,confusion_matrix
# confusion martix of logical Regresion
cm=confusion_matrix(Y_test,lr_pred)
cm
sns.heatmap(confusion_matrix(Y_test,lr_pred),annot=True,fmt="d")
##print("Classification Report of logical Regresion:\n",classification_report(Y_test,lr_pred,digits=4))
TN=cm[0,0]
FP=cm[0,1]
FN=cm[1,0]
TP=cm[1,1]
TN,FP,FN,TP
# Making Confusion martix of logical Regresion 
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,lr_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))
80.51948051948052+19.480519480519483
import matplotlib.pyplot as plt 
plt.clf()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Wistia)
classNames=['0','1']
plt.title("Confusion Matrix Of Logical Regression ")
plt.ylabel("Actual(true) values")
plt.xlabel("Predicted values")
tick_marks=np.arange(len(classNames))
plt.xticks(tick_marks, classNames,rotation=45)
plt.yticks(tick_marks,classNames)
s=[['TN','FP'],['FN','TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()
pd.crosstab(Y_test,lr_pred,margins=False)
pd.crosstab(Y_test,lr_pred,margins=True)
pd.crosstab(Y_test,lr_pred, rownames=['Actual value'], colnames=['predicted vlues'],margins=True)
PRECISION(PPV-Positive Prediction Value)
TP,FP
Precision=TP/(TP+FP)
Precision
33/(33+11)
# print precision score
​
precision_score = TP/float(TP+FP)*100
print("Precision_Score : {0:0.4} ",format(precision_score))
from sklearn.metrics import precision_score
print("precision_score is :",precision_score(Y_test,lr_pred)*100)
print("Micro Average precision score is ",precision_score(Y_test,lr_pred,average='micro')*100)
print("Macro Average precision score is ",precision_score(Y_test,lr_pred,average='macro')*100)
print("Weighetd Average precision score is ",precision_score(Y_test,lr_pred,average='weighted')*100)
print("precision score on non weighted score is ",precision_score(Y_test,lr_pred,average=None)*100)
print("Classification Report of Logical Regresion:\n",classification_report(Y_test,lr_pred,digits=4))
Recall(True Positive Rate(TPR))
recall_score=TP/float(TP+FN)*100
print("Recall_Score",recall_score)
TP,FN
33/(33+24)
from sklearn.metrics import recall_score
print(" Recall Sensitivity score : ",recall_score(Y_test,lr_pred)*100)
print("Micro Average Recall score is : ",recall_score(Y_test,lr_pred,average='micro')*100)
print("Macro Average Recall score is : ",recall_score(Y_test,lr_pred,average='macro')*100)
print("Weighetd Average Recall score is : ",recall_score(Y_test,lr_pred,average='weighted')*100)
print("Recall score on Non weighted score is : ",recall_score(Y_test,lr_pred,average=None)*100)
False Positive rate (FRP)
FPR=FP/float(FP+TN)*100
print("False Positive Rate : {0:0.4} ",format(FPR))
FP,TN
11/(11+86) 
Specificity
specificity=TN/float(TN+FP)*100
print("Specificity : {0:0.4f} ",format(specificity))
F1-Score
from sklearn.metrics import f1_score
print(" f1_score of macro : ",f1_score(Y_test,lr_pred)*100)
print("Micro Average f1_score is : ",f1_score(Y_test,lr_pred,average='micro')*100)
print("Macro Average f1_score is : ",f1_score(Y_test,lr_pred,average='macro')*100)
print("Weighetd Average f1_score is : ",f1_score(Y_test,lr_pred,average='weighted')*100)
print("f1_score on Non weighted score is : ",f1_score(Y_test,lr_pred,average=None)*100)
Classification Report of Logical Regression
from sklearn.metrics import classification_report
print("Classification Report of Logical Regresion:\n",classification_report(Y_test,lr_pred,digits=4))
ROC Curve & ROC AUC
# Area under curve Logistic Regression
auc = roc_auc_score(Y_test,lr_pred)
print("ROC Curve & ROC AUC Logistic Regression is",auc)
fpr,tpr,thresholds=roc_curve(Y_test,lr_pred)
plt.plot(fpr,tpr,color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC curve(area=%0.2f) '%auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('Receiver Operating Characteristic (ROC) Curve of Logistic Regression')
plt.legend()
plt.grid()
plt.show()
10.2.2 Confusion Matrix of "KNN"
sns.heatmap(confusion_matrix(Y_test,knn_pred),annot=True,fmt="d")
# Making Confusion martix of KNN
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,knn_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))
72.72727272727273+27.27272727272727
Area under curve of KNN
# Area under curve of KNN
auc = roc_auc_score(Y_test,knn_pred)
print("ROC Curve & ROC AUC KNN is",auc)
fpr,tpr,thresholds=roc_curve(Y_test,knn_pred)
plt.plot(fpr,tpr,color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC curve(area=%0.2f) '%auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('Receiver Operating Characteristic (ROC) Curve of KNN')
plt.legend()
plt.grid()
plt.show()
10.2.3) Confusion Matrix of "Naivie Bayes"
# Making Confusion martix of Naivie Bayes
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,nb_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))
74.67532467532467+25.324675324675322
sns.heatmap(confusion_matrix(Y_test,nb_pred),annot=True,fmt="d")
Classification Report of Naivie Bayes
from sklearn.metrics import classification_report
print("Classification Report of Logical Regresion:\n",classification_report(Y_test,nb_pred,digits=4))
ROC AUC Score of Naivie Bayes
# Area under curve Naivie Bayes
auc = roc_auc_score(Y_test,nb_pred)
print("ROC Curve & ROC AUC Naivie Bayes is",auc)
10.2.4)fusion Matrix of "SVM"
sns.heatmap(confusion_matrix(Y_test,sv_pred),annot=True,fmt="d")
# Making Confusion martix of SVM
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,sv_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))
76.62337662337663+23.376623376623375
Classification Report of SVM
print("Classification Report of SVM:\n",classification_report(Y_test,sv_pred,digits=4))
ROC AUC Score of SVM
from sklearn.metrics import roc_auc_score
auc = round(roc_auc_score(Y_test,sv_pred)*100,2)
print("roc_auc_score SVC is",auc)
​
fpr,tpr,thresholds=roc_curve(Y_test,sv_pred)
plt.plot(fpr,tpr,color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC curve(area=%0.2f) '%auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('Receiver Operating Characteristic (ROC) Curve of KNN')
plt.legend()
plt.grid()
plt.show()
10.2.5) Confusion Matrix of "Decision Tree"
# Making Confusion martix of Decision Tree
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,dt_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))
72.72727272727273+ 27.27272727272727
Classification Report of Decision Tree
print("Classification Report of Decision Tree:\n",classification_report(Y_test,dt_pred,digits=4))
#ROC AUC Score of Decision Tree
from sklearn.metrics import roc_auc_score
auc = round(roc_auc_score(Y_test,dt_pred)*100,2)
print("roc_auc_score Decision Tree is",auc)
​
fpr,tpr,thresholds=roc_curve(Y_test,dt_pred)
plt.plot(fpr,tpr,color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC curve(area=%0.2f) '%auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('Receiver Operating Characteristic (ROC) Curve of Decision Tree')
plt.legend()
plt.grid()
plt.show()
sns.heatmap(confusion_matrix(Y_test,dt_pred),annot=True,fmt="d")
10.2.6)Confusion Matrix of "Random Forest"
# Making Confusion martix of Random Forest
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,rf_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))
75.97402597402598+24.025974025974026
Classification Report of Random Forest
print("Classification Report of Random Forest:\n",classification_report(Y_test,rf_pred,digits=4))
#ROC AUC Score of Random Forest
from sklearn.metrics import roc_auc_score
auc = round(roc_auc_score(Y_test,rf_pred)*100,2)
print("roc_auc_score Random Forest  is",auc)
​
fpr,tpr,thresholds=roc_curve(Y_test,rf_pred)
plt.plot(fpr,tpr,color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC curve(area=%0.2f) '%auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('Receiver Operating Characteristic (ROC) Curve of Random Forest')
plt.legend()
plt.grid()
plt.show()
sns.heatmap(confusion_matrix(Y_test,rf_pred),annot=True,fmt="d")
​
​


# coding: utf-8

# Importing Dataset

# In[409]:


#Importing Libraries:
import numpy as np
import pandas as pd


# In[410]:


#Reading csv file and converting it in dataframe
ds=pd.read_excel("Data_Train.xlsx")
ds1= pd.read_excel("Test_set.xlsx")
train=pd.DataFrame(ds)
test=pd.DataFrame(ds1)


# In[334]:


train.head()


# Dataset in Dataframe format.
# Regression Model

# In[335]:


test.head()


# Test dataset

# In[336]:


# Rows & Columns in dataset:
train.shape


# Rows: 10683, Column: 11 of Train dataset

# In[337]:


# Rows & Columns in dataset:
test.shape


# Rows: 2671, Column: 10 of test dataset

# In[338]:


# Datatype of dataset
train.dtypes


# 10 object datatype , 1 numeric datatype

# In[339]:


# Datatype of dataset
test.dtypes


# 10 object datatype 

# In[340]:


#Null Value:
train.isnull().sum()


# null value present in Route and Total_Stops columns.

# In[341]:


#Null Value:
test.isnull().sum()


# no null value present.

# In[411]:


train.dropna(inplace = True)
train.isnull().sum()


# No null value as null values dropped

# In[343]:


# Information about Dataset:

train.info()


# In[344]:


test.info()


# In[345]:


# Checking Unique values of train Dataset:

column_name =train.columns.values
for column in column_name:
    print("{0}: {1}".format(column, train[column].unique()))


# Unique values of train Dataset

# In[346]:


# Checking Unique values of test Dataset:

column_name =test.columns.values
for column in column_name:
    print("{0}: {1}".format(column, test[column].unique()))


# Unique values of test Dataset

# In[413]:


train['Additional_Info'].replace(['No Info','No info'])


# replacing No Info with No info

# # Data Analysis

# In[414]:


#Importing plotting libraries:

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ## Plotting of Numeric Datatype:

# In[349]:


sns.distplot(train['Price'], kde=True)


# Data is not normally distributed, skewness present

# ## Plotting categorical data

# In[198]:


plt.figure(figsize=(18,5))
ax=sns.countplot(x="Airline", data=train)
print(train["Airline"].value_counts())


# Jet Aiways Airline has max count than other airline.

# In[199]:


plt.figure(figsize=(45,5))
ax=sns.countplot(x="Date_of_Journey", data=train)
print(train["Date_of_Journey"].value_counts())


# 18/05/2019 and 06/06/2019 dates has maximumnumber of travellers.

# In[42]:


ax=sns.countplot(x="Source", data=train)
print(train["Source"].value_counts())


# Delhi has high count of 4537 

# In[31]:


ax=sns.countplot(x="Destination", data=train)
print(train["Destination"].value_counts())


# Maximum destination travelled is Cochin.

# In[6]:


plt.figure(figsize=(50,8))
ax=sns.countplot(x="Route", data=train)
print(train["Route"].value_counts())


# DEL → BOM → COK route was travelled maximum time.

# In[45]:


plt.figure(figsize=(18,5))
ax=sns.countplot(x="Dep_Time", data=train)
print(train["Dep_Time"].value_counts())


# 18:55 depature time has maximum count.

# In[46]:


plt.figure(figsize=(18,5))
ax=sns.countplot(x="Arrival_Time", data=train)
print(train["Arrival_Time"].value_counts())


# 19:00 arrival time has maximum count.

# In[47]:


plt.figure(figsize=(18,5))
ax=sns.countplot(x="Duration", data=train)
print(train["Duration"].value_counts())


# 2h 50m duration has maximum count.

# In[30]:


ax=sns.countplot(x="Total_Stops", data=train)
print(train["Total_Stops"].value_counts())


# Maximum count is of 1 stop: 5626 

# In[49]:


plt.figure(figsize=(18,5))
ax=sns.countplot(x="Additional_Info", data=train)
print(train["Additional_Info"].value_counts())


# No infor has maximum count

# In[126]:


plt.figure(figsize=(20,5))
sns.catplot(x = "Price", y = "Source", data = train)


# Price of Delhi source is high

# In[127]:


plt.figure(figsize=(20,5))
sns.catplot(x = "Price", y = "Destination", data = train)


# Common range in all destination lies between 0 to 100000

# In[128]:


plt.figure(figsize=(20,5))
sns.catplot(x = "Price", y = "Total_Stops", data = train)


# Maximum prices are for 1 stop 

# # Encoding of DataFrame:

# In[415]:


# Label Encoding for object to numeric datatype:
from sklearn.preprocessing import LabelEncoder
en= LabelEncoder()
for i in train.columns:
    if train[i].dtypes=="object":
        train[i]=en.fit_transform(train[i].values.reshape(-1,1))


# In[416]:


train.head()


# train dataset encoded.

# In[417]:


# Label Encoding for object to numeric test datatype:
from sklearn.preprocessing import LabelEncoder
en= LabelEncoder()
for i in test.columns:
    if test[i].dtypes=="object":
        test[i]=en.fit_transform(test[i].values.reshape(-1,1))


# In[418]:


test.head()


# test dataset encoded

# ## Describing the dataset

# In[419]:


train.describe()


# Key Observations:
# 
#     1. Mean and Median: Values of mean and median have differences alternatively so skewness are present.
#     2. 75% percentile and max: Airline, source, additional info and price has outliers.
#     3. Standard deviation is less than mean so high peak data.

# In[420]:


test.describe()


# Key Observations:
# 
#     1. Mean and Median: Values of mean and median have differences alternatively so skewness are present.
#     2. 75% percentile and max: Dep_Time and duration have highest outliers.
#     3. Standard deviation is less than mean so high peak data.

# In[421]:


import matplotlib.pyplot as plt
plt.figure(figsize=(22,7))
sns.heatmap(train.describe(),annot=True,linewidths=0.1,linecolor="Green",fmt="0.2f")


# Heatmap representation of dataset description

# ## Outliers

# In[422]:


train.plot(kind='box',subplots=True,layout=(4,6),figsize=(12,12))


# Airline, source, additional info and price has outliers.

# In[423]:


test.plot(kind='box',subplots=True,layout=(4,6),figsize=(12,12))


# Only additional infor has small dataset

# In[424]:


#zscore for outlier removal

from scipy.stats import zscore
import numpy as np
z = np.abs(zscore(train))
threshold=3
np.where(z>3)


# In[425]:


#removing value greater than threshold value 

df_new=train[(z<3).all(axis=1)]
train= df_new


# Outliers removed

# In[426]:


train.head()


# In[427]:


train.shape


# In[428]:


Rows: 10572 ; Column: 11


# #### Percentage Loss of data: Train dataset

# In[429]:


Data_loss=((10683-10572)/10683)*100
Data_loss


# 1% data loss

# # Correlation of the columns with the target columns:

# In[430]:


train.corr()


# correlation of input with target variable Price

# In[431]:


plt.figure(figsize=(10,7))
sns.heatmap(train.corr(),annot=True,linewidths=0.1,linecolor="Black",fmt="0.2f")


# Key Observation:
#  1. Price have positive correlation with Route and negative correlation with Total_Stops.
#  2. Route has positive correlation with Source and negative correlation with Total_Stops
#  3. Multicollinearity exist

# ## Separating Target and feature variables

# In[432]:


x= train.drop("Price",axis=1)
y= train["Price"]


# ### Multicollinearity removal using VIF
# 

# In[433]:


# Importing library:
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[434]:


def vif_calc():
    vif=pd.DataFrame()
    vif["VIF Factor"]=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
    vif["features"]=x.columns
    print(vif)


# In[435]:


vif_calc()


# Only Additional_Info has higher skewness()

# ## Data Cleaning:

# In[436]:


x.drop(['Additional_Info','Route'],axis=1,inplace=True)
x.head()


# Additional_Info dropped as its vif value is high.

# ## Skewness: 

# In[437]:


x.skew()


# Taking range of skewness between +/- 0.5, all values are under control so no need to remove skewness.

# In[438]:


from sklearn.preprocessing import power_transform
x=power_transform(x,method='yeo-johnson')


# In[439]:


#converting ndarray to dataframe:
x=pd.DataFrame(x)
#x.skew()


# In[440]:


x.skew()


# skewness removed

# # Scaling Data

# In[441]:


# Train dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)
x


# # Model Selection :
# 
# ## Linear Regression:

# In[442]:


from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# In[443]:


for i in range(0,100):
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=i)
    lr.fit(train_x,train_y)
    pred_train=lr.predict(train_x)
    pred_test=lr.predict(test_x)
    print(f"At random state {i},the training accuracy is:- {r2_score(train_y,pred_train)}")
    print(f"At random state {i},the testing accuracy is:- {r2_score(test_y,pred_test)}")
    print("\n")


# In[444]:


lr.fit(train_x,train_y)
lr.score(test_x,test_y)


# 30% is Linear regression score

# In[445]:


#Predicted data
pred=lr.predict(test_x)
pred


# In[446]:


print("Mean squared error:",mean_squared_error(test_y,pred))
print("Mean absolute error:",mean_absolute_error(test_y,pred))
print("R2Score:",r2_score(test_y,pred))


# R2Score = 30%

# ### Cross-Validation of the model:

# In[447]:


Train_accuracy=r2_score(train_y,pred_train)
Test_accuracy=r2_score(test_y,pred_test)

from sklearn.model_selection import cross_val_score
for j in range(2,10):
    cv_score=cross_val_score(lr,x,y,cv=j)
    cv_mean=cv_score.mean()
    print(f"At cross fold {j} the cv score is {cv_mean} and accuracy score for training is {Train_accuracy} and accuracy for the testing is {Test_accuracy}")
    print("\n")


# cv score calculated

# In[448]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(x=test_y, y=pred_test, color='r')
plt.plot(test_y,test_y, color='b')
plt.xlabel('Actual data',fontsize=14)
plt.ylabel('Predicted data',fontsize=14)
plt.title('Linear Regression',fontsize=18)
plt.show()


# Plotting of Predicted and actual data

# ### Regularization

# In[449]:


# Importing Libraries and Hyper parameter tuning:
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# In[450]:


from sklearn.linear_model import Lasso

parameters = {'alpha':[.0001, .001, .01, .1, 1,10],'random_state':list(range(0,10))}
ls = Lasso()
clf = GridSearchCV(ls,parameters)
clf.fit(train_x,train_y)

print(clf.best_params_)


# Best parameters for Linear Regression

# In[451]:


ls = Lasso(alpha=1,random_state=0)
ls.fit(train_x,train_y)
ls.score(train_x,train_y)
print('ls score',ls.score(train_x,train_y))
pred_ls = ls.predict(test_x)

r2s = r2_score(test_y,pred_ls)
print('r2 score',r2s*100)


# Ls score is 29% and r2score is 30.5%

# In[452]:


cv_score=cross_val_score(ls,x,y,cv=5)
cv_mean=cv_score.mean()
cv_mean


# cv score is 29%

# ## 1. Random Forest Regressor
# 

# Best parameters for Random Forest Regressor

# In[ ]:


rf= RandomForestRegressor(criterion="mse",max_features="log2")
rf.fit(train_x,train_y)
rf.score(train_x,train_y)
predrf = rf.predict(test_x)
print('rf score',rf.score(train_x,train_y))
rfs = r2_score(test_y,predrf)
print('R2 Score:',rfs*100)

rfscore = cross_val_score(rf,x,y,cv=9)
rfc = rfscore.mean()
print('Cross Val Score:',rfc*100)
print("Mean squared error:",mean_squared_error(test_y,predrf))
print("Mean absolute error:",mean_absolute_error(test_y,predrf))


# rf score 96.9% R2 Score: 77% Cross Val Score: 79%

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(x=test_y, y=predrf, color='r')
plt.plot(test_y,test_y, color='b')
plt.xlabel('Actual data',fontsize=14)
plt.ylabel('Predicted data',fontsize=14)
plt.title('Random Forest Regressor',fontsize=18)
plt.show()


# ## 2. Decision Tree Regressor

# In[454]:


# Importing Libraries and Hyper parameter tuning:
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

parameters = {'criterion':['mse', 'mae'],
              'max_features':["auto","sqrt", "log2"],
              'max_depth':[2,4,8,10,None],
            'min_samples_split':[0.25,0.5,1.0]}
dt =DecisionTreeRegressor()
clf = GridSearchCV(dt,parameters)
clf.fit(train_x,train_y)

print(clf.best_params_)


# Best parameters for DecisionTreeRegressor

# In[455]:


dt =DecisionTreeRegressor(criterion="mse",max_features="auto",max_depth=8,min_samples_split=0.25)
dt.fit(train_x,train_y)
dt.score(train_x,train_y)
predt = dt.predict(test_x)
print('dt score',dt.score(train_x,train_y))
dts = r2_score(test_y,predt)
print('R2 Score:',dts*100)

dtscore = cross_val_score(dt,x,y,cv=9)
dtc = dtscore.mean()
print('Cross Val Score:',dtc*100)
print("Mean squared error:",mean_squared_error(test_y,predt))
print("Mean absolute error:",mean_absolute_error(test_y,predt))


# dt score is 58% , r2score is 59.6% and cross val score is 57.8%

# In[456]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(x=test_y, y=predt, color='r')
plt.plot(test_y,test_y, color='b')
plt.xlabel('Actual data',fontsize=14)
plt.ylabel('Predicted data',fontsize=14)
plt.title('Decision Tree Regressor',fontsize=18)
plt.show()


# ## 3. KNeighborsRegressor

# In[457]:


# Importing Libraries and Hyper parameter tuning:
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error

parameters = {'n_neighbors':list(range(0,10)),
              'weights':['uniform', 'distance'],
              'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
              }
kn =KNeighborsRegressor()
clf = GridSearchCV(kn,parameters)
clf.fit(train_x,train_y)

print(clf.best_params_)


# Best parameters for KNeighborsRegressor 

# In[459]:


kn =KNeighborsRegressor(n_neighbors=6,algorithm="brute",weights='uniform')
kn.fit(train_x,train_y)
kn.score(train_x,train_y)
predkn = kn.predict(test_x)
print('kn score',kn.score(train_x,train_y))
kns = r2_score(test_y,predkn)
print('R2 Score:',kns*100)

knscore = cross_val_score(kn,x,y,cv=9)
knc = knscore.mean()
print('Cross Val Score:',knc*100)
print("Mean squared error:",mean_squared_error(test_y,predkn))
print("Mean absolute error:",mean_absolute_error(test_y,predkn))


# kn score 76%
# R2 Score: 68.4%
# Cross Val Score: 68.1%

# In[460]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(x=test_y, y=predkn, color='r')
plt.plot(test_y,test_y, color='b')
plt.xlabel('Actual data',fontsize=14)
plt.ylabel('Predicted data',fontsize=14)
plt.title('KNeighbors Regressor',fontsize=18)
plt.show()


# ## 4. SVRegressor

# Best parameters for SV Regressor

# In[ ]:


svr =SVR(C=1000,gamma=0.1,kernel='rbf')
svr.fit(train_x,train_y)
svr.score(train_x,train_y)
predsvr = svr.predict(test_x)
print('svr score',svr.score(train_x,train_y))
svrs = r2_score(test_y,predsvr)
print('R2 Score:',svrs*100)

svrscore = cross_val_score(svr,x,y,cv=9)
svrc = svrscore.mean()
print('Cross Val Score:',svrc*100)
print("Mean squared error:",mean_squared_error(test_y,predsvr))
print("Mean absolute error:",mean_absolute_error(test_y,predsvr))


# In[ ]:


svr score = 88.8%
r2score = 81.5%
cross val score = 75.7%


# Best Model is KNeighborRegressor as the difference between r2score and cross val score is less 

# # Model saving:

# In[462]:


import pickle
filename = 'flight_prices.pkl'
pickle.dump(kn, open(filename, 'wb'))


# ## Accuracy

# In[463]:


loaded_model = pickle.load(open('flight_prices.pkl','rb'))
result = loaded_model.score(test_x, test_y)
print(result)


# 68.4% accuracy of model.

# ## Conclusion:
# 

# In[464]:


import numpy as np
a=np.array(test_y)
predkn =np.array(kn.predict(test_x))
df_com=pd.DataFrame({"original":a, "predicted": predkn},index=range(len(a)))
df_com


# Model is able to predict values which are approximately actual data as model's accuracy is 69%

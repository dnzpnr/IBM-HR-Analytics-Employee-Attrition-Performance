import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

employee_df =   pd.read_csv('Human_Resources.csv')
df = employee_df.copy()
df.head()

df.info()

df['Attrition'].value_counts()

df['Attrition'] = df['Attrition'].apply(lambda x : 1 if x == 'Yes' else 0)
df.head()

num_cols = df.select_dtypes(include = [np.number])
num_cols.head()

num_cols.hist(bins = 30,figsize = (20,20), color = 'y');

f,ax = plt.subplots(figsize = (20,20))
sns.heatmap(num_cols.corr(), annot=True, fmt = '.2f', cmap = 'RdYlGn');

df['EmployeeCount'].value_counts()

df['StandardHours'].value_counts()

df.drop(['EmployeeCount','StandardHours'], axis = 1, inplace=True)
df.head()

at_df = df[df['Attrition'] == 1]
at_df.head()

at_df.describe()

df.describe()

df.info()

df['Age'].value_counts()

fig,ax = plt.subplots(figsize = (16,10))
ax = sns.countplot(x= 'Age', hue = 'Attrition', data = df)

df['BusinessTravel'].value_counts()

fig,ax = plt.subplots(figsize = (16,10))
ax = sns.countplot(x= 'BusinessTravel', hue = 'Attrition', data = df)

df['BusinessTravel'].replace({'Travel_Frequently':'Non-Travel'},inplace=True)
df['BusinessTravel'].value_counts()

df.drop(['EmployeeNumber'], axis=1, inplace=True)

df['HourlyRate'].value_counts()

fig,ax = plt.subplots(figsize = (40,20))
ax = sns.countplot(x= 'HourlyRate', hue = 'Attrition', data = df)

df.drop(['Over18'], axis=1, inplace= True)

df['OverTime'].value_counts()

fig,ax = plt.subplots(figsize = (16,10))
ax = sns.countplot(x= 'OverTime', hue = 'Attrition', data = df)

f,ax = plt.subplots(figsize = (16,10))
sns.scatterplot(x = 'JobLevel', y = 'MonthlyIncome', data=df)

df[['JobLevel','MonthlyIncome']].head(15)

df.drop(['JobLevel'],axis=1,inplace=True)
df.head()

df.info()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.drop(['Attrition'], axis=1),
                                                 df['Attrition'],test_size=0.15, random_state=1, shuffle=True)
print(x_train.shape, x_test.shape)

categorical_features_indices = np.where(df.drop(['Attrition'],axis=1).dtypes != np.number)[0]

from catboost import CatBoostClassifier
cat_model = CatBoostClassifier()
catb_params = {
    'iterations': [200,500],
    'learning_rate': [0.01,0.02,0.05, 0.1,0.2],
    'depth': [3,5,8,15] }
catb = CatBoostClassifier()
from sklearn.model_selection import GridSearchCV
catb_cv_model = GridSearchCV(catb, catb_params, cv=5, n_jobs = -1, verbose = 2)
catb_cv_model.fit(x_train, y_train, cat_features=categorical_features_indices)

catb_cv_model.best_params_

from catboost import CatBoostClassifier
catb = CatBoostClassifier(iterations = 500,
                          learning_rate = 0.2,
                          depth = 3)

catb_tuned = catb.fit(x_train, y_train, cat_features=categorical_features_indices)

cross_val_score(catb_tuned, X_test, y_test, cv = 10).mean()

log_df = df.loc[:,['OverTime','Attrition']].copy()
log_df.head()

log_df['OverTime'] = log_df['OverTime'].apply(lambda x: 1 if x=='Yes' else 0)
log_df.head()

X = log_df['OverTime']
y = log_df['Attrition']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=1,shuffle=True)
X_train.shape

X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)

from sklearn.linear_model import LogisticRegression
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train,y_train)

cross_val_score(loj_model, X_test, y_test, cv = 10).mean()

df.head()

new_df = pd.get_dummies(df,columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'],
                        drop_first=True)
new_df.info()

from sklearn.preprocessing import StandardScaler
df_new = StandardScaler().fit_transform(new_df.copy())
from sklearn.decomposition import PCA
pca = PCA().fit(df_new)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

pca = PCA(n_components = 30)
pca_fit = pca.fit_transform(df_new)
bilesen_df = pd.DataFrame(data = pca_fit,
                          columns = ["1","2","3",'4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19',
                                    '20','21','22','23','24','25','26','27','28','29','30'])
bilesen_df.head()

df_ = pd.concat([bilesen_df,new_df['Attrition']], axis=1,join= 'inner')
df_.head()

x_tr,x_te,y_tr,y_te = train_test_split(df_.drop(['Attrition'], axis=1),
                                                 df_['Attrition'],test_size=0.15, random_state=1, shuffle=True)
print(x_tr.shape, x_te.shape)

catb_ = CatBoostClassifier(iterations = 500,
                          learning_rate = 0.2,
                          depth = 3)

catb_tuned_ = catb.fit(x_tr, y_tr)

cross_val_score(catb_, x_te, y_te, cv = 10).mean()


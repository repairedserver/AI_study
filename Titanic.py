!unzip /content/drive/MyDrive/titanic/titanic.zip

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = 'Malgun Gothic'

train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")
sample_submission = pd.read_csv("/content/gender_submission.csv")

fe_name = list(test)
df_train = train[fe_name]
df = pd.concat((df_train, test))

target = train['Survived']

def stack_plot(feature):
  survived = train[train['Survived'] == 1][feature].value_counts()
  dead = train[train['Survived']== 0][feature].value_counts()
  df = pd.DataFrame([survived , dead])
  df.index = ['survived','dead']
  df.plot(kind='bar', stacked=True, figsize = (10, 5))

train[train['Survived'] == 1]

label = ['dead', 'survived']
plt.title('생존 수')
plt.pie(train['Survived'].value_counts(),labels= label,autopct='%.f%%')

train['Survived'].value_counts()

stack_plot("Pclass")

stack_plot("Sex")

Pclass_encoded = pd.get_dummies(df['Pclass'],prefix='Pclass')
df = pd.concat((df, Pclass_encoded), axis=1)
df = df.drop(columns = 'Pclass')

sex_encoded = pd.get_dummies(df['Sex'],prefix='Sex')
df = pd.concat((df, sex_encoded), axis=1)
df = df.drop(columns = 'Sex')

stack_plot('SibSp')
stack_plot('Parch')

df['Travelpeople']=df["SibSp"]+df["Parch"]
df['TravelAlone']=np.where(df['Travelpeople']>0,0,1)

df.drop('SibSp', axis=1, inplace=True)
df.drop('Parch', axis=1, inplace=True)
df.drop('Travelpeople', axis=1, inplace=True)

df['Name']

df.drop('Name', axis=1, inplace=True)
df['Age'].hist(bins = 15)
df['Age'].fillna(28, inplace=True)
df['Age'].hist(bins = 15)

sns.countplot(x='Embarked', data=df)
df['Embarked'].fillna('S', inplace=True)
sns.countplot(x='Embarked', data =df)

from scipy.stats import norm
sns.distplot(train['Fare'],fit=norm)

df['Fare'] = df['Fare'].map(lambda i : np.log(i) if i > 0 else 0)
sns.distplot(df['Fare'],fit=norm)

df.isnull().sum()
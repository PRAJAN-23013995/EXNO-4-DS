# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```

<img width="1110" height="681" alt="Screenshot 2025-09-29 114228" src="https://github.com/user-attachments/assets/9cbaab86-750a-463b-a2bc-7d64e39fd750" />

```
data.isnull().sum()
```

<img width="237" height="276" alt="Screenshot 2025-09-29 114302" src="https://github.com/user-attachments/assets/3626b06f-5818-4a5c-812c-d4e709803c54" />

```
missing=data[data.isnull().any(axis=1)]
missing
```

<img width="1112" height="667" alt="Screenshot 2025-09-29 114338" src="https://github.com/user-attachments/assets/c8c8b29e-88bb-4f4a-9c63-a0326a7352ed" />

```
data2=data.dropna(axis=0)
data2
```

<img width="1119" height="668" alt="Screenshot 2025-09-29 114410" src="https://github.com/user-attachments/assets/1c59551a-9233-41c5-aef7-8dc50ef42666" />


```
data2=data.dropna(axis=0)
data2
```
<img width="431" height="231" alt="Screenshot 2025-09-29 114452" src="https://github.com/user-attachments/assets/0534dd34-d214-4e07-a0e1-de25e7632342" />

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="446" height="393" alt="Screenshot 2025-09-29 114522" src="https://github.com/user-attachments/assets/6e71ee4a-9519-4ad0-a9ae-a992a2a32c1f" />


```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="1125" height="519" alt="Screenshot 2025-09-29 114548" src="https://github.com/user-attachments/assets/b425888a-d0e1-4f3f-889f-df2d3456c84c" />


```
data2
```

<img width="1126" height="454" alt="Screenshot 2025-09-29 114622" src="https://github.com/user-attachments/assets/47947474-8dcb-4429-b0ec-50f3d6f3945d" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="1126" height="454" alt="Screenshot 2025-09-29 114622" src="https://github.com/user-attachments/assets/7a859a02-362c-499c-98ef-41f97d0cd73c" />

```
columns_list=list(new_data.columns)
print(columns_list)
```

<img width="1129" height="371" alt="Screenshot 2025-09-29 114650" src="https://github.com/user-attachments/assets/223829c4-208c-4452-a61d-2d27f337a3eb" />


```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

<img width="1122" height="372" alt="Screenshot 2025-09-29 114726" src="https://github.com/user-attachments/assets/d345d518-7c33-4104-85ad-1162b71141c0" />

```
y=new_data['SalStat'].values
print(y)
```

<img width="190" height="32" alt="Screenshot 2025-09-29 114759" src="https://github.com/user-attachments/assets/9d187614-afaf-45c7-acae-dfa0d41c1e4b" />

```
x=new_data[features].values
print(x)
```

<img width="286" height="143" alt="Screenshot 2025-09-29 114829" src="https://github.com/user-attachments/assets/4966f6a6-c786-4da1-9c07-35d66d8b2e79" />

```

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```

<img width="316" height="92" alt="Screenshot 2025-09-29 114901" src="https://github.com/user-attachments/assets/14ef1079-dd57-4f60-b601-abf6dc12ffff" />


```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="170" height="47" alt="Screenshot 2025-09-29 114927" src="https://github.com/user-attachments/assets/6c021d1a-7375-4fe7-bd7e-fa465e9b5103" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

<img width="1136" height="29" alt="Screenshot 2025-09-29 114954" src="https://github.com/user-attachments/assets/abd60ef0-c01d-4df4-9f31-822f260f73cd" />


```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="288" height="22" alt="image" src="https://github.com/user-attachments/assets/9906d5dd-b920-46fd-9aca-4131ea42db3b" />

```
data.shape
```

<img width="156" height="32" alt="image" src="https://github.com/user-attachments/assets/c8faa545-37c0-4d6e-bb48-fca09ad67aa6" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="363" height="47" alt="image" src="https://github.com/user-attachments/assets/ef97c8b1-a50e-4c0d-83a4-85b1c4fb1083" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="432" height="176" alt="image" src="https://github.com/user-attachments/assets/233b0c08-882b-48dd-bda4-75038df781b6" />

```
tips.time.unique()
```

<img width="396" height="50" alt="image" src="https://github.com/user-attachments/assets/02e7d957-8116-4fb9-aac2-59d1e66fd386" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="273" height="80" alt="image" src="https://github.com/user-attachments/assets/5a2ef126-daae-414a-9c6b-846d114aec19" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="411" height="47" alt="image" src="https://github.com/user-attachments/assets/7a22fb2e-045b-46da-b906-abca25431ee4" />

# RESULT:

To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file.

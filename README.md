# Bank Customer Churn Analysis

Churner is generally defined as a customer who stops using a product or service for a given period of time.

This notebook is to do the data analysis and predictions on the churn.csv file.

The first step in the Data Preprocessing is to import the libraries, load the data and do some Exploratory Data Analysis (EDA).
## Import all important Libraries and dataset
```python

import pandas as pd    #for EDA
import numpy as np     #for numerical operation if required
from pySankey.sankey import sankey   #for sankey plot
import matplotlib.pyplot as plt      #for visualization

import seaborn as sns
%matplotlib inline
```
## Loading Dataset
```python
dataset=pd.read_csv(r'/home/ajaygoswami/Documents/Bobby/DATA SET/churn.csv')
#importing dataset and put it in a variable called dataset
dataset.head()
```

## Funcions to show acuuracy report
```python
Accuracy_Report = pd.DataFrame(columns=["Models","Accuracy"])
models_lis, acc_lis = [], []
def Submit_Score(lis1,lis2):
    models_lis.append(lis1)
    acc_lis.append(lis2)
    return
    
def Show_Model_Score():
    temp_df = pd.DataFrame({'Models': models_lis, 'Accuracy': acc_lis})
    return temp_df
 ```



## Function for plotting the confusion matrix
```python
def plot_Confusion_matrix(cm, target_names, cmap, title, accuracy):
    
    Submit_Score(title,accuracy)
    
    if cmap is None:
            cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    for i in range(2):
        for j in range(2):
            text = plt.text(j, i, cm[i, j],
                           ha="center", va="center", color="black")


    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}'.format(accuracy))
    plt.show()
```

## 1. Exploratory Data Analysis
### for dataset information
```python
dataset.info()  

dataset.head()
dataset.shape  #understanding the shape of our dataset



dataset.isna().sum()  #finding all the null values as per each column



dataset.describe() #for descriptive information(statical)


round(dataset.describe(),2)  
#'''rounding off all the decimal values up-to 2 place'''
```


dataset.head()

### '''exit = ONLY ROWS WHERE Exited == 1'''
```python
exit=dataset[dataset["Exited"]==1]
exit
```


### '''notexit = ONLY ROWS WHERE Exited == 0'''
```python
notexit=dataset[dataset["Exited"] ==0]
notexit
```



### Return given columns of notexit
```python

notexit[["Surname","CreditScore","Balance"]]
```

### calculating Exited and not_exited client
```python

exited=len(dataset[dataset["Exited"] ==0])
notexited=len(dataset[dataset["Exited"] ==1])
print("length of exited persons :- {}".format(exited))
#return length of exited persons
print("length of exited persons :- {}".format(notexited))
#return length of notexited persons

length of exited persons :- 7963
length of exited persons :- 2037
```

### finding the percentage of exited and notexited persons from the length of whole dataset 
### and rounded of its decimal value up to 2 place
```python

exited_perc=round(exited/len(dataset)*100,2)
notexited_perc=round(notexited/len(dataset)*100,2)

print("Exited :- {} %".format(exited_perc))
print("Not-exited :- {} %".format(notexited_perc))

Exited :- 79.63 %
Not-exited :- 20.37 %
```


## plotting Exited and not_exited persons as per ratio
```python
labels=["Exited","Not_exited"]
data = [exited_perc,notexited_perc]
colors=["#95f542","#02f2ee"]
plt.pie(data, 
        labels = labels,
        colors=colors,
        autopct="%0.2f%%")            # after point. numbers
plt.show()

So, around of 20% of the clients exited the bank, while around 80% stayed. As the goal here is to identify which of the customers are at higher risk to discontinue their services with the bank, we are dealing with a classification problem.

important point to take into consideration here is that we are dealing with an imbalanced dataset.

```
### finding the unique values of country and gender wih ".unique()/.nunique()" methods
```python
country = list(dataset["Geography"].unique()) 
gender = list(dataset["Gender"].unique())



print(country)
print(gender)

['France', 'Spain', 'Germany']
['Female', 'Male']
```

### creating a new column with the help of "Exited" column where numeric values(0,1) converted as-
### ..................string values(stayed,exited))repectively
```python
dataset["Exited_str"]=dataset["Exited"]
dataset["Exited_str"]=dataset["Exited_str"].map({1:'Exited',0:"Stayed"})


dataset["Exited_str"]

Out[78]:

0       Exited
1       Stayed
2       Exited
3       Stayed
4       Stayed
         ...  
9995    Stayed
9996    Stayed
9997    Exited
9998    Exited
9999    Stayed
```
### Here we can see that those clients who consist one product exited the most and other side who has2 product has mazority in stayed clients.
```python
sns.catplot("Exited_str",data=dataset,kind="count",hue="NumOfProducts")



<seaborn.axisgrid.FacetGrid at 0x7f22e620a510>
```

### '''We can also convert other modified column with the help of given columns'''
```python
#exampole:-
#dataset["Geography_str"]=dataset["Geography"]
#dataset["Geography_str"]=dataset["Geography_str"].map({'Spain':'Spa','France':"Fra",'Germany':'Ger'})
#dataset["Geography_str"]

gender_count=dataset["Gender"].value_counts() 
#return different categorical values of gender column
gender_count



Male      5457
Female    4543
Name: Gender, dtype: int64

gender_pct=gender_count/len(dataset.index) *100    #(ratio/percentage)
gender_pct

Male      54.57
Female    45.43
Name: Gender, dtype: float64
```
### concate the count and percentage different gender wise
```python
gender=pd.concat([gender_count,round(gender_pct,2)],axis=1)   #to bring it in rows
gender=pd.concat([gender_count,gender_pct],axis=1)

gender

Out[83]:
	Gender 	Gender
Male 	5457 	54.57
Female 	4543 	45.43

gender=pd.concat([gender_count,round(gender_pct,2)],axis=1).set_axis(['count','pct'],axis=1,inplace=False)
#changing the axis name as per our understanding
gender

Out[84]:
	count 	pct
Male 	5457 	54.57
Female 	4543 	45.43
In [85]:
```
### counting the geographical values and concat
```python

geography_count=dataset["Geography"].value_counts() #return different categorical values
geography_count

Out[85]:

France     5014
Germany    2509
Spain      2477
Name: Geography, dtype: int64

In [86]:

geography_pct=geography_count/len(dataset.index) *100    #(ratio)
geography_pct

Out[86]:

France     50.14
Germany    25.09
Spain      24.77
Name: Geography, dtype: float64

In [87]:

geography=pd.concat([geography_count,round(geography_pct,2)],axis=1).set_axis(['G_count','G_pct'],axis=1,inplace=False)
geography

Out[87]:
	G_count 	G_pct
France 	5014 	50.14
Germany 	2509 	25.09
Spain 	2477 	24.77
```


### plotting MALE and FEMALE  and geographical ratio
```python
labels=["Males","Females"]
colors=["#16915c","#f25ae6"]
plt.pie(gender_pct,
        colors=colors,
        labels = labels,
         autopct="%0.2f%%"
       )            # after point. numbers
plt.show()


#Plotting GEOGRAPHICAL RATIO

labels=["France","Germany","Spain"]
colors=["#51f5ae","#ebf060","#ee78f0"]
plt.pie(geography_pct,
        colors=colors,
        labels = labels,
        autopct="%0.2f%%"
       )            # after point. numbers
plt.show()

In the dataset, there are more men (55%) than women (45%), and it has only 3 different countries: France, Spain, and Germany. Where 50% of the customers are from France and 25% are from Germany, and the other group are from Spain.
```
#### Now, let's just check the relationship between the features and the outcome ('Exited'). and plot it
```python

salary_min=dataset[dataset["EstimatedSalary"]<30000]
salary_min

Out[90]:
	RowNumber 	CustomerId 	Surname 	CreditScore 	Geography 	Gender 	Age 	Tenure 	Balance 	NumOfProducts 	HasCrCard 	IsActiveMember 	EstimatedSalary 	Exited 	Exited_str
6 	7 	15592531 	Bartlett 	822 	France 	Male 	50 	7 	0.00 	2 	1 	1 	10062.80 	0 	Stayed
12 	13 	15632264 	Kay 	476 	France 	Female 	34 	10 	0.00 	2 	1 	0 	26260.98 	0 	Stayed
16 	17 	15737452 	Romeo 	653 	Germany 	Male 	58 	1 	132602.88 	1 	1 	0 	5097.67 	1 	Exited
17 	18 	15788218 	Henderson 	549 	Spain 	Female 	24 	9 	0.00 	2 	1 	1 	14406.41 	0 	Stayed
23 	24 	15725737 	Mosman 	669 	France 	Male 	46 	3 	0.00 	2 	0 	1 	8487.75 	0 	Stayed
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
9974 	9975 	15695474 	Barker 	583 	France 	Male 	33 	7 	122531.86 	1 	1 	0 	13549.24 	0 	Stayed
9977 	9978 	15579969 	Mancini 	683 	France 	Female 	32 	9 	0.00 	2 	1 	1 	24991.92 	0 	Stayed
9979 	9980 	15692664 	Diribe 	677 	France 	Female 	58 	1 	90022.85 	1 	0 	1 	2988.28 	0 	Stayed
9987 	9988 	15588839 	Mancini 	606 	Spain 	Male 	30 	8 	180307.73 	2 	1 	1 	1914.41 	0 	Stayed
9993 	9994 	15569266 	Rahman 	644 	France 	Male 	28 	7 	155060.41 	1 	1 	0 	29179.52 	0 	Stayed

1478 rows × 15 columns
In [91]:

salary_min_count=len(dataset[dataset["EstimatedSalary"]<30000])
salary_max_count=len(dataset[dataset["EstimatedSalary"]>30000])
print("persons sallery below 30,000 :- {}".format(salary_min_count))
print("persons sallery above 30,000 :- {}".format(salary_max_count))

persons sallery below 30,000 :- 1478
persons sallery above 30,000 :- 8522

In [92]:

salary_min_pct=salary_min_count/len(dataset.index)*100
salary_max_pct=salary_max_count/len(dataset.index)*100
print("(salary < 30,000) % :- {}".format(salary_min_pct))
print("(salary > 30,000) % :- {}".format(salary_max_pct))

(salary < 30,000) % :- 14.78
(salary > 30,000) % :- 85.22

In [93]:

labels=["salary under 30000","salary above 30000"]
data=[salary_min_pct,salary_max_pct]
colors=["skyblue",'lightgreen']
plt.pie(data,labels=labels,colors=colors,autopct="%0.2f%%")
plt.show()

    Here we can see the salary Percentage of clients with salary more than 30,000 are 85% and below 30,000 are 15%
```
#### return different categorical values of Hascreditcard/Dont_havecreditcards and percebntage
 ```python
HasCrCard_count=dataset["HasCrCard"].value_counts() 

HasCrCard_count

Out[94]:

1    7055
0    2945
Name: HasCrCard, dtype: int64

In [95]:

HasCrCard_pct=HasCrCard_count/len(dataset.index) *100    #(ratio)
HasCrCard_pct

Out[95]:

1    70.55
0    29.45
Name: HasCrCard, dtype: float64

In [96]:

HasCrCard=pd.concat([HasCrCard_count,round(HasCrCard_pct,2)],axis=1).set_axis(['H_count','H_pct'],axis=1,inplace=False)
HasCrCard

Out[96]:
	H_count 	H_pct
1 	7055 	70.55
0 	2945 	29.45
In [97]:

#Ratio of who has credit cards and who dont have credit cards
labels=["Has_Cedit_Card","Dont_Has_Cedit_Card"]
colors=["#db0494","#12147a"]
plt.pie(HasCrCard_pct,
        colors=colors,
        labels = labels,
        autopct="%0.2f%%")
plt.show()

    As we can see from this Pie chart From all the clients 71% clients has credit cards and approx 29% dont has credit cards

Features and outcome(exited)
```
#### count the values of all Numeric columns as per  group.
```python
dataset.groupby(['Gender']).agg(["count"])


Out[98]:
	RowNumber 	CustomerId 	Surname 	CreditScore 	Geography 	Age 	Tenure 	Balance 	NumOfProducts 	HasCrCard 	IsActiveMember 	EstimatedSalary 	Exited 	Exited_str
	count 	count 	count 	count 	count 	count 	count 	count 	count 	count 	count 	count 	count 	count
Gender 														
Female 	4543 	4543 	4543 	4543 	4543 	4543 	4543 	4543 	4543 	4543 	4543 	4543 	4543 	4543
Male 	5457 	5457 	5457 	5457 	5457 	5457 	5457 	5457 	5457 	5457 	5457 	5457 	5457 	5457
In [99]:

dataset.groupby(['Gender']).agg(["max"])

#MAX values of all Numeric columns as per "gender" group.

Out[99]:
	RowNumber 	CustomerId 	Surname 	CreditScore 	Geography 	Age 	Tenure 	Balance 	NumOfProducts 	HasCrCard 	IsActiveMember 	EstimatedSalary 	Exited 	Exited_str
	max 	max 	max 	max 	max 	max 	max 	max 	max 	max 	max 	max 	max 	max
Gender 														
Female 	10000 	15815690 	Zuyeva 	850 	Spain 	85 	10 	238387.56 	4 	1 	1 	199992.48 	1 	Stayed
Male 	9999 	15815645 	Zuyeva 	850 	Spain 	92 	10 	250898.09 	4 	1 	1 	199953.33 	1 	Stayed
In [100]:

df = dataset.groupby(['Gender','Exited']).agg(["count"])
#count all the vales as per gender and exited croup separately
df

Out[100]:
		RowNumber 	CustomerId 	Surname 	CreditScore 	Geography 	Age 	Tenure 	Balance 	NumOfProducts 	HasCrCard 	IsActiveMember 	EstimatedSalary 	Exited_str
		count 	count 	count 	count 	count 	count 	count 	count 	count 	count 	count 	count 	count
Gender 	Exited 													
Female 	0 	3404 	3404 	3404 	3404 	3404 	3404 	3404 	3404 	3404 	3404 	3404 	3404 	3404
1 	1139 	1139 	1139 	1139 	1139 	1139 	1139 	1139 	1139 	1139 	1139 	1139 	1139
Male 	0 	4559 	4559 	4559 	4559 	4559 	4559 	4559 	4559 	4559 	4559 	4559 	4559 	4559
1 	898 	898 	898 	898 	898 	898 	898 	898 	898 	898 	898 	898 	898
In [101]:

df = dataset.groupby(['Gender','Exited'])['Exited'].agg(["count"])
#count the vlue of column "Exited" as per Gender and Exited group
df

Out[101]:
		count
Gender 	Exited 	
Female 	0 	3404
1 	1139
Male 	0 	4559
1 	898


temp=dataset.groupby(['Gender'])['Exited'].agg(["count"])
temp

Out[103]:
	count
Gender 	
Female 	4543
Male 	5457
```
#### function for grouping data
```pthon
def count_by_group(data, feature, target):
    df = data.groupby([feature, target])[target].agg(['count'])
    temp = data.groupby([feature])[target].agg(['count'])
    df['pct'] = 100 * df.div(temp, level = feature).reset_index()['count'].values
    return df.reset_index()


count_by_group(dataset, feature="Gender", target='Exited')

Out[105]:
	Gender 	Exited 	count 	pct
0 	Female 	0 	3404 	74.928461
1 	Female 	1 	1139 	25.071539
2 	Male 	0 	4559 	83.544072
3 	Male 	1 	898 	16.455928
In [106]:

count_by_group(dataset, feature="Geography", target='Exited')

Out[106]:
	Geography 	Exited 	count 	pct
0 	France 	0 	4204 	83.845233
1 	France 	1 	810 	16.154767
2 	Germany 	0 	1695 	67.556796
3 	Germany 	1 	814 	32.443204
4 	Spain 	0 	2064 	83.326605
5 	Spain 	1 	413 	16.673395
In [107]:
```
##### plotting
```python
!pip install pySankey

Requirement already satisfied: pySankey in /home/ajaygoswami/anaconda3/lib/python3.7/site-packages (0.0.1)

In [108]:

from pySankey.sankey import sankey

In [109]:

colorDict = {
    'Exited':'#f71b1b',
    'Stayed':'grey',
    'France':'#f3f71b',
    'Spain':'#12e23f',
    'Germany':'#f78c1b'
}
sankey(dataset['Geography'], dataset['Exited_str'],
       aspect=20, colorDict=colorDict,
       fontsize=12,figure_name="Geography")
```

```python

HasCrCard_count=dataset["HasCrCard"].value_counts() 
HasCrCard_count

Out[110]:

1    7055
0    2945
Name: HasCrCard, dtype: int64

In [111]:

HasCrCard_pct=HasCrCard_count/len(dataset.index) *100   
HasCrCard_pct

Out[111]:

1    70.55
0    29.45
Name: HasCrCard, dtype: float64
```



#### From the tables and plots above, we can have some insights:

    #####As for gender, women are lower in number than the men, but have a higher rate to close the account.
    #####There is a higher rate of exited clients in Germany(32%, which is about 2x higher), and lower in Spain and France (around 16% each).
    #####On age, customer below 40 and above 65 years old have a tendency to keep their account.
    #####Has or not credit card does not impact on the decision to stay in the bank (both groups has 20% of exited customers)
    #####Non active members tend to discontinue their services with a bank compared with the active clients (27% vs 14%).
   #####The dataset has 96% of clients with 1 or 2 product, and customers with 1 product only have a higher rate to close the account than those with 2 products (around 3x higher).
    #####Estimated Salary does not seem to affect the churn rate

## 2. Predictive Models
######Separating Dataset into X and y subsets
###2.1 One-Hot encoding Categorical Attributes

    #####One-Hot encoding Categorical Attributes::::::::::::::::::It refers to splitting the column which contains numerical categorical data to many columns depending on the number of categories present in that column. Each column contains “0” or “1” corresponding to which column it has been placed.

    ######One-Hot encoding -- process by which categorical variables are converted into a form that could be provided to ML algorithms.

```python
list_cat=['Geography','Gender']
dataset=pd.get_dummies(dataset,columns=list_cat,prefix=list_cat)
#we have converted the both categorical columns into dummy variables for processing
dataset.head()

Out[125]:
	RowNumber 	CustomerId 	Surname 	CreditScore 	Age 	Tenure 	Balance 	NumOfProducts 	HasCrCard 	IsActiveMember 	... 	Exited 	Exited_str 	HasCrCard_str 	IsActiveMember_str 	NumOfProducts_str 	Geography_France 	Geography_Germany 	Geography_Spain 	Gender_Female 	Gender_Male
0 	1 	15634602 	Hargrave 	619 	42 	2 	0.00 	1 	1 	1 	... 	1 	Exited 	Has Credit Card 	IsActiveMember 	1 	1 	0 	0 	1 	0
1 	2 	15647311 	Hill 	608 	41 	1 	83807.86 	1 	0 	1 	... 	0 	Stayed 	dont Has Credit Card 	IsActiveMember 	1 	0 	0 	1 	1 	0
2 	3 	15619304 	Onio 	502 	42 	8 	159660.80 	3 	1 	0 	... 	1 	Exited 	Has Credit Card 	Is_not_ActiveMember 	3 	1 	0 	0 	1 	0
3 	4 	15701354 	Boni 	699 	39 	1 	0.00 	2 	0 	0 	... 	0 	Stayed 	dont Has Credit Card 	Is_not_ActiveMember 	2 	1 	0 	0 	1 	0
4 	5 	15737888 	Mitchell 	850 	43 	2 	125510.82 	1 	1 	1 	... 	0 	Stayed 	Has Credit Card 	IsActiveMember 	1 	0 	0 	1 	1 	0

5 rows × 21 columns
In [126]:
```

### dropping the unnecessory columns 
```python
dataset = dataset.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited_str','HasCrCard_str', 'IsActiveMember_str','NumOfProducts_str'], axis = 1)

In [127]:

dataset.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 14 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   CreditScore        10000 non-null  int64  
 1   Age                10000 non-null  int64  
 2   Tenure             10000 non-null  int64  
 3   Balance            10000 non-null  float64
 4   NumOfProducts      10000 non-null  int64  
 5   HasCrCard          10000 non-null  int64  
 6   IsActiveMember     10000 non-null  int64  
 7   EstimatedSalary    10000 non-null  float64
 8   Exited             10000 non-null  int64  
 9   Geography_France   10000 non-null  uint8  
 10  Geography_Germany  10000 non-null  uint8  
 11  Geography_Spain    10000 non-null  uint8  
 12  Gender_Female      10000 non-null  uint8  
 13  Gender_Male        10000 non-null  uint8  
dtypes: float64(2), int64(7), uint8(5)
memory usage: 752.1 KB
```


#### creating input variables (dropping output column)
```python
features = list(dataset.drop('Exited', axis = 1))

In [129]:

#this is our output columns
target = 'Exited'
```

### 2.2 Splitting the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split

In [134]:

train, test = train_test_split(dataset, test_size = 0.2, random_state = 1)

#here we are deviding the dataset into 80-20 ratio for training and testing

print('Number of clients in the dataset: {}'.format(len(dataset)))
print('Number of clients in the train set: {}'.format(len(train)))
print('Number of clients in the test set: {}'.format(len(test)))

Number of clients in the dataset: 10000
Number of clients in the train set: 8000
Number of clients in the test set: 2000

```
#### training and testing the model

```python
exited_train = len(train[train['Exited'] == 1]['Exited'])
exited_train_perc = round(exited_train/len(train)*100,1)

exited_test = len(test[test['Exited'] == 1]['Exited'])
exited_test_perc = round(exited_test/len(test)*100,1)

print('Complete Train set - Number of clients that have exited the program: {} ({}%)'.format(exited_train, exited_train_perc))
print('Test set - Number of clients that haven\'t exited the program: {} ({}%)'.format(exited_test, exited_test_perc))

Complete Train set - Number of clients that have exited the program: 1622 (20.3%)
Test set - Number of clients that haven't exited the program: 415 (20.8%)
```

#### 2.3 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
#Standardize features by removing the mean and scaling to unit variance

    #Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

In [140]:

sc = StandardScaler()

# fit on training set
train[features] = sc.fit_transform(train[features])

# only transform on test set
test[features] = sc.transform(test[features])


train.head(3)

Out[142]:
	CreditScore 	Age 	Tenure 	Balance 	NumOfProducts 	HasCrCard 	IsActiveMember 	EstimatedSalary 	Exited 	Geography_France 	Geography_Germany 	Geography_Spain 	Gender_Female 	Gender_Male
2694 	-0.230820 	-0.944500 	-0.701742 	0.588173 	0.802257 	-1.553374 	0.977259 	0.427394 	0 	-0.998501 	1.714901 	-0.572731 	-0.915091 	0.915091
5140 	-0.251509 	-0.944500 	-0.355203 	0.469849 	0.802257 	-1.553374 	-1.023271 	-1.025487 	0 	1.001501 	-0.583124 	-0.572731 	1.092788 	-1.092788
2568 	-0.396330 	0.774987 	0.337876 	0.858788 	-0.911510 	0.643760 	0.977259 	-0.944798 	1 	-0.998501 	1.714901 	-0.572731 	1.092788 	-1.092788
In [143]:

test.head(3)

Out[143]:
	CreditScore 	Age 	Tenure 	Balance 	NumOfProducts 	HasCrCard 	IsActiveMember 	EstimatedSalary 	Exited 	Geography_France 	Geography_Germany 	Geography_Spain 	Gender_Female 	Gender_Male
9953 	-1.037681 	0.774987 	-1.048281 	-1.225992 	0.802257 	0.643760 	0.977259 	-0.053606 	0 	1.001501 	-0.583124 	-0.572731 	-0.915091 	0.915091
3850 	0.307087 	-0.466865 	-0.701742 	1.071524 	-0.911510 	0.643760 	-1.023271 	-0.583927 	0 	1.001501 	-0.583124 	-0.572731 	-0.915091 	0.915091
4962 	-1.234224 	0.297352 	-1.048281 	-1.225992 	0.802257 	-1.553374 	0.977259 	-0.166853 	0 	1.001501 	-0.583124 	-0.572731 	1.092788 	-1.092788
```

#### 2.4 Complete Trainning Set
#### 2.4.1 Logistic Regression (Sklearn)
```python

from sklearn.linear_model import LogisticRegression

In [180]:

LR_Classifier = LogisticRegression(penalty = 'l2').fit(train[features], train[target])
LR_Classifier = LR_Classifier.fit(train[features], train[target])
LR_pred = LR_Classifier.predict(test[features])
```

#### calculating the accuracy through confusion matrix
```python
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
LR_acc = accuracy_score(test[target], LR_pred)
print("Logistic Regression accuracy is",LR_acc)
cm = confusion_matrix(test[target], LR_pred)

plot_Confusion_matrix(cm,['Exited','Not Exited'],'Blues',"Logistic Regression", LR_acc)

Logistic Regression accuracy is 0.8125

In [190]:

LR_Cla_Rep = classification_report(test[target], LR_pred)
print(LR_Cla_Rep)

              precision    recall  f1-score   support

           0       0.83      0.97      0.89      1585
           1       0.64      0.22      0.33       415

    accuracy                           0.81      2000
   macro avg       0.73      0.59      0.61      2000
weighted avg       0.79      0.81      0.77      2000

    1- Accuracy of our Machine Learning model is 81%
```
   

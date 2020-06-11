# Bank Customer Churn Analysis

Churner is generally defined as a customer who stops using a product or service for a given period of time.

This notebook is to do the data analysis and predictions on the churn.csv file.

The first step in the Data Preprocessing is to import the libraries, load the data and do some Exploratory Data Analysis (EDA).


#### STEP1 - Import all important Libraries and dataset

import pandas as pd    #for EDA
import numpy as np     #for numerical operation if required
from pySankey.sankey import sankey   #for sankey plot
import matplotlib.pyplot as plt      #for visualization

import seaborn as sns
%matplotlib inline
dataset=pd.read_csv(r'/home/ajaygoswami/Documents/Bobby/DATA SET/churn.csv')
#importing dataset and put it in a variable called dataset
dataset.head()

Accuracy_Report = pd.DataFrame(columns=["Models","Accuracy"])
models_lis, acc_lis = [], []
def Submit_Score(lis1,lis2):
    models_lis.append(lis1)
    acc_lis.append(lis2)
    return
    
def Show_Model_Score():
    temp_df = pd.DataFrame({'Models': models_lis, 'Accuracy': acc_lis})
    return temp_df

# Function for plotting the confusion matrix

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

## 1. Exploratory Data Analysis

dataset.info()  #for dataset information

#dataset.head()
dataset.shape  #understanding the shape of our dataset 

dataset.isna().sum()  #finding all the null values as per each column

dataset.describe() #for descriptive information(statical)

round(dataset.describe(),2)  
#'''rounding off all the decimal values up-to 2 place'''

dataset.head()

exit=dataset[dataset["Exited"]==1]
exit

#'''exit = ONLY ROWS WHERE Exited == 1'''

notexit=dataset[dataset["Exited"] ==0]
notexit
#'''notexit = ONLY ROWS WHERE Exited == 0'''

#notexit["Surname"]

#return all the Surname values of notexit

notexit[["Surname","CreditScore","Balance"]]
#Return given columns of notexit

##### calculating Exited and not_exited client 

exited=len(dataset[dataset["Exited"] ==0])
notexited=len(dataset[dataset["Exited"] ==1])
print("length of exited persons :- {}".format(exited))
#return length of exited persons
print("length of exited persons :- {}".format(notexited))
#return length of notexited persons

exited_perc=round(exited/len(dataset)*100,2)
notexited_perc=round(notexited/len(dataset)*100,2)

#finding the percentage of exited and notexited persons from the length of whole dataset 
#and rounded of its decimal value up to 2 place

print("Exited :- {} %".format(exited_perc))
print("Not-exited :- {} %".format(notexited_perc))

#plotting Exited and not_exited persons as per ratio
labels=["Exited","Not_exited"]
data = [exited_perc,notexited_perc]
colors=["#95f542","#02f2ee"]
plt.pie(data, 
        labels = labels,
        colors=colors,
        autopct="%0.2f%%")            # after point. numbers
plt.show()

###### So, around of 20% of the clients exited the bank, while around 80% stayed. As the goal here is to identify which of the customers are at higher risk to discontinue their services with the bank, we are dealing with a classification problem.

important point to take into consideration here is that we are dealing with an imbalanced dataset.


country = list(dataset["Geography"].unique()) 
gender = list(dataset["Gender"].unique())

#finding the unique values of country and gender wih ".unique()/.nunique()" methods

print(country)
print(gender)

dataset["Exited_str"]=dataset["Exited"]
dataset["Exited_str"]=dataset["Exited_str"].map({1:'Exited',0:"Stayed"})
#creating a new column with the help of "Exited" column where numeric values(0,1) converted as-
#..................string values(stayed,exited))repectively

dataset["Exited_str"]

sns.catplot("Exited_str",data=dataset,kind="count",hue="NumOfProducts")

Here we can see that those clients who consist one product exited the most and other side who has2 product has mazority in stayed clients.

'''We can also convert other modified column with the help of given columns'''
#exampole:-
#dataset["Geography_str"]=dataset["Geography"]
#dataset["Geography_str"]=dataset["Geography_str"].map({'Spain':'Spa','France':"Fra",'Germany':'Ger'})
#dataset["Geography_str"]

gender_count=dataset["Gender"].value_counts() 
#return different categorical values of gender column
gender_count

gender_pct=gender_count/len(dataset.index) *100    #(ratio/percentage)
gender_pct

#gender=pd.concat([gender_count,round(gender_pct,2)],axis=1)   #to bring it in rows
gender=pd.concat([gender_count,gender_pct],axis=1)
#concate the count and percentage different gender wise
gender




gender=pd.concat([gender_count,round(gender_pct,2)],axis=1).set_axis(['count','pct'],axis=1,inplace=False)
#changing the axis name as per our understanding
gender

geography_count=dataset["Geography"].value_counts() #return different categorical values
geography_count

geography_pct=geography_count/len(dataset.index) *100    #(ratio)
geography_pct

geography=pd.concat([geography_count,round(geography_pct,2)],axis=1).set_axis(['G_count','G_pct'],axis=1,inplace=False)
geography

##plotting MALE and FEMALE  ratio
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

Now, let's just check the relationship between the features and the outcome ('Exited').


salary_min=dataset[dataset["EstimatedSalary"]<30000]
salary_min

salary_min_count=len(dataset[dataset["EstimatedSalary"]<30000])
salary_max_count=len(dataset[dataset["EstimatedSalary"]>30000])
print("persons sallery below 30,000 :- {}".format(salary_min_count))
print("persons sallery above 30,000 :- {}".format(salary_max_count))


salary_min_pct=salary_min_count/len(dataset.index)*100
salary_max_pct=salary_max_count/len(dataset.index)*100
print("(salary < 30,000) % :- {}".format(salary_min_pct))
print("(salary > 30,000) % :- {}".format(salary_max_pct))

labels=["salary under 30000","salary above 30000"]
data=[salary_min_pct,salary_max_pct]
colors=["skyblue",'lightgreen']
plt.pie(data,labels=labels,colors=colors,autopct="%0.2f%%")
plt.show()

- Here we can see the salary Percentage of clients with salary more than 30,000 are 85% and below 30,000 are 15%


HasCrCard_count=dataset["HasCrCard"].value_counts() 
#return different categorical values of Hascreditcard/Dont_havecreditcards
HasCrCard_count

HasCrCard_pct=HasCrCard_count/len(dataset.index) *100    #(ratio)
HasCrCard_pct

HasCrCard=pd.concat([HasCrCard_count,round(HasCrCard_pct,2)],axis=1).set_axis(['H_count','H_pct'],axis=1,inplace=False)
HasCrCard

#Ratio of who has credit cards and who dont have credit cards
labels=["Has_Cedit_Card","Dont_Has_Cedit_Card"]
colors=["#db0494","#12147a"]
plt.pie(HasCrCard_pct,
        colors=colors,
        labels = labels,
        autopct="%0.2f%%")
plt.show()

- As we can see from this Pie chart From all the clients 71% clients has credit cards  and approx 29% dont has credit cards

###### Features and outcome(exited)

dataset.groupby(['Gender']).agg(["count"])
#count the values of all Numeric columns as per "gender" group.




dataset.groupby(['Gender']).agg(["max"])

#MAX values of all Numeric columns as per "gender" group.

df = dataset.groupby(['Gender','Exited']).agg(["count"])
#count all the vales as per gender and exited croup separately
df

df = dataset.groupby(['Gender','Exited'])['Exited'].agg(["count"])
#count the vlue of column "Exited" as per Gender and Exited group
df

sns.catplot('Exited',data=dataset,kind="count",hue="Gender")

- From this catplot we can understand that from Exitd Clients the number of female are little bit more than males 

temp=dataset.groupby(['Gender'])['Exited'].agg(["count"])
temp

def count_by_group(data, feature, target):
    df = data.groupby([feature, target])[target].agg(['count'])
    temp = data.groupby([feature])[target].agg(['count'])
    df['pct'] = 100 * df.div(temp, level = feature).reset_index()['count'].values
    return df.reset_index()



count_by_group(dataset, feature="Gender", target='Exited')

count_by_group(dataset, feature="Geography", target='Exited')

!pip install pySankey

from pySankey.sankey import sankey

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

HasCrCard_count=dataset["HasCrCard"].value_counts() 
HasCrCard_count

HasCrCard_pct=HasCrCard_count/len(dataset.index) *100   
HasCrCard_pct

HasCrCard=pd.concat([HasCrCard_count,HasCrCard_pct],axis=1).set_axis(['H_count','H_pct'],axis=1,inplace=False)
HasCrCard

count_by_group(dataset, feature = "HasCrCard",target = "Exited")

dataset["HasCrCard_str"]=dataset['HasCrCard'].map({1:"Has Credit Card", 0:"dont Has Credit Card"})
colorDict = {
    'Exited':'#f71b1b',
    'Stayed':'grey',
    "Has Credit Card":'#f3f71b',
    "dont Has Credit Card":'#71f5ec'}
  
sankey(dataset['HasCrCard_str'], dataset['Exited_str'],
       aspect=20, colorDict=colorDict,
       fontsize=12,figure_name="HasCrCard_str")

IsActiveMember_count=dataset["IsActiveMember"].value_counts() 
IsActiveMember_count

IsActiveMember_pct=IsActiveMember_count/len(dataset.index) *100   
IsActiveMember_pct

IsActiveMember=pd.concat([IsActiveMember_count,IsActiveMember_pct],axis=1).set_axis(['I_count','I_pct'],axis=1,inplace=False)
IsActiveMember

count_by_group(dataset, feature = "IsActiveMember",target = "Exited")

dataset["IsActiveMember_str"]=dataset['IsActiveMember'].map({1:"IsActiveMember", 0:"Is_not_ActiveMember"})


colorDict = {
    'Exited':'#f71b1b',
    'Stayed':'grey',
    "IsActiveMember":'#4ddbc4',
    "Is_not_ActiveMember":'#75b84b',
    
}
sankey(dataset['IsActiveMember_str'], dataset['Exited_str'],
       aspect=20, colorDict=colorDict,
       fontsize=12,figure_name="IsActiveMember_str")


NumOfProducts_count = dataset['NumOfProducts'].value_counts()
NumOfProducts_pct= NumOfProducts_count / len(dataset.index)

NumOfProducts = pd.concat([NumOfProducts_count, round(NumOfProducts_pct,2)], axis=1)\
        .set_axis(['count', 'pct'], axis=1, inplace=False)
NumOfProducts


# STEP 2
count_by_group(dataset, feature = 'NumOfProducts', target = 'Exited')



# STEP 3
dataset['NumOfProducts_str'] = dataset['NumOfProducts'].map({1: '1', 2: '2', 3: '3', 4: '4'})

# STEP 4

colorDict = {
    'Exited':'#f71b1b',
    'Stayed':'grey',
    '1':'#f3f71b',
    '2':'#12e23f',
    '3':'#f78c1b',
    '4':'#8E388E'
}
sankey(
    dataset['NumOfProducts_str'], dataset['Exited_str'], aspect=20, colorDict=colorDict,
    fontsize=12, figure_name="NumOfProducts")

dataset[(dataset["Exited"]==0)]

figure = plt.figure(figsize=(15,8))
plt.hist([
        dataset[(dataset.Exited==0)]['Age'],
        dataset[(dataset.Exited==1)]['Age']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
plt.xlabel('Age (years)')
plt.ylabel('Number of customers')
plt.legend()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (15,15))
fig.subplots_adjust(left=0.2, wspace=0.6)
ax0, ax1, ax2, ax3 = axes.flatten()

ax0.hist([
        dataset[(dataset.Exited==0)]['CreditScore'],
        dataset[(dataset.Exited==1)]['CreditScore']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax0.legend()
ax0.set_title('Credit Score')

ax1.hist([
        dataset[(dataset.Exited==0)]['Tenure'],
        dataset[(dataset.Exited==1)]['Tenure']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax1.legend()
ax1.set_title('Tenure')

ax2.hist([
        dataset[(dataset.Exited==0)]['Balance'],
        dataset[(dataset.Exited==1)]['Balance']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax2.legend()
ax2.set_title('Balance')

ax3.hist([
        dataset[(dataset.Exited==0)]['EstimatedSalary'],
        dataset[(dataset.Exited==1)]['EstimatedSalary']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax3.legend()
ax3.set_title('Estimated Salary')

fig.tight_layout()
plt.show()

From the tables and plots above, we can have some insights:

1. As for gender, `women are lower in number` than the men, but have a `higher rate to close` the account.
2. There is a `higher rate of exited clients in Germany `(32%, which is about 2x higher), and `lower in Spain` and France (around 16% each).
3. On age, `customer below 40 and above 65` years old have a `tendency to keep their account`.
4. Has or not `credit card does not impact on the decision` to stay in the bank (both groups has 20% of exited customers)
5. Non active members tend to discontinue their services with a bank compared with the active clients (27% vs 14%). 
6. The dataset has 96% of clients  with 1 or 2 product, and `customers with 1 product only have a higher rate to close the account` than those with 2 products (around 3x higher).
7. Estimated `Salary does not seem to affect` the churn rate

# 2. Predictive Models

#### Separating Dataset into X and y subsets

#### 2.1 One-Hot encoding Categorical Attributes

- One-Hot encoding Categorical Attributes::::::::::::::::::It refers to splitting the column which contains numerical categorical data to many columns depending on the number of categories present in that column. Each column contains “0” or “1” corresponding to which column it has been placed.

- One-Hot encoding -- process by which categorical variables are converted into a form that could be provided to ML algorithms.

list_cat=['Geography','Gender']
dataset=pd.get_dummies(dataset,columns=list_cat,prefix=list_cat)
#we have converted the both categorical columns into dummy variables for processing
dataset.head()

#dropping the unnecessory columns 
dataset = dataset.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited_str','HasCrCard_str', 'IsActiveMember_str','NumOfProducts_str'], axis = 1)

dataset.info()

#creating input variables (dropping output column)
features = list(dataset.drop('Exited', axis = 1))


#this is our output columns
target = 'Exited'

### 2.2 Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

train, test = train_test_split(dataset, test_size = 0.2, random_state = 1)

#here we are deviding the dataset into 80-20 ratio for training and testing

print('Number of clients in the dataset: {}'.format(len(dataset)))
print('Number of clients in the train set: {}'.format(len(train)))
print('Number of clients in the test set: {}'.format(len(test)))

exited_train = len(train[train['Exited'] == 1]['Exited'])
exited_train_perc = round(exited_train/len(train)*100,1)

exited_test = len(test[test['Exited'] == 1]['Exited'])
exited_test_perc = round(exited_test/len(test)*100,1)

print('Complete Train set - Number of clients that have exited the program: {} ({}%)'.format(exited_train, exited_train_perc))
print('Test set - Number of clients that haven\'t exited the program: {} ({}%)'.format(exited_test, exited_test_perc))

#### 2.3 Feature Scaling

from sklearn.preprocessing import StandardScaler
#Standardize features by removing the mean and scaling to unit variance

- Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

sc = StandardScaler()

# fit on training set
train[features] = sc.fit_transform(train[features])

# only transform on test set
test[features] = sc.transform(test[features])

train.head(3)

test.head(3)

### 2.4 Complete Trainning Set

#### 2.4.1 Logistic Regression (Sklearn)

from sklearn.linear_model import LogisticRegression

LR_Classifier = LogisticRegression(penalty = 'l2').fit(train[features], train[target])
LR_Classifier = LR_Classifier.fit(train[features], train[target])
LR_pred = LR_Classifier.predict(test[features])

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
LR_acc = accuracy_score(test[target], LR_pred)
print("Logistic Regression accuracy is",LR_acc)
cm = confusion_matrix(test[target], LR_pred)

plot_Confusion_matrix(cm,['Exited','Not Exited'],'Blues',"Logistic Regression", LR_acc)

LR_Cla_Rep = classification_report(test[target], LR_pred)
print(LR_Cla_Rep)

- 1- Accuracy of our Machine Learning model is 81% 

- 2- precision of exited - notexited predictive result is respectively 83% and 64%

#(also called positive predictive value) is the fraction of relevant instances among the retrieved instances

- recall of exited - notexited predictive result is respectively 97% and 33%

#(also known as sensitivity) is the fraction of the total amount of relevant instances that were actually retrieved. Both precision and recall are therefore based on an understanding and measure of relevance. 

- 3- F1 score  of exited - notexited predictive result is respectively 89% and 33%

#F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution.

- 4- macro avg of precision ,recall , f1-score is respectively 73%,59%,61%

#Macro-average recall = (R1+R2)/2 = (80+84.75)/2 = 82.25. The Macro-average F-Score will be simply the harmonic mean of these two figures. Suitability. Macro-average method can be used when you want to know how the system performs overall across the sets of data


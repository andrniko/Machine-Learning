'''KNN is used to group data with simmilar atributes in higher dimenstion.

We decide the number of K, which means we find the K closest neighbors
to our new data point and from there deduce where it "belongs".

We measure the eucledian distance of our new data point from it's neighbors.

It runs more efficient on smaller datasets.


It classifies a new data point according to it's eucladian distance
from other points.

In this program I will make my own model and compare it to sklearns.

-----------------------------------------------------------

I will be using the Breast Cancer Wisconsin (Original) Data Set from UCI's ML repository
which can be found on https://archive.ics.uci.edu/ml/datasets.php

The set contains 699 points as of July of 1992.

It has 11 columns
1. Sample code number: id number
2. Clump Thickness: 1 - 10
3. Uniformity of Cell Size: 1 - 10
4. Uniformity of Cell Shape: 1 - 10
5. Marginal Adhesion: 1 - 10
6. Single Epithelial Cell Size: 1 - 10
7. Bare Nuclei: 1 - 10
8. Bland Chromatin: 1 - 10
9. Normal Nucleoli: 1 - 10
10. Mitoses: 1 - 10
11. Class: (2 for benign, 4 for malignant)

I've edited the original dataset so that it contains a rown with collumn names
id,clump_thickness,unif_cell_size,unif_cell_shape,marg_adhesion,single_epith_cell_size,bare_nuclei,bland_chrom,norm_nucleolu,mitoses,class



'''


import numpy as np
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import random


#Defining the function which gives us the eucledian distance and calculates the KNN

def k_nearest_neighbros(data,predict,k=3):
  
  distances=[]
  for group in data:
    for features in data[group]:
     eucledian_distance=np.linalg.norm(np.array(features)-np.array(predict))
     distances.append([eucledian_distance,group])
  
  votes=[i[1] for i in sorted(distances)[:k]]
  vote_result=Counter(votes).most_common(1)[0][0]
  confidence=Counter(votes).most_common(1)[0][1]/k
  

  #print(vote_result,confidence)
  return vote_result,confidence


#Using the Breast cancer database
df=pd.read_csv('breast-cancer-wisconsin.data')


#Replacing the NaN values(marked with ?) with -99999 sow they are recognised as outliers, instead of removingt them
df.replace('?',-99999,inplace=True)

#Dropping the id column for it may cause serious problems in the future
df.drop(['id'],1,inplace=True)

#Setting all values to float
full_data=df.astype(float).values.tolist() #To remove some of the values being strings


#Shuffling the data
random.shuffle(full_data)

#Splitting train and test sets by hand
test_size=0.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]# up to last 20%
test_data=full_data[:int(test_size*len(full_data)):]#the last 20%

for i in train_data:
  train_set[i[-1]].append(i[:-1])

for i in test_data:
  test_set[i[-1]].append(i[:-1])


#Setting up accuracy calculation
correct=0
total=0

for group in test_set:
  for data in test_set[group]:
    vote,confidence=k_nearest_neighbros(train_set,data,k=5)
    if group==vote:
      correct+=1
    total+=1


print('"Homemade" KNN accuracy: ',correct/total)



#Using sklearn's KNN model


#splitting the original dataset:

X=np.array(df.drop(['class'],1)) #Removing the class, which tells us whether it's benign or malignant
y=np.array(df['class']) #setting y just to class

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

clf=neighbors.KNeighborsClassifier(n_jobs=-1)

clf.fit(X_train,y_train)

accuracy1=clf.score(X_test,y_test)

print('Sklearns KNN accuracy: ',accuracy1)
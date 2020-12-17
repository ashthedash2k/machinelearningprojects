import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv('/Users/ashleyczumak/MachineLearningWork/data/breastcancer/data 2.csv')

sns.countplot(x='diagnosis', data=data)
plt.title('Breast Cancer Diagnosis')
plt.show()

amount_of_malignant_and_benign = data.diagnosis.value_counts()
print(amount_of_malignant_and_benign)

data.drop(["id","Unnamed: 32"], axis =1, inplace = True)
data = data.rename(columns = {"diagnosis" : "target"})


#map strinng value to numerical value

data['target'] = [1 if i.strip() == 'M' else 0 for i in data['target']]
#print(data['target'][234])

print(data.keys())

#visualize correlation
c_matrix = data.corr()
threshold = 0.7
filter = np.abs(c_matrix['target']) > threshold
corr_features = c_matrix.columns[filter].tolist()
plt.figure(figsize =(15,15))
sns.clustermap(data[corr_features].corr(),annot=True, fmt=".2f")
plt.title("Correlation Between Features w Corr Threshold 0.70")
plt.show()

#splitting dataset
x = data.iloc[:,2:31].values
y = data.iloc[:,1].values #values converts to numpy array

#print(type(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#knn
knn = KNeighborsRegressor(n_neighbors=6)
knn.fit(x_train, y_train)
print(f'Score is: {knn.score(x_test, y_test)}')

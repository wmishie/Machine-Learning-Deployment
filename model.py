
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

#Loading the data set
df = pd.read_csv('iris.csv')
print(df.head())

#Select independent and dependend variables
x = df[["Sepal_Length", "Sepal_Width", "Petal_Width"]]
y = df["Class"]

# Split the data set to train and test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=50)

#Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#Instantiate the model
classifier = RandomForestClassifier()

#Fit the model
classifier.fit(x_train, y_train)

#Make Pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))
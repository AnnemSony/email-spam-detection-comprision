# email-spam-detection-comprision
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
//////
//path="/content/drive/MyDrive/spam.csv"
spam = pd.read_csv(path) 
spam.describe
////////
spam=pd.read_csv('/content/spam.csv', encoding="ISO-8859-1")
spam.describe
///////
from google.colab import drive
drive.mount('/content/drive')
/////////////
from google.colab import files
upoad=files.upload()
///////////
z = spam['v1']
y = spam["v2"]
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)
////////
cv = CountVectorizer()
features = cv.fit_transform(z_train)
/////////////
model = svm.SVC()
model.fit(features,y_train)
//////////
features_test = cv.transform(z_test)
print("Accuracy: {}".format(model.score(features_test,y_test)))
///////
model = svm.SVC()
model.fit(features,y_train)

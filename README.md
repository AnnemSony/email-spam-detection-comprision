# email-spam-detection-comprision
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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
/////
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb = GaussianNB()
//////
#rfc = DecisionTreeClassifier(max_depth=5)
lrc.fit(X_train,y_train)
y_pred1 = lrc.predict(X_test)
print(accuracy_score(y_test,y_pred1))
////
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)

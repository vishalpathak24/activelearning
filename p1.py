from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm



twenty_train = fetch_20newsgroups(subset='train',categories=['comp.graphics', 'sci.med'], shuffle=True, random_state=42)
count_vect = CountVectorizer()
data_feature = count_vect.fit_transform(twenty_train.data)
data_tf_feature = TfidfTransformer().fit_transform(data_feature)


#creating SVM

class_SVM = svm.SVC(kernel='linear')
class_SVM.fit(data_tf_feature,twenty_train.target)

i=0
#print data_tf_feature
for data in data_tf_feature:
	#print data.toarray()
	print "**********"
	score = class_SVM.coef_.dot(data.toarray()[0])+class_SVM.intercept_
	print "score ="+str(score)+" predicted class="+str(class_SVM.predict(data))+" actual class="+str(twenty_train.target[i])
	i=i+1
	if i >50:
		break

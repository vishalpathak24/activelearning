from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn import metrics



twenty_train = fetch_20newsgroups(subset='train',categories=['comp.graphics', 'sci.med'], shuffle=True, random_state=42)
count_vect = CountVectorizer()
tf_idf_converter = TfidfTransformer()

#transforming training data
data_feature = count_vect.fit_transform(twenty_train.data)

data_tf_feature = tf_idf_converter.fit_transform(data_feature)


#creating SVM

class_SVM = svm.SVC(kernel='linear')



#Training the Classifier

class_SVM.fit(data_tf_feature,twenty_train.target)

i=0
#Score in Training Set
error = 0
for data in data_tf_feature:
	#print data.toarray()
	#print "**********"
	score = class_SVM.coef_.dot(data.toarray()[0])+class_SVM.intercept_
	predicted = class_SVM.predict(data)
	#
	if predicted != twenty_train.target[i]:
		error=error+1
		print "score ="+str(score)+" predicted class="+str(class_SVM.predict(data))+" actual class="+str(twenty_train.target[i])

	i=i+1
	#if i >50:
	#	break



print "Training error encontered is",error

print "Printing Training metrics"
predicted = class_SVM.predict(data_tf_feature)
print(metrics.classification_report(twenty_train.target, predicted,target_names=twenty_train.target_names))



#Fetching testing data
twenty_test = fetch_20newsgroups(subset='test',categories=['comp.graphics', 'sci.med'], shuffle=True, random_state=42)

data_feature = count_vect.transform(twenty_test.data)
data_tf_feature = tf_idf_converter.transform(data_feature)


print "Printing Testing metrics"
predicted = class_SVM.predict(data_tf_feature)
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))

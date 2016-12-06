from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn import metrics
import sys
import numpy as np

NTRAIN = 14
NROUND = 1


twenty_train = fetch_20newsgroups(subset='train',categories=['comp.graphics', 'sci.med'], shuffle=True, random_state=42)
count_vect = CountVectorizer()
tf_idf_converter = TfidfTransformer()

#Reducing training set
pool_data = twenty_train.data[NTRAIN:]
pool_target = twenty_train.target[NTRAIN:]
twenty_train.data=twenty_train.data[:NTRAIN]
twenty_train.target=twenty_train.target[:NTRAIN]

#twenty_train.data.

#transforming training data


#creating SVM

while NROUND !=0:
	data_feature = count_vect.fit_transform(twenty_train.data)
	data_tf_feature = tf_idf_converter.fit_transform(data_feature)

	class_SVM = svm.SVC(kernel='linear')
	#Training the Classifier
	class_SVM.fit(data_tf_feature,twenty_train.target)
	#Score in Training Set
	error = 0
	min_score = sys.maxint 
	min_index = -1
	i=0
	pool_data_transform = tf_idf_converter.transform(count_vect.transform(pool_data))
	for data in pool_data_transform:
		#data = tf_idf_converter.transform(count_vect.transform(data))
		score = class_SVM.coef_.dot(data.toarray()[0])+class_SVM.intercept_		
		score = abs(score)
		#predicted = class_SVM.predict(data)
		if score < min_score:
			min_score=score
			min_index=i
		i=i+1

	if min_index == -1:
		print "unable to find min_index"
		exit(0)
	twenty_train.data.append(pool_data[min_index])
	twenty_train.target=np.append(twenty_train.target,np.array([pool_target[min_index]]))
	del pool_data[min_index]
	pool_target=np.delete(pool_target,min_index)
	print "ROUND FINISHED..with min score =",min_score
	NROUND=NROUND-1
	



	#print "Training error encontered is",error

#print "Printing Training metrics"
#predicted = class_SVM.predict(data_tf_feature)
#print(metrics.classification_report(twenty_train.target, predicted,target_names=twenty_train.target_names))



#Fetching testing data
twenty_test = fetch_20newsgroups(subset='test',categories=['comp.graphics', 'sci.med'], shuffle=True, random_state=42)

data_feature = count_vect.transform(twenty_test.data)
data_tf_feature = tf_idf_converter.transform(data_feature)


print "Printing Testing metrics"
predicted = class_SVM.predict(data_tf_feature)
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))

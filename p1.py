from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



twenty_train = fetch_20newsgroups(subset='train',categories=['comp.graphics', 'sci.med'], shuffle=True, random_state=42)
count_vect = CountVectorizer()
data_feature = count_vect.fit_transform(twenty_train.data)
data_tf_feature = TfidfTransformer().fit(data_feature)


import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing

from collections import Counter

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsRegressor

dating = pd.read_csv("profiles.csv")


print(dating.columns)

print(dating['age'].value_counts())

print(dating['diet'].value_counts())

print(dating['drugs'].value_counts())

print(dating['sex'].value_counts())

print(dating['income'].unique())

print(dating['income'].value_counts())



drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}

smoke_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}

drug_mapping = {"never": 0, "sometimes": 1, "often": 2}

sex_mapping = {"m": 0, "f": 1}

dating["drinks_code"] = dating.drinks.map(drink_mapping)

dating["drinks_code"].fillna(-1, inplace=True)

dating["smokes_code"] = dating.smokes.map(smoke_mapping)

dating["smokes_code"].fillna(-1, inplace=True)

dating["drugs_code"] = dating.drugs.map(drug_mapping)

dating["drugs_code"].fillna(-1, inplace=True)

dating["sex_code"] = dating.sex.map(sex_mapping)

dating["sex_code"].fillna(-1, inplace=True)

dating.dropna(inplace=True)

dating['essay0_len'] = dating['essay0'].apply(lambda x: len(x))

dating['essay1_len'] = dating['essay1'].apply(lambda x: len(x))

dating['essay2_len'] = dating['essay2'].apply(lambda x: len(x))

dating['essay3_len'] = dating['essay3'].apply(lambda x: len(x))

dating['essay4_len'] = dating['essay4'].apply(lambda x: len(x))

dating['essay5_len'] = dating['essay5'].apply(lambda x: len(x))

dating['essay6_len'] = dating['essay6'].apply(lambda x: len(x))

dating['essay7_len'] = dating['essay7'].apply(lambda x: len(x))

dating['essay8_len'] = dating['essay8'].apply(lambda x: len(x))

dating['essay9_len'] = dating['essay9'].apply(lambda x: len(x))

essay_lengths = ['essay0_len', 'essay1_len', 'essay2_len','essay3_len', 'essay4_len','essay5_len', 'essay6_len','essay7_len','essay8_len', 'essay9_len']

dating['avg_essay_len'] = dating[essay_lengths].mean(axis=1)

print(dating['avg_essay_len'].unique())
def avg_word_len(essay):
    # return avg length of words within string essay
    word_list = essay.split()
    word_lengths = [len(x) for x in word_list]
    return sum(word_lengths)/len(word_lengths)

dating['essay0_word_len'] = dating['essay0'].apply(lambda x: avg_word_len(x))
dating['essay1_word_len'] = dating['essay1'].apply(lambda x: avg_word_len(x))
dating['essay2_word_len'] = dating['essay2'].apply(lambda x: avg_word_len(x))
dating['essay3_word_len'] = dating['essay3'].apply(lambda x: avg_word_len(x))
dating['essay4_word_len'] = dating['essay4'].apply(lambda x: avg_word_len(x))
dating['essay5_word_len'] = dating['essay5'].apply(lambda x: avg_word_len(x))
dating['essay6_word_len'] = dating['essay6'].apply(lambda x: avg_word_len(x))
dating['essay7_word_len'] = dating['essay7'].apply(lambda x: avg_word_len(x))
dating['essay8_word_len'] = dating['essay8'].apply(lambda x: avg_word_len(x))
dating['essay9_word_len'] = dating['essay9'].apply(lambda x: avg_word_len(x))


essay_word_lengths = ['essay0_word_len', 'essay1_word_len','essay2_word_len','essay3_word_len', 'essay4_word_len', 'essay5_word_len', 'essay6_word_len','essay7_word_len', 'essay8_word_len', 'essay9_word_len']

dating['avg_word_len'] = dating[essay_word_lengths].mean(axis=1)

def avg_word_freq(essay):
    word_list = essay.split()
    counts = Counter(word_list)
    return counts['me'] + counts['i']

dating['essay0_count'] = dating['essay0'].apply(lambda x: avg_word_freq(x))
dating['essay1_count'] = dating['essay1'].apply(lambda x: avg_word_freq(x))
dating['essay2_count'] = dating['essay2'].apply(lambda x: avg_word_freq(x))
dating['essay3_count'] = dating['essay3'].apply(lambda x: avg_word_freq(x))
dating['essay4_count'] = dating['essay4'].apply(lambda x: avg_word_freq(x))
dating['essay5_count'] = dating['essay5'].apply(lambda x: avg_word_freq(x))
dating['essay6_count'] = dating['essay6'].apply(lambda x: avg_word_freq(x))
dating['essay7_count'] = dating['essay7'].apply(lambda x: avg_word_freq(x))
dating['essay8_count'] = dating['essay8'].apply(lambda x: avg_word_freq(x))
dating['essay9_count'] = dating['essay9'].apply(lambda x: avg_word_freq(x))

essay_word_count = ['essay0_count', 'essay1_count', 'essay2_count', 'essay3_count', 'essay4_count', 'essay5_count', 'essay6_count', 'essay7_count', 'essay8_count', 'essay9_count']

dating['essay_word_count'] = dating[essay_word_count].mean(axis=1)

dating["edu_code"] = dating.education.astype("category").cat.codes

dating["age_code"] = dating.age.astype("category").cat.codes

dating["edu_code"].fillna(-1, inplace=True)

dating["income_code"] = dating.income.astype("category").cat.codes

dating["income_code"].fillna(-1, inplace=True)

dating["sign_code"] = dating.sign.astype("category").cat.codes

dating["sign_code"].fillna(-1, inplace=True)

feature_data = dating[['smokes_code', 'drinks_code', 'drugs_code', 'avg_essay_len', 'avg_word_len']]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

labels= dating['sign_code']


k_classifier = KNeighborsClassifier(n_neighbors=294)
k_classifier.fit(feature_data, labels)
yk_predict = k_classifier.predict(feature_data)
print(accuracy_score(labels, yk_predict))

svc_classifier = SVC(kernel = "rbf", gamma=6)
svc_classifier.fit(feature_data, labels)
sy_predict = svc_classifier.predict(feature_data)
print(accuracy_score(labels, sy_predict))

feature_data2 = dating[['edu_code', 'income_code']]

x2 = feature_data2.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled2 = min_max_scaler.fit_transform(x2)

feature_data2 = pd.DataFrame(x_scaled2, columns=feature_data2.columns)

labels2=dating['sex_code']


k2_classifier = KNeighborsClassifier(n_neighbors=26)
k2_classifier.fit(feature_data2, labels2)
yk2_predict = k2_classifier.predict(feature_data2)
print(accuracy_score(labels2, yk2_predict))


svc2_classifier = SVC(kernel = "rbf", gamma=2)
svc2_classifier.fit(feature_data2, labels2)
sy2_predict = svc2_classifier.predict(feature_data2)
print(accuracy_score(labels2, sy2_predict))




feature_data3 = dating[['avg_word_len']]

x3 = feature_data3.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled3 = min_max_scaler.fit_transform(x3)

feature_data3 = pd.DataFrame(x_scaled3, columns=feature_data3.columns)

labels3=dating['edu_code']




k3_classifier = KNeighborsClassifier(n_neighbors=34)
k3_classifier.fit(feature_data3, labels3)
yk3_predict = k3_classifier.predict(feature_data3)
print(accuracy_score(labels3, yk3_predict))


svc3_classifier = SVC(kernel = "rbf", gamma=2)
svc3_classifier.fit(feature_data3, labels3)
sy3_predict = svc3_classifier.predict(feature_data3)
print(accuracy_score(labels3, sy3_predict))


feature_data4 = dating[['avg_essay_len', 'avg_word_len']]

income_labels = dating['income_code']

train_data4, test_data4, train_labels4, test_labels4 = train_test_split(feature_data4, income_labels, test_size = 0.2, random_state = 1)

model = LinearRegression()
model.fit(train_data4,train_labels4)

print(model.score(test_data4,test_labels4))

print(model.coef_)


regressor = KNeighborsRegressor(n_neighbors = 2, weights = "distance")

regressor.fit(test_data4, test_labels4)

print(regressor.score(train_data4,train_labels4))




feature_data5 = dating[['essay_word_count']]

age_labels = dating['age_code']

train_data5, test_data5, train_labels5, test_labels5 = train_test_split(feature_data5, age_labels, test_size = 0.2, random_state = 1)

model = LinearRegression()
model.fit(train_data5,train_labels5)

print(model.score(test_data5,test_labels5))

print(model.coef_)


regressor = KNeighborsRegressor(n_neighbors = 2, weights = "distance")

regressor.fit(test_data5, test_labels5)

print(regressor.score(train_data5,train_labels5))


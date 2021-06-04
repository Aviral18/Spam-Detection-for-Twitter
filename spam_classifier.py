#-*- coding: utf-8 -*-
import time
import pickle as pk
import pandas as pd
import numpy as np
import tweet_catch as tC
import matplotlib.pyplot as plt
import seaborn as sns
import data_preproc as dp
import warnings
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


tweets = pd.read_csv('spamham.csv')
tweets['length'] = tweets['v2'].apply(len)


tweet_pre = tweets.tail(10)
tweet_group = tweets.groupby('v1').describe()

print('*********** DATASET ***********')
print(tweet_pre)
print('\n------------------\n')
print('*********** DATASET GROUPED BY LABEL ***********')
print(tweet_group)
print('\n------------------\n')



totalTweets = tweets['v2'].shape[0]
    
spamwrds = ' '.join(list(tweets[tweets['v1'] == 'spam']['v2']))
spamWcloud = WordCloud(width = 512, height = 512).generate(spamwrds)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spamWcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
print('*********** SPAM WORD CLOUD ***********')
plt.show()
print('\n------------------\n')
    
hamwrds = ' '.join(list(tweets[tweets['v1'] == 'ham']['v2']))
hamWcloud = WordCloud(width = 512, height = 512).generate(hamwrds)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hamWcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
print('*********** HAM WORD CLOUD ***********')
plt.show()
print('\n------------------\n')

trainOffset, testOffset = list(), list()



for i in range(tweets.shape[0]):
    if np.random.uniform(0, 1) < 0.8:
        trainOffset += [i]
    else:
        testOffset += [i]


tr_data = tweets.loc[trainOffset]
ts_data = tweets.loc[testOffset]

tr_data.reset_index(inplace = True)
tr_data.drop(['index'], axis = 1, inplace = True)
tr_prev = tr_data.head(15)
print('*********** WINDOW OF TRAINING DATA ***********')
print(tr_prev)
print('\n')


train_group = tr_data['v1'].value_counts()
print('*********** TRAINING DATA GROUPED BY LABEL ***********')
print(train_group)
print('\n------------------\n')

tweet_featrs = tweets['v2'].copy()
tweet_featrs = tweet_featrs.apply(dp.snowball_process)
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(tweet_featrs)

features_tr, features_ts, labels_tr, labels_ts = train_test_split(features, tweets['v1'], test_size=0.2, random_state=111)

def saveMNB(vectorizer, classifier):
    
    with open('modelMNB.pkl', 'wb') as infile:
        pk.dump((vectorizer, classifier), infile)
        
def loadMNB():
    
    with open('modelMNB.pkl', 'rb') as file:
      vectorizer, classifier = pk.load(file)
    return vectorizer, classifier

alg_name = []
alg_accuracies = []
alg_f1 = []

mnb = MultinomialNB(alpha=0.2)
mnb.fit(features_tr, labels_tr)
prediction = mnb.predict(features_ts)
conf_mat = metrics.confusion_matrix(labels_ts, prediction)
print('MULTINOMIAL NAIVE BAYES SCORE:', accuracy_score(labels_ts, prediction))
alg_name.append('Multinomial Naive Bayes')
alg_accuracies.append(mnb.score(features_ts, labels_ts) * 100)

bnb = BernoulliNB()
bnb.fit(features_tr, labels_tr)
pred_bnb = bnb.predict(features_ts)
conf_mat_bnb = metrics.confusion_matrix(labels_ts, pred_bnb)
print('BERNOULLI NAIVE BAYES SCORE:', accuracy_score(labels_ts,pred_bnb))
alg_name.append('Bernoulli Naive Bayes')
alg_accuracies.append(bnb.score(features_ts, labels_ts) * 100)

svc = SVC(gamma="scale")
svc.fit(features_tr, labels_tr)
pred_svc = svc.predict(features_ts)
conf_mat_svc = metrics.confusion_matrix(labels_ts, pred_svc)
print('SUPPORT VECTOR MACHINE SCORE:', accuracy_score(labels_ts, pred_svc))
alg_name.append('Support Vector Machine')
alg_accuracies.append(svc.score(features_ts, labels_ts) * 100)

saveMNB(vectorizer, mnb)

print("\nMNB Classifier accuracy {:.2f}%".format(mnb.score(features_ts, labels_ts) * 100))
print("BNB Classifier accuracy {:.2f}%".format(bnb.score(features_ts, labels_ts) * 100))
print("SVC Classifier accuracy {:.2f}%".format(svc.score(features_ts, labels_ts) * 100))



mnb_fscore = metrics.f1_score(labels_ts, prediction, average='macro')
print("\nMNB F1 score is: {:.2f}".format(mnb_fscore))
alg_f1.append(mnb_fscore)

bnb_fscore = metrics.f1_score(labels_ts, pred_bnb, average='macro')
print("BNB F1 score is: {:.2f}".format(bnb_fscore))
alg_f1.append(bnb_fscore)

svc_fscore = metrics.f1_score(labels_ts, pred_svc, average='weighted')
print("SVC F1 score is: {:.2f}".format(svc_fscore))
alg_f1.append(svc_fscore)

data = pd.DataFrame({'alg_name': alg_name, 'alg_accuracies': alg_accuracies})
sorted_data = data.reindex((data['alg_accuracies'].sort_values(ascending=False)).index.values)

plt.subplots(figsize=(10,6))
sns.barplot(x=sorted_data['alg_name'], y=sorted_data['alg_accuracies'], edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=-45)
plt.xlabel('Algorithm Name')
plt.ylabel('Algorithm Accuracy')
plt.title('Algorithm Train Accuracy Comparison')
plt.show()

dataF1 = pd.DataFrame({'alg_name': alg_name, 'alg_f1': alg_f1})
sorted_dataF1 = dataF1.reindex((dataF1['alg_f1'].sort_values(ascending=False)).index.values)

plt.subplots(figsize=(10,6))
sns.barplot(x=sorted_dataF1['alg_name'], y=sorted_dataF1['alg_f1'], edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=-45)
plt.xlabel('Algorithm Name')
plt.ylabel('Algorithm F1 Score')
plt.title('Algorithm F1 Accuracy Score Comparison')
plt.show()



labels = ['HAM', 'SPAM']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat)
plt.title('Confusion Matrix Of The MNB Classifier\n')
fig.colorbar(cax)
tick_marks = np.arange(len(labels))
plt.xlabel('\nPredicted Label')
plt.ylabel('True Label')
plt.xticks(tick_marks, labels, rotation=0)
plt.yticks(tick_marks, labels)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(conf_mat[i][j]))
plt.show()

vectorizer, mnb = loadMNB()




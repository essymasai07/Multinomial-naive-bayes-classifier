# importing necessary libraries and listing files in a directory
import os
directory = 'C:\\Users\\Masai\\Documents\\python projects\\youtube-dataset'
files = os.listdir(directory)
print("List of Files in the Directory:")
print(files)

#Importing CSV files, dropping some columns and concatenating them into a single dataframe
import pandas as pd
import glob
files = glob.glob('*.csv')
print("List of CSV Files:")
print(files)
all_df = []
for i in files:
    all_df.append(pd.read_csv(i).drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1))
print(all_df)
data= pd.concat(all_df, axis=0, ignore_index=True)
print(data)

# check for any null values
print(data.isnull().sum())

# look at the number of spams(1) and hams(0)
print(data['CLASS'].value_counts())

# count vectorizer to count the occurence of each token in a comment
from sklearn.feature_extraction.text import CountVectorizer
message_sample= ['This is a dog']
vectorizer_sample= CountVectorizer()
vectorizer_sample.fit(message_sample)
print(vectorizer_sample.transform(message_sample).toarray())
print(vectorizer_sample.get_feature_names_out())
print(vectorizer_sample.transform(['This is a cat']).toarray())
message_sample2= ['This is a dog and that is a dog', 'That is a cat']
vectorizer_sample2= CountVectorizer()
print(vectorizer_sample2.fit_transform(message_sample2).toarray())
print(vectorizer_sample2.get_feature_names_out())
print(vectorizer_sample2.transform(['Those are birds']).toarray())

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
inputs=data['CONTENT']
target= data['CLASS']
x_train,  x_test, y_train, y_test= train_test_split(inputs, target, test_size=0.2, random_state=365, stratify=target)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
vectorizer= CountVectorizer()
x_train_trans=vectorizer.fit_transform(x_train)
x_test_trans=vectorizer.transform(x_test)
print(x_train_trans.toarray())
print(x_train_trans.shape)

# training a naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
clf=MultinomialNB(class_prior=np.array([0.6,0.4]))
clf.fit(x_train_trans, y_train)
print(clf.get_params())

# performing evaluation on the test dataset and generating visualizations
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
y_test_pred=clf.predict(x_test_trans)
cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, labels=clf.classes_, cmap='magma')
cm_display.plot(cmap='magma', values_format='d')
import matplotlib.pyplot as plt
plt.show()
print(classification_report(y_test,y_test_pred, target_names=['Ham', 'Spam']))

import seaborn as sns
print(np.exp(clf.class_log_prior_))
sns.reset_orig()

# conditional distribution probability figures
spam_proba= clf.predict_proba(x_test_trans).round(3)[:,1]
df_scatter= pd.DataFrame()
df_scatter['True class']=y_test
df_scatter['Predicted class']=y_test_pred
df_scatter['Predicted probability (spam)']=spam_proba
df_scatter=df_scatter.reset_index(drop=True)
palette_0=sns.color_palette(['#000000'])
palette_1=sns.color_palette(['#FF0000'])

df_scatter_0=df_scatter[df_scatter['True class'] ==0].reset_index(drop=True)
df_scatter_1=df_scatter[df_scatter['True class'] ==1].reset_index(drop=True)
sns.set()
fig, (ax1, ax2)= plt.subplots(2,1, figsize=(12,5))
fig.tight_layout(pad=3)
sns.scatterplot(x='Predicted probability (spam)',
                y=np.zeros(df_scatter_0.shape[0]),
                data=df_scatter_0,
                hue= 'True class',
                s=50,
                markers='o',
                palette=palette_0,
                style='True class',
                legend=False,
                ax=ax1).set(yticklabels=[])
ax1.set_title('Probability distribution of comments belonging to the true \'ham\'class')
ax1.vlines(0.5, -1, 1, linestyles='dashed', colors='red')
plt.show()
sns.scatterplot(x='Predicted probability (spam)',
                y=np.zeros(df_scatter_1.shape[0]),
                data=df_scatter_1,
                hue= 'True class',
                s=50,
                markers='x',
                palette=palette_1,
                style='True class',
                legend=False,
                ax=ax2).set(yticklabels=[])
ax2.set_title('Probability distribution of comments belonging to the true \'spam\'class')
ax2.vlines(0.5, -1, 1, linestyles='dashed', colors='black')
plt.show()

# predicting new data
predict_data=vectorizer.transform(['This song is amazing!',
                                   'You can win 1M dollars right now, just click here!!!','You have a good taste of fashion!', 'You sing so well'])
print(clf.predict(predict_data))
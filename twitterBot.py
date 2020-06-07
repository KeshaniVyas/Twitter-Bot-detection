import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                           r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                           r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                           r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

def bot_detection(df1):
   # df=df1.copy()
   df=df1.loc[:,:]
   
   plt.figure(figsize=(10,6))
   sns.heatmap(df1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
   plt.tight_layout()
   plt.show()
   
   df['id'] = df['id'].apply(lambda x: int(x))
   df['followers_count'] =df.followers_count.apply(lambda x: 0 if x=='None' else int(x))
   df['friends_count'] =df.friends_count.apply(lambda x: 0 if x=='None' else int(x))
   df['listed_count'] =df.listed_count.apply(lambda x: 0 if x=='None' else int(x))
   df['statuses_count'] =df.statuses_count.apply(lambda x: 0 if x=='None' else int(x))
   df['verified'] = df.verified.apply(lambda x: 1 if ((x == True) or x == 'TRUE') else 0)
   df['screen_name_binary'] = (df.screen_name.str.contains(bag_of_words_bot, case=False)==True)
   df['description_binary'] = (df.description.str.contains(bag_of_words_bot, case=False)==True)
   df['name_binary'] = (df.name.str.contains(bag_of_words_bot, case=False)==True)
   df['status_binary'] = (df.status.str.contains(bag_of_words_bot, case=False)==True)
   df['verified_binary'] = (df.verified==False)
   
   features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified','followers_count','friends_count','statuses_count','listed_count']
   X=df[features]
   y=df['bot']

   clf = RandomForestClassifier(n_estimators=80)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
   clf.fit(X_train,y_train)

   clf1=LogisticRegression(C=10000)
   clf1.fit(X_train,y_train)
   
   clf2 = GaussianNB()
   clf2.fit(X_train,y_train)

   clf4 = DecisionTreeClassifier()
   clf4.fit(X_train,y_train)

   from sklearn import metrics
   results=[]
   names=[]
   y_pred_test = clf.predict(X_test) 
   y_pred_test1 = clf1.predict(X_test) 
   y_pred_test2 = clf2.predict(X_test) 
   y_pred_test4 = clf4.predict(X_test) 
   acc1=metrics.accuracy_score(y_test, y_pred_test)
   
   print('RandomForest')
   print("Accuracy:",acc1)
   print("F1-Score:",metrics.f1_score(y_test, y_pred_test))
   print("Classification report:",metrics.classification_report(y_test, y_pred_test))
   results.append(acc1*100)
   names.append('RandomForest')

   print('LogisticRegression')
   acc2=metrics.accuracy_score(y_test, y_pred_test1)
   print("Accuracy:",acc2)
   print("F1-Score:",metrics.f1_score(y_test, y_pred_test1))
   print("Classification report:",metrics.classification_report(y_test, y_pred_test1))
   results.append(acc2*100)
   names.append('LogisticRegression')

   print('NaiveBayes')
   acc3=metrics.accuracy_score(y_test, y_pred_test2)
   print("Classification report:",metrics.classification_report(y_test, y_pred_test2))
   print("Accuracy:",acc3)
   print("F1-Score:",metrics.f1_score(y_test, y_pred_test2))
   results.append(acc3*100)
   names.append('Naive bayes')

   print('Decision Tree')
   acc5=metrics.accuracy_score(y_test, y_pred_test4)
   print("Accuracy:",acc5)
   print("Classification report:",metrics.classification_report(y_test, y_pred_test4))
   print("F1-Score:",metrics.f1_score(y_test, y_pred_test4))
   results.append(acc5*100)
   names.append('Decision Tree')
   
   import numpy as np
   print(results)
   print(names)
   fig=plt.figure(1,figsize=(9,6))
   fig.suptitle('Algorithm Comparison')
#   ax=fig.add_subplot(111)
   index=np.arange(len(names))
   plt.bar(index,results,width=0.5)
   plt.xticks(index,names,rotation=15)
   plt.xlabel('ML Algorithm')
   plt.ylabel('Accuracy score')
   plt.show()
   
   
train_df = pd.read_csv('mini_train.csv')
pd.set_option('mode.chained_assignment',None)
bot_detection(train_df)
   

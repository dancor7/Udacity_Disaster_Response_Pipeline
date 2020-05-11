#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[22]:


# import libraries
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.simplefilter("ignore")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
import pickle
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


# In[2]:


# load data from database
def load_data():
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('clean_dataset', con=engine).head(10000)
    X = df['message']
    y = df.drop(['original','genre','message','offer','request'], axis=1)
    return X, y


# In[3]:


engine = create_engine('sqlite:///data/DisasterResponse.db')
pd.read_sql_table('clean_dataset', con=engine).shape


# ### 2. Write a tokenization function to process your text data

# In[4]:


def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[5]:


model1 = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[6]:


X, y = load_data()
y= y.astype(int)

# split data, train and predict
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[7]:



#model1.fit(X_train.values, y_train.values)
#y_pred = model1.predict(X_test)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[8]:


# Get results and add them to a dataframe.
def get_results(y_test, y_pred):
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for cat in y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[cat], y_pred[:,num], average='weighted')
        results.set_value(num+1, 'Category', cat)
        results.set_value(num+1, 'f_score', f_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)
        num += 1
    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    return results


# In[9]:


#results = get_results(y_test, y_pred)
#results


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[10]:


#model1.get_params()


# In[13]:


#parameters = {'clf__estimator__max_depth': [10, 50, None],
#              'clf__estimator__min_samples_leaf':[2, 5, 10]}

#model2 = GridSearchCV(model1, parameters)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[14]:



#model2.fit(X_train.as_matrix(), y_train.as_matrix())
#y_pred = model2.predict(X_test)


#results2 = get_results(y_test, y_pred)
#results2


# In[15]:


#model2.best_estimator_.get_params()


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[16]:



class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# In[17]:


from sklearn.tree import DecisionTreeClassifier

def model_pipeline():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])

    return pipeline


# In[18]:


model3 = model_pipeline()
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)


# In[19]:


results3 = get_results(y_test, y_pred)
results3


# ### 9. Export your model as a pickle file

# In[23]:



pickle.dump(model3, open('models/classifier.pkl', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:





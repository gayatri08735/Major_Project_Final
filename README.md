github url : https://github.com/gayatri08735/Major_Project_Final/

app link : https://emotiondecoder.streamlit.app/

# ## DATASET INFORMATION
Dataset name : Text_Emotion_Dataset

No. of rows : 30805

No. of columns : 2

column names / features : Emotion, Text

Class labels : 8 (joy, sadness, fear, anger, surprise, neutral, disgust, shame)

Test size : 0.15 (85 percent training data + 15 percent testing data)

No. of rows in Training data : 0.85 * 30805 = 26184 rows

No. of rows in Testing data : 0.15 * 30805 = 4621 rows

# ## MODELS USED FOR TRAINING AND THEIR SCORES, REF LINKS
Random Forest Classifier (60.63 Percent) - https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/

Support Vector Machine (71.73 Percent) - https://www.geeksforgeeks.org/support-vector-machine-algorithm/

Multinomial Naive Bayes (64.92 Percent) - https://www.geeksforgeeks.org/applying-multinomial-naive-bayes-to-nlp-problems/

Logistic Regression (74.03 Percent) - https://www.geeksforgeeks.org/text-classification-using-logistic-regression/

# ## MODULES AND METHODS USED

# 1. Pandas:
Description: Pandas is a powerful data manipulation and analysis library for Python. It provides data structures like DataFrame and Series, making it easy to handle and analyze structured data.

Key Methods:

pd.read_csv("text_emotion_dataset.csv"): Reads a CSV file and creates a DataFrame.

df.head(): Displays the first few rows of the DataFrame.

df.shape: Returns the dimensions (rows, columns) of the DataFrame.

df.dtypes: Returns the data types of each column.

df.isnull().sum(): Returns the count of missing values in each column.

df['Emotion'].value_counts(): Returns the count of unique values in the 'Emotion' column.

# 2. Seaborn and Matplotlib:

Description: Seaborn and Matplotlib are visualization libraries for Python. Seaborn is built on top of Matplotlib and provides a high-level interface for creating informative and attractive statistical graphics.

Key Methods:

sns.countplot(x='Emotion', data=df): Plots a count distribution of the 'Emotion' column using Seaborn.

plt.figure(figsize=(20,10)): Sets the size of the Matplotlib figure.

sns.countplot(x='Emotion', data=df): Plots a count distribution of the 'Emotion' column using Seaborn.

plt.show(): Displays the Seaborn plot.

# 3. TextBlob:
Description: TextBlob is a library for processing textual data. It provides a simple API for common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.

Key Methods:

TextBlob(text): Creates a TextBlob object for sentiment analysis.

blob.sentiment.polarity: Returns the sentiment polarity of the text.

# 4. Neattext:

Description: Neattext is a Python library for cleaning and preprocessing textual data. It includes functions for removing user handles, stopwords, punctuations, emojis, and special characters from text.

Key Methods:

nfx.remove_userhandles(text): Removes user handles from the text.

nfx.remove_stopwords(text): Removes stopwords from the text.

nfx.remove_punctuations(text): Removes punctuations from the text.

nfx.remove_emojis(text): Removes emojis from the text.

nfx.remove_special_characters(text): Removes special characters from the text.

# 5. Counter:

Description: Counter is a built-in Python module that provides a convenient way to count occurrences of elements in a collection (e.g., a list).

Key Methods:

Counter(tokens).most_common(num): Counts the occurrences of each token and returns the most common ones.

# 6. WordCloud:

Description: WordCloud is a Python library for creating word clouds from text data. It visualizes the most frequent words in a given text, with more frequent words appearing larger.

Key Methods:

WordCloud(): Creates a WordCloud object for generating word clouds.

# 7. Scikit-Learn:

Description: Scikit-Learn is a machine learning library for Python. It provides simple and efficient tools for data analysis and modeling, including various machine learning algorithms and tools for model selection and evaluation.

Key Methods:

train_test_split(x, y, test_size=0.15, random_state=42): Splits the data into training and test sets.

Pipeline(steps=[('cv', CountVectorizer()), ('rf', RandomForestClassifier(n_estimators=10))]): Creates a pipeline for Random Forest classification.

Pipeline(steps=[('cv', CountVectorizer()), ('svc', SVC(kernel='rbf', C=10))]): Creates a pipeline for Support Vector Machine classification.

MultinomialNB(): Creates a Multinomial Naive Bayes classifier.

Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression(max_iter=1000))]): Creates a pipeline for Logistic Regression classification.

# 8. Joblib:

Description: Joblib is a library for lightweight pipelining in Python. It provides tools to provide lightweight pipelining in Python.

Key Methods:

joblib.dump(pipe_lr, pipeline_file): Saves the trained Logistic Regression pipeline to a file.

joblib.load("text_emotion.pkl"): Loads the saved pipeline from a file.

# 9. Tesseract OCR:

Description: Tesseract OCR is an open-source optical character recognition engine developed by Google. It is used to extract text from images.

Key Methods:

extract_english_text_from_image(image_path): Uses Tesseract OCR to extract English text from an image.

extract_telugu_text_from_image(image_path): Uses Tesseract OCR to extract Telugu text from an image.

# 10. Translation Function:

Description: Custom translation function to translate Telugu text to English.

Key Methods:

translate_telugu_to_english(text): Translates Telugu text to English.

 # ## Code explanation ##

#  imported the necessary libraries for data analysis and visualization

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

# read the CSV file into a Pandas DataFrame named df
df = pd.read_csv("text_emotion_dataset.csv")

# head() method to display the first few rows of the DataFrame (df).
df.head()

Emotion	Text	Sentiment

0	neutral	Why ?	Neutral

1	joy	Sage Act upgrade on my to do list for tommorow.	Neutral

2	sadness	ON THE WAY TO MY HOMEGIRL BABY FUNERAL!!! MAN ...	Negative

3	joy	Such an eye ! The true hazel eye-and so brill...	Positive

4	joy	@Iluvmiasantos ugh babe.. hugggzzz for u .! b...	Neutral

# shape attribute to get the dimensions of the DataFrame (df). The result indicates that the DataFrame has 30,805 rows and 3 columns. This information helps you understand the size of your dataset. 

df.shape

(30805, 3)

# dtypes attribute to display the data types of each column in the DataFrame (df). 
df.dtypes

Emotion      object

Text         object

Sentiment    object

dtype: object

# isnull().sum() method to check for missing values in each column of the DataFrame (df). The result shows that there are no missing values in any of the columns, which is great for data analysis as it means your dataset is complete and doesn't require imputation.
df.isnull().sum()

Emotion      0

Text         0

Sentiment    0

dtype: int64

# value_counts() method on the 'Emotion' column to count the occurrences of each unique emotion in your dataset. 
df['Emotion'].value_counts()

joy         10264

sadness      5820

fear         4858

anger        3659

surprise     3430

neutral      1966

disgust       667

shame         141

Name: Emotion, dtype: int64

# Seaborn's countplot to visualize the distribution of emotions in your dataset. This type of plot is useful for understanding the relative frequencies of different emotions.

sns.countplot(x='Emotion',data=df)

<AxesSubplot:xlabel='Emotion', ylabel='count'>

# Matplotlib's figure function to create a larger figure for your subsequent plots. The figsize parameter is set to (20, 10), which means the width of the figure is 20 units, and the height is 10 units. This can be helpful when you want to create larger visualizations to display more information or make it easier to read.

plt.figure(figsize=(20,10))

# Seaborn's countplot to visualize the distribution of emotions in your dataset, and then you used Matplotlib's show function to display the plot. This bar plot provides a clear representation of the counts of each emotion category.

sns.countplot(x='Emotion',data=df)

plt.show()

# imported the TextBlob class from the textblob module. TextBlob is a powerful library for processing textual data, and it provides simple APIs for common natural language processing (NLP) tasks, such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.

from textblob import TextBlob

# You've defined a function named get_sentiment that takes a text input, uses TextBlob to analyze the sentiment polarity of the text, and returns a sentiment label ("Positive," "Negative," or "Neutral") based on the polarity score. If the sentiment polarity is greater than 0, it's labeled as "Positive"; if it's less than 0, it's labeled as "Negative"; otherwise, it's labeled as "Neutral." In your example with the text "I Love coding," the sentiment analysis classifies it as "Positive."

def get_sentiment(text):

    blob=TextBlob(text)
    
    sentiment=blob.sentiment.polarity
    
    if sentiment>0:
    
        result="Positive"
    
    elif sentiment<0:
    
        result="Negative"
    
    else:
    
        result="Neutral"
    
    return result

get_sentiment("I Love coding")

'Positive'

# You've added a new column named 'Sentiment' to your DataFrame (df) by applying the get_sentiment function to the 'Text' column. This new column contains the sentiment labels ('Positive,' 'Negative,' or 'Neutral') based on the sentiment analysis of the corresponding text in the 'Text' column.

df['Sentiment']=df['Text'].apply(get_sentiment)

df.head()


Emotion	Text	Sentiment

0	neutral	Why ?	Neutral

1	joy	Sage Act upgrade on my to do list for tommorow.	Neutral

2	sadness	ON THE WAY TO MY HOMEGIRL BABY FUNERAL!!! MAN ...	Negative

3	joy	Such an eye ! The true hazel eye-and so brill...	Positive

4	joy	@Iluvmiasantos ugh babe.. hugggzzz for u .! b...	Neutral

# groupby function to group the DataFrame by both 'Emotion' and 'Sentiment' columns and then applied the size function to get the count of each group. The result is a Series with a multi-level index, representing the counts for each combination of 'Emotion' and 'Sentiment.' 

df.groupby(['Emotion','Sentiment']).size()



Emotion   Sentiment

anger     Negative     1612

          Neutral      1115
          
          Positive      932

disgust   Negative      250

          Neutral       197
          
          Positive      220

fear      Negative     1368

          Neutral      1636
         
          Positive     1854

joy       Negative     1498

          Neutral      3382
          
          Positive     5384

neutral   Negative      108

          Neutral      1406
          
          Positive      452

sadness   Negative     2355

          Neutral      1793
          
          Positive     1672

shame     Negative       43

          Neutral        48
          
          Positive       50

surprise  Negative      513

          Neutral      1348
          
          Positive     1569

dtype: int64

# You've used the plot function to create a bar plot of the counts obtained from the groupby operation. The x-axis represents the unique combinations of 'Emotion' and 'Sentiment,' and the y-axis represents the corresponding count for each combination. The resulting bar plot visually shows the distribution of sentiments within each emotion category.

df.groupby(['Emotion','Sentiment']).size().plot(kind='bar')

<AxesSubplot:xlabel='Emotion,Sentiment'>

# factorplot function from Seaborn to create a categorical plot with the 'Emotion' variable on the x-axis, the hue representing the 'Sentiment' variable, and the count of occurrences for each combination of 'Emotion' and 'Sentiment.' However, please note that the factorplot function has been deprecated, and you might want to use catplot instead. Also, the size parameter has been renamed to height in recent versions of Seaborn.

sns.factorplot(x='Emotion',hue='Sentiment',data=df,kind='count',size=6,aspect=1.5)

C:\Users\gayatri\anaconda3\lib\site-packages\seaborn\categorical.py:3717: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
  warnings.warn(msg)

C:\Users\gayatri\anaconda3\lib\site-packages\seaborn\categorical.py:3723: UserWarning: The `size` parameter has been renamed to `height`; please update your code.
  warnings.warn(msg, UserWarning)
<seaborn.axisgrid.FacetGrid at 0x1bf48436c70>


# Data pre-processing, neattext is a Python library that provides functions for text cleaning and preprocessing.
import neattext.functions as nfx

# neattext is a Python library that provides functions for text cleaning and preprocessing. Let's go through the methods used in your code:

# nfx.remove_userhandles: This function is used to remove user handles (mentions) from the text. User handles typically start with '@' in social media contexts.
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

# The dir(nfx) command returns a list of all the attributes and methods available in the neattext.functions module.
dir(nfx)

['BTC_ADDRESS_REGEX',
 'CURRENCY_REGEX',
 'CURRENCY_SYMB_REGEX',
 'Counter',
 'DATE_REGEX',
 'EMAIL_REGEX',
 'EMOJI_REGEX',
 'HASTAG_REGEX',
 'MASTERCard_REGEX',
 'MD5_SHA_REGEX',
 'MOST_COMMON_PUNCT_REGEX',
 'NUMBERS_REGEX',
 'PHONE_REGEX',
 'PoBOX_REGEX',
 'SPECIAL_CHARACTERS_REGEX',
 'STOPWORDS',
 'STOPWORDS_de',
 'STOPWORDS_en',
 'STOPWORDS_es',
 'STOPWORDS_fr',
 'STOPWORDS_ru',
 'STOPWORDS_yo',
 'STREET_ADDRESS_REGEX',
 'TextFrame',
 'URL_PATTERN',
 'USER_HANDLES_REGEX',
 'VISACard_REGEX',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__generate_text',
 '__loader__',
 '__name__',
 '__numbers_dict',
 '__package__',
 '__spec__',
 '_lex_richness_herdan',
 '_lex_richness_maas_ttr',
 'clean_text',
 'defaultdict',
 'digit2words',
 'extract_btc_address',
 'extract_currencies',
 'extract_currency_symbols',
 'extract_dates',
 'extract_emails',
 'extract_emojis',
 'extract_hashtags',
 'extract_html_tags',
 'extract_mastercard_addr',
 'extract_md5sha',
 'extract_numbers',
 'extract_pattern',
 'extract_phone_numbers',
 'extract_postoffice_box',
 'extract_shortwords',
 'extract_special_characters',
 'extract_stopwords',
 'extract_street_address',
 'extract_terms_in_bracket',
 'extract_urls',
 'extract_userhandles',
 'extract_visacard_addr',
 'fix_contractions',
 'generate_sentence',
 'hamming_distance',
 'inverse_df',
 'lexical_richness',
 'markov_chain',
 'math',
 'nlargest',
 'normalize',
 'num2words',
 'random',
 're',
 'read_txt',
 'remove_accents',
 'remove_bad_quotes',
 'remove_btc_address',
 'remove_currencies',
 'remove_currency_symbols',
 'remove_custom_pattern',
 'remove_custom_words',
 'remove_dates',
 'remove_emails',
 'remove_emojis',
 'remove_hashtags',
 'remove_html_tags',
 'remove_mastercard_addr',
 'remove_md5sha',
 'remove_multiple_spaces',
 'remove_non_ascii',
 'remove_numbers',
 'remove_phone_numbers',
 'remove_postoffice_box',
 'remove_puncts',
 'remove_punctuations',
 'remove_shortwords',
 'remove_special_characters',
 'remove_stopwords',
 'remove_street_address',
 'remove_terms_in_bracket',
 'remove_urls',
 'remove_userhandles',
 'remove_visacard_addr',
 'replace_bad_quotes',
 'replace_currencies',
 'replace_currency_symbols',
 'replace_dates',
 'replace_emails',
 'replace_emojis',
 'replace_numbers',
 'replace_phone_numbers',
 'replace_special_characters',
 'replace_term',
 'replace_urls',
 'string',
 'term_freq',
 'to_txt',
 'unicodedata',
 'word_freq',
 'word_length_freq']

# nfx.remove_stopwords: This function removes common stopwords from the text. Stopwords are words that are often excluded from text data because they are considered to be of little value in terms of content (e.g., "the", "and", "is").
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)


# nfx.remove_punctuations: Removes punctuation marks from the text.
df['Clean_Text']=df['Text'].apply(nfx.remove_punctuations)

# nfx.remove_emojis: Removes emojis from the text.
df['Clean_Text']=df['Text'].apply(nfx.remove_emojis)

# nfx.remove_special_characters: Removes special characters from the text.
df['Clean_Text']=df['Text'].apply(nfx.remove_special_characters)

df

Emotion	Text	Sentiment	Clean_Text

0	neutral	Why ?	Neutral	Why

1	joy	Sage Act upgrade on my to do list for tommorow.	Neutral	Sage Act upgrade on my to do list for tommorow

2	sadness	ON THE WAY TO MY HOMEGIRL BABY FUNERAL!!! MAN ...	Negative	ON THE WAY TO MY HOMEGIRL BABY FUNERAL MAN I H...

3	joy	Such an eye ! The true hazel eye-and so brill...	Positive	Such an eye The true hazel eyeand so brillia...

4	joy	@Iluvmiasantos ugh babe.. hugggzzz for u .! b...	Neutral	Iluvmiasantos ugh babe hugggzzz for u babe n...

...	...	...	...	...

30800	joy	When I understood that I was admitted to the U...	Neutral	When I understood that I was admitted to the U...

30801	joy	Tuesday woken up to Oscar and Cornet practice X	Neutral	Tuesday woken up to Oscar and Cornet practice X

30802	surprise	@MichelGW have you gift! Hope you like it! It'...	Positive	MichelGW have you gift Hope you like it Its ha...

30803	joy	The world didnt give it to me..so the world MO...	Positive	The world didnt give it to meso the world MOST...

30804	fear	Youu call it JEALOUSY, I call it of #Losing YO...	Neutral	Youu call it JEALOUSY I call it of Losing YOU

30805 rows × 4 columns


# Counter class from the collections module. The Counter class is a convenient tool for counting the occurrences of elements in a collection, such as a list
from collections import Counter

# defining a function called extract_keywords that takes a text as input and extracts the most common keywords. The function uses the Counter class to count the occurrences of each token (word) in the text and then returns a dictionary containing the most common tokens.
# The num parameter specifies the number of most common tokens to be extracted, and the default value is set to 50.
# Additionally, you have created a list called emotion_list containing unique emotions from the 'Emotion' column of your DataFrame.


def extract_keywords(text,num=50):

    tokens=[tok for tok in text.split()]
    
    most_common_tokens=Counter(tokens).most_common(num)
    
    return dict(most_common_tokens)

emotion_list=df['Emotion'].unique().tolist()

emotion_list

['neutral', 'joy', 'sadness', 'fear', 'surprise', 'anger', 'shame', 'disgust']

# creating a list called joy_list that contains the cleaned text from the 'Clean_Text' column of your DataFrame for rows where the emotion is labeled as 'joy'. 
# combining the text from the 'joy_list' into a single document (joy_docx) and then extracting keywords using the extract_keywords function you defined earlier. This can help you identify the most common keywords associated with the emotion of joy in your dataset.


joy_list=df[df['Emotion']=='joy']['Clean_Text'].tolist()

joy_docx=' '.join(joy_list)

keyword_joy=extract_keywords(joy_docx)

keyword_joy
{'the': 5044,
 'to': 4493,
 'I': 4100,
 'a': 3399,
 'and': 3081,
 'of': 2687,
 'my': 2608,
 'in': 2225,
 'for': 1907,
 'is': 1489,
 'with': 1312,
 'that': 1143,
 'you': 1118,
 'at': 1090,
 'was': 1073,
 'on': 1058,
 'me': 1047,
 'it': 916,
 'have': 905,
 'be': 844,
 'this': 753,
 'day': 706,
 'amp': 664,
 'up': 631,
 'all': 600,
 'had': 594,
 'time': 580,
 'so': 558,
 'Im': 503,
 'work': 499,
 'your': 498,
 'when': 489,
 'The': 486,
 'When': 480,
 'today': 463,
 'tomorrow': 461,
 'Christmas': 454,
 'an': 450,
 'not': 450,
 'get': 449,
 'like': 448,
 'from': 442,
 'love': 434,
 'now': 420,
 'about': 416,
 'just': 408,
 'out': 405,
 'are': 400,
 'happy': 374,
 'as': 365}

# created a function plot_most_common_words that takes a dictionary of tokens and their counts (mydict) along with the emotion name, and then plots the most common keywords associated with that emotion. In this case, you've applied it to the joy emotion and generated a bar plot.


def plot_most_common_words(mydict,emotion_name):

    df_01=pd.DataFrame(mydict.items(),columns=['token','count'])
    
    plt.figure(figsize=(20,10))
    
    plt.title("Plot of {} Most Common Keywords".format(emotion_name))
    
    sns.barplot(x='token',y='count',data=df_01)
    
    plt.xticks(rotation=45)
    
    plt.show()

plot_most_common_words(keyword_joy,"joy")

# implemented a function called plot_wordcloud that generates and displays a word cloud from a given document (docx). In this case, you applied it to the joy_docx, which contains the text associated with the 'joy' emotion.
# Word clouds are visual representations of the most frequent words in a given text, with the size of each word indicating its frequency.

from wordcloud import WordCloud

def plot_wordcloud(docx):

    mywordcloud=WordCloud().generate(docx)
    
    plt.figure(figsize=(20,10))
    
    plt.imshow(mywordcloud,interpolation='bilinear')
    
    plt.axis('off')
    
    plt.show()

plot_wordcloud(joy_docx)


# Splitting data into input variables and target variable
# x: Features are the attributes and variables extracted from the dataset. These extracted features are used as inputs to the model during training.
# y: Labels are the output or the target variable.


x = df['Clean_Text']

y = df['Emotion']

# Splitting data into train and test set
# starting to work on a machine learning task, specifically splitting your data into training and testing sets using train_test_split from scikit-learn (sklearn). Additionally, you've imported accuracy_score from sklearn.metrics, which is typically used to evaluate the performance of classification models.
# train_test_split is a function that splits your data into training and testing sets.
# x is your feature data, and y is your target variable.
# test_size=0.15 specifies that 15% of the data should be used for testing, and the remaining 85% will be used for training.
# random_state=42 is a seed for the random number generator, ensuring 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)

from sklearn.metrics import accuracy_score

# Training the model

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

# It looks like you've created a machine learning pipeline using scikit-learn's Pipeline class. This pipeline consists of two steps:

# CountVectorizer ('cv'): This is a feature extraction method that converts a collection of text documents to a matrix of token counts. It's often used in natural language processing (NLP) tasks.

# RandomForestClassifier ('rf'): This is a machine learning model from the scikit-learn library that is an ensemble of decision trees. The n_estimators=10 parameter indicates that the random forest consists of 10 decision trees.

# By combining these two steps into a pipeline, you can streamline the process of transforming your text data and training a machine learning model.


pipe_rf = Pipeline(steps=[('cv',CountVectorizer()),('rf', RandomForestClassifier(n_estimators=10))])

pipe_rf.fit(x_train,y_train)

pipe_rf.score(x_test,y_test)

0.6109067301449903

# training using support vector machine
# RBF (Radial Basis Function) Kernel: Used for handling non-linear relationships in SVMs.
# C Parameter: Controls the regularization in SVM, balancing between smooth decision boundaries and accurate classification of training points.


pipe_svm = Pipeline(steps=[('cv',CountVectorizer()),('svc', SVC(kernel = 'rbf', C = 10))])

pipe_svm.fit(x_train,y_train)

pipe_svm.score(x_test,y_test)

0.717377191084181

xfeatures=df['Clean_Text']

ylabels=df['Emotion']

# CountVectorizer: This is a text feature extraction technique in scikit-learn. It converts a collection of text documents to a matrix of token counts. In simpler terms, it counts the frequency of each word in the text.

# max_features=5000: The max_features parameter is used to limit the number of features (words) in the output matrix. In your case, it's set to 5000, meaning only the top 5000 most frequent words will be considered, and the rest will be ignored.

# fit_transform(xfeatures): This method fits the CountVectorizer on your input data (xfeatures) and transforms it into a matrix of token counts. The result is a sparse matrix, which you then convert to a dense array using toarray().

cv=CountVectorizer(max_features=5000)

x=cv.fit_transform(xfeatures)

x.toarray()

array([[0, 0, 0, ..., 0, 0, 0],

       [0, 0, 0, ..., 0, 0, 0],
       
       [0, 0, 0, ..., 0, 0, 0],
       
       ...,
       
       [0, 0, 0, ..., 0, 0, 0],
       
       [0, 0, 0, ..., 0, 0, 0],
       
       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)

# trained a Multinomial Naive Bayes (NB) model on your data and evaluated its accuracy on the test set

from sklearn.naive_bayes import MultinomialNB

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)

nv_model=MultinomialNB()

nv_model.fit(x_train,y_train)

nv_model.score(x_test,y_test)

0.6492101276779918

# CountVectorizer (cv): This step is responsible for converting a collection of text documents to a matrix of token counts. It prepares the text data for the machine learning model.

# Logistic Regression (lr): This step involves the logistic regression model. Logistic regression is a commonly used algorithm for binary and multiclass classification problems. The max_iter parameter is set to 1000, which represents the maximum number of iterations for the solver to converge.

# After creating the pipeline, you fit it to the training data (x_train and y_train) using pipe_lr.fit(x_train, y_train). Finally, you evaluate the model's performance on the test set using pipe_lr.score(x_test, y_test).

x = df['Clean_Text']

y = df['Emotion']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)

pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression(max_iter=1000))])

pipe_lr.fit(x_train,y_train)

pipe_lr.score(x_test,y_test)

0.7403159489288033

# Saving the model
# opened a file named "text_emotion.pkl" in binary write mode ("wb").
# used joblib.dump() to save your pipeline (pipe_lr) into the file.
# Closed the file after saving.

import joblib

pipeline_file = open("text_emotion.pkl","wb")

joblib.dump(pipe_lr,pipeline_file)

pipeline_file.close()




import joblib

# Load the trained pipeline
loaded_pipeline = joblib.load("text_emotion.pkl")

# Function to predict emotion for a single text and prediction scores
# Prediction: The function uses the loaded_pipeline to make predictions. loaded_pipeline.predict(text_list) is used to get the predicted emotion, and loaded_pipeline.predict_proba(text_list) is used to get the prediction scores.

def predict_emotion_with_scores(text):

    # Ensure the input text is a list (as required by the pipeline)
    
    text_list = [text]

    
    
    # Use the pipeline to get both predicted emotion and prediction scores
    
    predicted_result = loaded_pipeline.predict(text_list)[0]
    
    prediction_scores = loaded_pipeline.predict_proba(text_list)[0]

    return predicted_result, prediction_scores
    
# Function to predict emotion for a single text

def predict_emotion(text):

    # Ensure the input text is a list (as required by the pipeline)
    
    text_list = [text]

    
    
    # Use the pipeline to predict the emotion
    
    predicted_emotion = loaded_pipeline.predict(text_list)[0]

    
    return predicted_emotion

# Get text input from the user
user_input_text = input("Enter a text for emotion prediction: ")

# Call the prediction function
predicted_emotion, prediction_scores = predict_emotion_with_scores(user_input_text)

# Display the results, the prediction scores of the text are calculated and the emotion with highest score is finalised

print(f"\nThe predicted emotion for the text is: {predicted_emotion}")

print("\nPrediction Scores:")

for emotion, score in zip(loaded_pipeline.classes_, prediction_scores):

    print(f"{emotion}: {score:.4f}")
    
Enter a text for emotion prediction: The cinematography is hauntingly beautiful, with each frame carefully composed to amplify the feeling of impending doom. The play of shadows and eerie lighting choices enhance the unsettling ambiance, making every creaking floorboard and distant murmur a cause for genuine apprehension.

The predicted emotion for the text is: fear


Prediction Scores:

anger: 0.0019

disgust: 0.0156

fear: 0.5203

joy: 0.4547

neutral: 0.0000

sadness: 0.0046

shame: 0.0020

surprise: 0.0009

df['Predicted_Emotion'] = df['Clean_Text'].apply(predict_emotion)

# Print the DataFrame with relevant columns

print(df[['Emotion', 'Text', 'Predicted_Emotion']])

        Emotion                                               Text  \

0       neutral                                             Why ?    

1           joy    Sage Act upgrade on my to do list for tommorow.   

2       sadness  ON THE WAY TO MY HOMEGIRL BABY FUNERAL!!! MAN ...   

3           joy   Such an eye ! The true hazel eye-and so brill...   

4           joy  @Iluvmiasantos ugh babe.. hugggzzz for u .!  b...   

...         ...                                                ...   

30800       joy  When I understood that I was admitted to the U...   

30801       joy    Tuesday woken up to Oscar and Cornet practice X   


30802  surprise  @MichelGW have you gift! Hope you like it! It'...   

30803       joy  The world didnt give it to me..so the world MO...   

30804      fear  Youu call it JEALOUSY, I call it of #Losing YO...   


      Predicted_Emotion  

0               neutral  

1                   joy  

2               sadness  

3                   joy  

4                   joy  

...                 ...  

30800               joy  

30801               joy  

30802          surprise  

30803               joy  

30804              fear  


[30805 rows x 3 columns]

# imported some useful functions from scikit-learn's metrics and plotting modules for evaluating and visualizing classification results. 
from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix

# Print the classification report
# classification_report: This function generates a text-based summary of the classification performance. It includes metrics such as precision, recall, F1-score, and support for each class.

print("Classification Report:")

print(classification_report(df['Emotion'], df['Predicted_Emotion']))

Classification Report:
              precision    recall  f1-score   support



       anger       0.96      0.93      0.95      3659

     disgust       0.97      0.86      0.91       667
     
        fear       0.97      0.96      0.96      4858
        
         joy       0.94      0.97      0.96     10264
     
     neutral       0.92      0.97      0.94      1966
     
     sadness       0.95      0.94      0.94      5820
     
       shame       0.99      0.97      0.98       141
    
    surprise       0.96      0.90      0.93      3430

    
    accuracy                           0.95     30805
   
   macro avg       0.96      0.94      0.95     30805

weighted avg       0.95      0.95      0.95     30805

# Print the confusion matrix
# confusion_matrix: This function computes a confusion matrix, which is a table showing the number of true positives, true negatives, false positives, and false negatives. It's a useful tool for assessing the performance of a classification algorithm.


conf_matrix = confusion_matrix(df['Emotion'], df['Predicted_Emotion'])

print("\nConfusion Matrix:")

print(conf_matrix)

Confusion Matrix:
[[ 3410     5    41    81    28    75     1    18]

 [   15   573     7    34     4    26     0     8]
 
 [   30     3  4641    96    19    47     0    22]
 
 [   20     3    37 10006    50    93     0    55]
 
 [    5     0     2    40  1901    15     0     3]
 
 [   42     3    50   203    41  5445     0    36]
 
 [    3     0     1     0     0     0   137     0]
 
 [   25     2    27   221    17    56     0  3082]]


# plot_confusion_matrix: This function is used for visualizing the confusion matrix. It provides a visual representation of how well a classifier performs.

plot_confusion_matrix(pipe_lr,x_test,y_test)

C:\Users\gayatri\anaconda3\lib\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
  warnings.warn(msg, category=FutureWarning)
<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1bf4dff9370>


# Text Extraction from image
# PIL (Pillow): The Pillow library is a powerful image processing library in Python. It provides functionalities for opening, manipulating, and saving various image file formats.
# pytesseract: This module is a Python wrapper for Google's Tesseract-OCR Engine. Tesseract is an open-source OCR engine that can recognize text in images.
from PIL import Image

import pytesseract

# Set the path to the Tesseract executable (change this to your Tesseract installation path)
# pytesseract.pytesseract.tesseract_cmd to point to the Tesseract-OCR executable on your system. This is a common practice to ensure that pytesseract can locate the Tesseract binary.
# Your function extract_english_text_from_image uses this setup to extract English text from an image. 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# This function takes the path of an image file as input, opens the image using Pillow, and then uses Tesseract (pytesseract.image_to_string) to perform OCR and extract English text from the image.
def extract_english_text_from_image(image_path):

    # Open the image file
    
    image = Image.open(image_path)



    # Use Tesseract to do OCR with English language
    
    text = pytesseract.image_to_string(image, lang='eng')



    return text

# Example usage

image_path = r'C:\Users\gayatri\Documents\Downloads\download.jfif'  # Replace with the path to your image file

extracted_english_text = extract_english_text_from_image(image_path)



print("Extracted English Text:")

print(extracted_english_text)

Extracted English Text:

“Trying to be happy by accumulating
possessions is like trying to satisfy

hunger by taping sandwiches all
over your body.”
- George Carlin


predicted_image_result=predict_emotion(extracted_english_text)

print(f"The predicted emotion for the text is: {predicted_image_result}")

The predicted emotion for the text is: joy

from PIL import Image

import pytesseract

# Set the path to the Tesseract executable (change this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# This function performs OCR on the given image using Tesseract and extracts Telugu text (lang='tel'). Ensure that Tesseract is properly configured and the Telugu language data is available.
def extract_telugu_text_from_image(image_path):

    # Open the image file
    
    image = Image.open(image_path)



    # Use Tesseract to do OCR with Telugu language
    
    text = pytesseract.image_to_string(image, lang='tel')


    return text

# Example usage
image_path = r'C:\Users\gayatri\Documents\Downloads\t4.png'

extracted_telugu_text = extract_telugu_text_from_image(image_path)

print("Extracted Telugu Text:")

print(extracted_telugu_text)

Extracted Telugu Text:

ఈరోజును!
ఒకరిపట్ల
నిర్లక్ష్యం

రేపు ఇంకొకరి నుండి.
అన్పభవిస్తే అర్థం అవుతుంది
ఎంత బాధగఉంటుందో

#  to install a specific version of the googletrans library. To install version 4.0.0-rc1, you can use the following command:
pip install googletrans==4.0.0-rc1

Requirement already satisfied: googletrans==4.0.0-rc1 in c:\users\gayatri\anaconda3\lib\site-packages (4.0.0rc1)

Requirement already satisfied: httpx==0.13.3 in c:\users\gayatri\anaconda3\lib\site-packages (from googletrans==4.0.0-rc1) (0.13.3)

Requirement already satisfied: certifi in c:\users\gayatri\anaconda3\lib\site-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (2021.10.8)

Requirement already satisfied: hstspreload in c:\users\gayatri\anaconda3\lib\site-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (2024.3.1)

Requirement already satisfied: sniffio in c:\users\gayatri\anaconda3\lib\site-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (1.2.0)

Requirement already satisfied: chardet==3.* in c:\users\gayatri\anaconda3\lib\site-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (3.0.4)

Requirement already satisfied: idna==2.* in c:\users\gayatri\anaconda3\lib\site-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (2.10)

Requirement already satisfied: rfc3986<2,>=1.3 in c:\users\gayatri\anaconda3\lib\site-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (1.5.0)

Requirement already satisfied: httpcore==0.9.* in c:\users\gayatri\anaconda3\lib\site-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (0.9.1)

Requirement already satisfied: h11<0.10,>=0.8 in c:\users\gayatri\anaconda3\lib\site-packages (from httpcore==0.9.*->httpx==0.13.3->googletrans==4.0.0-rc1) (0.9.0)

Requirement already satisfied: h2==3.* in c:\users\gayatri\anaconda3\lib\site-packages (from httpcore==0.9.*->httpx==0.13.3->googletrans==4.0.0-rc1) (3.2.0)

Requirement already satisfied: hyperframe<6,>=5.2.0 in c:\users\gayatri\anaconda3\lib\site-packages (from h2==3.*->httpcore==0.9.*->httpx==0.13.3->googletrans==4.0.0-rc1) 
(5.2.0)

Requirement already satisfied: hpack<4,>=3.0 in c:\users\gayatri\anaconda3\lib\site-packages (from h2==3.*->httpcore==0.9.*->httpx==0.13.3->googletrans==4.0.0-rc1) (3.0.0)

Note: you may need to restart the kernel to use updated packages.

[notice] A new release of pip is available: 23.2.1 -> 24.0

[notice] To update, run: python.exe -m pip install --upgrade pip


# Translator class from the googletrans library. This class allows you to translate text from one language to another using Google Translate.
from googletrans import Translator

# defined a function named translate_telugu_to_english that utilizes the Translator class to translate Telugu text to English. 
def translate_telugu_to_english(text):

    translator = Translator()
    
    translated_text = translator.translate(text, src='te', dest='en').text
    
    return translated_text

# Example usage of translating the extracted telugu text from image in to english language and predicting its emotion
telugu_text = extracted_telugu_text

english_translation = translate_telugu_to_english(telugu_text)

print(f"Telugu Text:\n {telugu_text}")

print(f"English Translation:\n {english_translation}")

predicted_timage_result=predict_emotion(english_translation)

print(f"\nThe predicted emotion for the text is: {predicted_timage_result}")

Telugu Text:
 ఈరోజును!
ఒకరిపట్ల
నిర్లక్ష్యం

రేపు ఇంకొకరి నుండి.
అన్పభవిస్తే అర్థం అవుతుంది
ఎంత బాధగఉంటుందో


English Translation:
 Today!
One with each other
Negligence

From someone else tomorrow.
It is understandable
How sad

The predicted emotion for the text is: sadness







# ## app.py file explanation

# Modules and methods used
# 1. Streamlit:

Streamlit is a Python library that simplifies the process of creating web applications for data science and machine learning.

Methods:

st.title: Sets the title of the Streamlit app.

st.subheader: Displays a subheading in the Streamlit app.

st.text_area: Creates a multiline text input area for user input.

st.button: Creates a button for user interaction.

st.columns: Splits the app into columns for layout purposes.

st.success: Displays a success message with a green background.

st.write: Writes text or data to the app.

st.file_uploader: Allows the user to upload a file.

st.error: Displays an error message with a red background.

st.sidebar: Creates a sidebar for additional controls, allowing users to interact with the app.



# 2.Pandas:

Pandas is a powerful data manipulation library for Python, providing data structures like DataFrames.

Methods:

pd.read_csv: Reads a CSV file into a DataFrame, facilitating data handling.

DataFrame.fillna: Fills missing values in a DataFrame with a specified value.

DataFrame.apply: Applies a function along the axis of the DataFrame, often used for column-wise operations.



# 3.NumPy:

NumPy is a numerical computing library in Python, providing support for large, multi-dimensional arrays and matrices.

Methods:

np.max: Returns the maximum value along a specified axis, used for finding the maximum confidence value.



# 4.Altair:

Altair is a declarative statistical visualization library for Python.

Methods:

alt.Chart: Creates a base Altair chart.

mark_bar: Marks data points with bars, used for creating bar charts.

encode: Encodes visual channels like x, y, color, etc., defining how data is mapped to visual properties.

properties: Sets properties of the chart, such as width and height.



# 5.Joblib:

Joblib is a set of tools for providing lightweight pipelining in Python.

Methods:

joblib.load: Loads a pre-trained model from a file, in this case, a machine learning model.



# 6.os:

The os module provides a way to interact with the operating system.

Methods:

os.path.dirname: Returns the directory name of a path.

os.path.abspath: Returns the absolute path of a file.

os.path.join: Joins path components into a single path.



# ## CODE EXPLANATION



import streamlit as st

import pandas as pd

import numpy as np

import altair as alt

import joblib

import os



# Get the absolute path to the current directory

# os.path.abspath(__file__):



# __file__: This is a special variable in Python that represents the path of the current script.

# os.path.abspath(__file__): Converts the script's path to an absolute path.

# os.path.dirname(os.path.abspath(__file__)):



# os.path.dirname: Returns the directory name of a path.

# os.path.dirname(os.path.abspath(__file__)): Takes the absolute path of the current script and extracts the directory portion.

# The end result, current_dir, is a variable containing the absolute path of the directory where the current script is located.



current_dir = os.path.dirname(os.path.abspath(__file__))



# Load the pre-trained model using a relative path

# os.path.join: This function is used to join one or more path components intelligently. It concatenates various path components with the correct separator ("/" or "" depending 
on the operating system).

# current_dir: The absolute path of the current script's directory obtained in the previous line.

# "model": The name of the subdirectory where the model file is stored.

# "text_emotion.pkl": The name of the pre-trained model file.

# The result, model_path, is the complete path to the pre-trained model file.



model_path = os.path.join(current_dir, "model", "text_emotion.pkl")



# open(model_path, "rb"): Opens the file specified by model_path in binary read mode.

# joblib.load: Loads a saved model from a file. It deserializes the model that was previously saved using joblib.dump.

# The variable pipe_lr now holds the pre-trained machine learning model, and it can be used for making predictions on new data.



pipe_lr = joblib.load(open(model_path, "rb"))



# This part of the code defines a dictionary emotions_emoji_dict that maps emotions to corresponding emojis. 

emotions_emoji_dict = {

    "anger": "😠", "disgust": "🤮", "fear": "😨😱",  "joy": "😂",
    
    "neutral": "😐", "sadness": "😔", "surprise": "😮"
}




# Additionally, there's a function predict_emotions that uses the pre-trained machine learning model (pipe_lr) to predict emotions for a given input text.

def predict_emotions(docx):

    results = pipe_lr.predict([docx])
    
    return results[0]



# The function get_prediction_proba is used to obtain the prediction probabilities for each emotion class for a given input text.

def get_prediction_proba(docx):

    results = pipe_lr.predict_proba([docx])
    
    return results



# The function display_emotion_rows is used to display the rows of text corresponding to a selected emotion from a DataFrame. 

def display_emotion_rows(df, selected_emotion):

    # Display rows of text for the selected emotion
    
    selected_rows = df[df['predictions'] == selected_emotion]['Text']
    
    st.write(selected_rows)



# The main function is the core of your Streamlit app. It defines the overall structure and behavior of the app.

# Sets the title and subheader for the Streamlit app.

# Creates a radio button to allow the user to choose between entering a single text or uploading a CSV file.

# If the user chooses to enter a single text, it provides a text area for input and a submit button. When submitted, it predicts the emotion and displays the result and 
prediction probability in two columns.

# Displays the original text, predicted emotion, and confidence in two columns.

# If the user chooses to upload a CSV file, it provides a file uploader. When a file is uploaded, it reads the CSV file into a DataFrame.

# Displays the uploaded dataset and checks if the 'Text' column exists in the dataset.

# Fills missing values in the 'Text' column with an empty string.

# Applies the predict_emotions function to make predictions for each text in the dataset.

# Displays the total number of rows in the dataset.

# Displays the count of each predicted emotion in the dataset.

# Creates a bar chart to visually represent the distribution of predicted emotions.

# In the sidebar, provides a selection box to choose an emotion and a button to display the corresponding text rows.

# This main function orchestrates the different components of your Streamlit app, handling user inputs, making predictions, and displaying results.



def main():

    st.title("Text Emotion Detection")
    
    st.subheader("Detect Emotions In Text")



    # Allow the user to choose between single text or CSV file
    
    option = st.radio("Choose an option", ["Single Text", "Upload CSV File"])



    if option == "Single Text":
    
        raw_text = st.text_area("Type Here")
        
        submit_text = st.button("Submit")



        if submit_text:
        
            col1, col2 = st.columns(2)
            
            prediction = predict_emotions(raw_text)
            
            probability = get_prediction_proba(raw_text)



            with col1:
            
                st.success("Original Text")
                
                st.write(raw_text)



                st.success("Prediction")
                
                emoji_icon = emotions_emoji_dict[prediction]
                
                st.write("{}:{}".format(prediction, emoji_icon))
                
                st.write("Confidence:{}".format(np.max(probability)))



            with col2:
            
                st.success("Prediction Probability")
                
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                
                proba_df_clean = proba_df.T.reset_index()
                
                proba_df_clean.columns = ["emotions", "probability"]
                
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                
                st.altair_chart(fig, use_container_width=True)



    elif option == "Upload CSV File":
    
        # Allow the user to upload a CSV file
        
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])



        if uploaded_file is not None:
        
            # Read the CSV file into a DataFrame
            
            df = pd.read_csv(uploaded_file)



            # Display the DataFrame
            
            st.write("Uploaded Dataset:")
            
            st.write(df)



            # Check if 'Text' column exists
            
            if 'Text' not in df.columns:
            
                st.error("The 'Text' column is missing in the dataset.")
                
                return



            # Fill missing values in the 'Text' column with an empty string
            
            df['Text'] = df['Text'].fillna('')



            # Make predictions for each text in the dataset
            
            df['predictions'] = df['Text'].apply(predict_emotions)



            # Display total number of reviews
            
            st.write(f"Total Number of Rows: {len(df)}")



            # Display count of all emotions
            
            st.write("Count of Emotions:")
            
            emotion_counts = df['predictions'].value_counts()
            
            st.write(emotion_counts)



            # Representation plot of emotions and their counts
            
            st.write("Representation Plot of Emotions:")
            
            chart = alt.Chart(df).mark_bar().encode(
            
                x='predictions',
                
                y='count()',
                
                color='predictions'
            ).properties(width=600, height=400)
            
            st.altair_chart(chart)



            # Display buttons for each emotion in the sidebar
           
            selected_emotion = st.sidebar.selectbox("Select Emotion", list(emotions_emoji_dict.keys()))

            if st.sidebar.button("Show Emotion Text"):
            
                # Show emotion text on another page
                
                st.write(f"Emotion Text for {selected_emotion.capitalize()}:")
                
                display_emotion_rows(df, selected_emotion)



if __name__ == '__main__':

    main()




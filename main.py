# created by ilayd on 26 may 2024
# spam detector
# import the necessary libraries
import re
import string
import pandas as pd
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.utils import resample
import tensorflow as tf
from sklearn.model_selection import train_test_split
tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
# takes: a string
# converts it to lower case
# returns: a string
def lowerCase(text):
    text = text.lower()
    return text

# takes: a string
# removes punctuations
# returns: a string
def removePunct(text):
    temp = str.maketrans('', '', string.punctuation)
    return text.translate(temp)

# takes: a string
# removes the stop words
# returns: a string
def removeStop(text):
    nonStop = []
    # stores the non-stop words
    for word in str(text).split():
        if word not in stopwords.words('english'):
            nonStop.append(word)

    output = " ".join(nonStop)
    return output

# takes: a string
# stems it
# returns a string
def stemming(text):
    lst = []
    words = word_tokenize(text)
    # stems (returns them to their root form) every word
    for word in words:
        lst.append(PorterStemmer().stem(word))

    return " ".join(lst)

# read the csv file
messages = pd.read_csv('messages.csv')

# store the hams and spams in seperate variables
ham = messages[messages.Category == "ham"]
spam = messages[messages.Category == "spam"]

# ham messages are significantly more than spam messages and we don't want
# our model to be biased towards that therefore we need to balance the
# training dataset.
#
# we do that with resample function. replace means that it does the resampling
# with replacement, n_samples is the number of samples drawn from ham (which
# in this case equals to the length of spam messages), and random_state just
# makes it reproducable, we can use any number as long as we put a number there
hamDownSample = resample(ham, replace = True, n_samples = len(spam),
                         random_state = 42)

# combines the two
data = pd.concat([hamDownSample, spam])

# lowercase all messages
data['Message'] = data['Message'].apply(lowerCase)

# remove stop words
data['Message'] = data['Message'].apply(removeStop)

# remove digits
data['Message'] = data['Message'].apply(lambda x:re.sub('[\d]','',x))

# stemming
data['Message'] = data['Message'].apply(stemming)

# converts categorical labels into numerical labels
data['Category'] = preprocessing.LabelEncoder().fit_transform(data['Category'])

data['Message_Length'] = data['Message'].apply(len)

MAX_LEN = data['Message_Length'].max()

# turns it into a numpy array
y = data['Category'].values

texts = data['Message'].values

tok = tokenizer(num_words =MAX_LEN)

tok.fit_on_texts(texts)

# converts each message into a sequence of integers
x = tok.texts_to_sequences(texts)

# this ensures that every message has the same length
x = pad_sequences(x, maxlen = MAX_LEN)

# seperates the data into testing and training
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,
                                                    random_state = 42)
# define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim = 30000, output_dim = 30), # converrts
    # integer-coded words into dense vectors of fixed size
    tf.keras.layers.LSTM(128, return_sequences = True),
    tf.keras.layers.Dropout(0.2), # this sets 20% of the input units to 0 each
    # update during training time, which helps prevent overfitting
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dropout(0.25), # this sets 25% of the input units to 0 each
    # update during training time, which helps prevent overfitting
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# compile the model
model.compile(loss = 'binary_crossentropy', optimizer =
tf.keras.optimizers.Adam(), metrics = ['accuracy'])

# this stops training the model early if a monitored metric stops improving
# to prevent overfitting
earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)

# an epoch is one complete pass through the training dataset
epochs = 10

# trains the model
history = model.fit(x_train, y_train, validation_data = (x_test, y_test),
                    epochs = epochs, callbacks = [earlyStop], verbose = 2)

model.evaluate(x_test, y_test)

# plotting
fig = go.Figure()

train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
epoch = [i + 1 for i in range(len(train_acc))]

acc_loss_df = pd.DataFrame({"Training Loss" : train_loss,
                            "Validation Loss": val_loss,
                            "Train Accuracy" : train_acc,
                            "Validation Accuracy" : val_acc,
                            "Epoch":epoch})

fig.add_trace(go.Scatter(x = acc_loss_df['Epoch'],
                         y = acc_loss_df['Train Accuracy'],
                         mode='lines+markers',
                         name='Training Accuracy'))

fig.add_trace(go.Scatter(x = acc_loss_df['Epoch'],
                         y = acc_loss_df['Validation Accuracy'],
                         mode='lines+markers',
                         name = 'Validation Accuracy'))

fig.update_layout(title = {'text':
                            "<b>Training Accuracy Vs Validation Accuracy</b>\n",
                           'xanchor': 'center',
                           'yanchor': 'top',
                           'y':0.9,'x':0.5,},
                  xaxis_title="Epoch",
                  yaxis_title = "Accuracy",
                  title_font = dict(size = 20))

fig.layout.template = 'plotly_dark'

fig.show()
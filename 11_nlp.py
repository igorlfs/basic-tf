import re
import string
from collections import Counter

import nltk
import pandas as pd
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from tensorflow import keras

# |%%--%%| <Any2RG83uZ|t5RifKZL2p>

# https://www.kaggle.com/c/nlp-getting-started : NLP Disaster Tweets
df = pd.read_csv("twitter_train.csv")

# |%%--%%| <t5RifKZL2p|GqeDRb8inN>

df.head()

# |%%--%%| <GqeDRb8inN|DgrE4BVOlV>

print((df.target == 1).sum())  # Disaster
print((df.target == 0).sum())  # No Disaster

# |%%--%%| <DgrE4BVOlV|0OarFFgpUQ>


# Preprocessing
def remove_url(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


# https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate/34294022
def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


string.punctuation

# |%%--%%| <0OarFFgpUQ|FvJCkeIXiV>


pattern = re.compile(r"https?://(\S+|www)\.\S+")
for t in df.text:
    matches = pattern.findall(t)
    for match in matches:
        print(t)
        print(match)
        print(pattern.sub(r"", t))
    if len(matches) > 0:
        break

# |%%--%%| <FvJCkeIXiV|KxD8YVNAnA>

df["text"] = df.text.map(remove_url)  # map(lambda x: remove_URL(x))
df["text"] = df.text.map(remove_punctuation)

# |%%--%%| <KxD8YVNAnA|kZ9GMpf9Ij>

# remove stopwords
# pip install nltk
nltk.download("stopwords")

# Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine
# has been programmed to ignore, both when indexing entries for searching and when retrieving them
# as the result of a search query.
stop = set(stopwords.words("english"))


# https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


# |%%--%%| <kZ9GMpf9Ij|o9X5E2hgnK>

stop

# |%%--%%| <o9X5E2hgnK|n9XfbWGCkO>

df["text"] = df.text.map(remove_stopwords)

# |%%--%%| <n9XfbWGCkO|ZQVAHcspAR>

df.text

# |%%--%%| <ZQVAHcspAR|NOGZW0iQvR>


# Count unique words
def counter_word(text_col: pd.Series):
    count = Counter()
    for text in text_col.to_numpy():
        for word in text.split():
            count[word] += 1
    return count


counter = counter_word(df.text)

# |%%--%%| <NOGZW0iQvR|zztAbmFfu8>

len(counter)

# |%%--%%| <zztAbmFfu8|YAfAG3aqrF>

counter

# |%%--%%| <YAfAG3aqrF|hCQfYxapZp>

counter.most_common(5)

# |%%--%%| <hCQfYxapZp|HPNasuVe4x>

num_unique_words = len(counter)

# |%%--%%| <HPNasuVe4x|bxx6nNvMgb>

# Split dataset into training and validation set
train_size = int(df.shape[0] * 0.8)

train_df = df[:train_size]
val_df = df[train_size:]

# split text and labels
train_sentences = train_df.text.to_numpy()
train_labels = train_df.target.to_numpy()
val_sentences = val_df.text.to_numpy()
val_labels = val_df.target.to_numpy()

# |%%--%%| <bxx6nNvMgb|zvtPAOFO4F>

train_sentences.shape, val_sentences.shape

# |%%--%%| <zvtPAOFO4F|NfzEfiALRD>

# Tokenize
# vectorize a text corpus by turning each text into a sequence of integers
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_sentences)  # fit only to training

# |%%--%%| <NfzEfiALRD|55zUisEWYm>

# each word has unique index
word_index = tokenizer.word_index

# |%%--%%| <55zUisEWYm|gAX0vATMlo>

word_index

# |%%--%%| <gAX0vATMlo|VgVaLB3nBf>

train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)

# |%%--%%| <VgVaLB3nBf|bDKgV0nlqz>

print(train_sentences[10:15])
print(train_sequences[10:15])

# |%%--%%| <bDKgV0nlqz|UMOtOQZzTV>

# Pad the sequences to have the same length
# Max number of words in a sequence
max_length = 20

train_padded = pad_sequences(
    train_sequences, maxlen=max_length, padding="post", truncating="post"
)
val_padded = pad_sequences(
    val_sequences, maxlen=max_length, padding="post", truncating="post"
)
train_padded.shape, val_padded.shape

# |%%--%%| <UMOtOQZzTV|BMwFQTUwvs>

train_padded[10]

# |%%--%%| <BMwFQTUwvs|4kwAPQ6X9U>

print(train_sentences[10])
print(train_sequences[10])
print(train_padded[10])

# |%%--%%| <4kwAPQ6X9U|u2JueSIU1y>

reverse_word_index = {idx: word for (word, idx) in word_index.items()}

# |%%--%%| <u2JueSIU1y|HMccd21KF1>

reverse_word_index

# |%%--%%| <HMccd21KF1|btnhDiOCgb>


def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])


# |%%--%%| <btnhDiOCgb|w9a9dZ61X8>

decoded_text = decode(train_sequences[10])

print(train_sequences[10])
print(decoded_text)


# |%%--%%| <w9a9dZ61X8|AnUYS21Isg>

# Create LSTM model
# Embedding: https://www.tensorflow.org/tutorials/text/word_embeddings
# Turns positive integers (indexes) into dense vectors of fixed size. (other approach could be one-hot-encoding)

# Word embeddings give us a way to use an efficient, dense representation in which similar words have
# a similar encoding. Importantly, you do not have to specify this encoding by hand. An embedding is a
# dense vector of floating point values (the length of the vector is a parameter you specify).

model = keras.models.Sequential()
model.add(layers.Embedding(num_unique_words, 32, input_length=max_length))

# The layer will take as input an integer matrix of size (batch, input_length),
# and the largest integer (i.e. word index) in the input should be no larger than num_words (vocabulary size).
# Now model.output_shape is (None, input_length, 32), where `None` is the batch dimension.


model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

# |%%--%%| <AnUYS21Isg|zXITZXzNQz>

loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = keras.metrics.BinaryAccuracy()

model.compile(loss=loss, optimizer=optim, metrics=metrics)


# |%%--%%| <zXITZXzNQz|K6zyg7AKsl>

model.fit(
    train_padded,
    train_labels,
    epochs=20,
    validation_data=(val_padded, val_labels),
    verbose=2,
)

# |%%--%%| <K6zyg7AKsl|EHo1anq48p>

predictions = model.predict(train_padded)
predictions = [1 if p > 0.5 else 0 for p in predictions]  # noqa

# |%%--%%| <EHo1anq48p|qk6M4mU8lz>

print(train_sentences[10:20])

print(train_labels[10:20])
print(predictions[10:20])

# |%%--%%| <qk6M4mU8lz|FMZqOmjNNW>

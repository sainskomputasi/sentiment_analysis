# Sentiment Analysis against the Community Activities Restrictions Enforcement (PPKM) Data using ANN (Artificial Neural Networks)
Today, social networking media is becoming popular as a powerful tool for influencing people and sharing their views with the general public. For example, Twitter is one of the social media channels that Indonesian politicians often use for political campaigns. Of the several social networking sites, Twitter is one of the most popular tools used by politicians. Twitter has been used by many Indonesian politicians to evoke public sympathy and increase popularity before the election. Politicians use Twitter to post their opinions, thoughts, and activities to attract and influence those who vote in parliamentary elections. In this study, sentiment analysis will be carried out using deep learning (ANN) with tweets dataset related to the enforcement of community activity restrictions (PPKM) in Indonesia. There are several steps to do sentiment analysis using ANN (Artificial Neural Networks), such as **Preprocessing Data**, **Training Data**, and **Data Vizualization.**
## Requirements
1. OS Windows / Linux 
2. Jupyter Notebook and Python
3. Pandas, JCopML, SKLearn, TextBlob, VADER, and Pytorch
## Preprocessing Data
1. Prepare the Dataset (The Dataset is Attached) 
```
import pandas as pd
df = pd.read_csv("Data01.csv", index_col="no")
```
2. Stemming and Character Removal
```
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Remove user mention
def func(string):

    strlist = string.split()
    strlistnew = []
    for i in strlist:
        if i[0] == '@' or 'http' or '#' in i:
            pass
        if i.isalpha():
            strlistnew.append(i)  
    new = ' '.join(strlistnew)   
    stringNew   = stemmer.stem(new)    
    return stringNew

df['text'] = df['text'].apply(func)
df['text'] = df['text'].str.lower()
```
3. Translate Dataset using TextBlob
```
from textblob import TextBlob

def func2(string):
    
    kata = TextBlob(string)
    hasil = kata.translate(from_lang='id', to='en')
    
    return str(hasil)

df['translate'] = df['text'].apply(func2)
df
```
4. Labelling Dataset using VADER
```
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def func(string):
    result = analyzer.polarity_scores(string)
    num = result['compound']
    if num < 0:
        polar = 0
    elif num > 0.3:
        polar = 2
    else:
        polar = 1
    return polar

df['vaderAnalysis'] = df['translate'].apply(func)
df[['text','translate','average','vaderAnalysis']]
```
5. Vader Labelling Result
```
correct = 0
count = 0
for index, row in df.iterrows():
    count += 1
    if row['vaderAnalysis'] == row['average']:
        correct += 1
print(correct/count)
```
## Training Data
1. Training Preparation using ANN
```
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease

import torch
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```
2. Data Review
```
df.count()
plot_missing_value(df, return_df=True)
df.average.value_counts(normalize=True)
df.vaderAnalysis.value_counts(normalize=True)
```
3. Training Preparation
```
X = df.text.values
y = df.average.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1234)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```
```
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
sw_indo = stopwords.words("indonesian") + list(punctuation)
```
stop words removal, tokenization, and tf-idf vectorizer
```
preprocessor = Pipeline([
    ('prep', TfidfVectorizer(tokenizer=word_tokenize, stop_words=sw_indo)), #, max_features=500, ngram_range=(1,2)
])
```
```
X_train = preprocessor.fit_transform(X_train).toarray()
X_test = preprocessor.transform(X_test).toarray()
```
```
preprocessor['prep'].get_feature_names()
```
```
from torch.utils.data import DataLoader, TensorDataset
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)

bs = 128
train_set = TensorDataset(X_train, y_train)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=8)

test_set = TensorDataset(X_test, y_test)
testloader = DataLoader(test_set, batch_size=bs, num_workers=8)
len(train_set), len(test_set), len(trainloader), len(testloader)
```
4. ANN Architecture and Configuration
```
from jcopdl.layers import linear_block
class FashionClassfier(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            linear_block(input_size, 64, dropout=dropout),
            linear_block(64, 64, dropout=dropout),
            linear_block(64, output_size, activation="lsoftmax") #, activation="lsoftmax"
        )
         
    def forward(self, x):
        return self.fc(x) 
```
```
from jcopdl.callback import Callback, set_config

config = set_config({
    "input_size": X_train.shape[1],
    "output_size": 3,
    "dropout": 0.2
})
```
Training Config and MCOC
```
model = FashionClassfier(config.input_size, config.output_size, config.dropout).to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)
callback = Callback(model, config, outdir="SentimenModel", early_stop_patience=10)

print(model.train())
print(model.eval())
```
5. Training Model
```
from tqdm.auto import tqdm

def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device):   
    if mode == "train":
        model.train()
    elif mode == "test":
        model.eval()
    cost = correct = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)
        
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()
    cost = cost / len(dataset)
    acc = correct / len(dataset)
    return cost, acc
```
```
while True:
    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
    with torch.no_grad():
        test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)
    
    # Logging
    callback.log(train_cost, test_cost, train_score, test_score)
    
    # Checkpoint
    callback.save_checkpoint()
    
    # Runtime Plotting
    callback.cost_runtime_plotting()
    callback.score_runtime_plotting()
    
    # Early Stopping
    if callback.early_stopping(model, monitor="test_score"):
        callback.plot_cost()
        callback.plot_score()
        break
```
## Data Vizualization
1. Confusion Matrix
```
from sklearn.metrics import confusion_matrix
import seaborn as sns
```
```
with torch.no_grad():
    model.train()
    output_train = model(X_train)

pred_train = output_train.argmax(1).numpy()
pred_train

cf_matrix_train = confusion_matrix(y_train, pred_train)
print(cf_matrix_train)

ax1 = sns.heatmap(cf_matrix_train/np.sum(cf_matrix_train), annot=True, 
            fmt='.2%', cmap='Blues')

ax1.set_title('Train Model Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

ax1.xaxis.set_ticklabels(['negative', 'netral', 'positive'])
ax1.yaxis.set_ticklabels(['negative', 'netral', 'positive'])
```
2. Word Cloud Vizualization
```
from wordcloud import WordCloud
text = " ".join(review for review in df.text)
# Generate a word cloud image
wordcloud = WordCloud(stopwords=sw_indo, background_color="white",max_words=100).generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

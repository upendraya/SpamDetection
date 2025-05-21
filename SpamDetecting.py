# Spam Email Detection Project - Complete Corrected Code

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from collections import Counter

# Sklearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score

# WordCloud for visualization
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download('punkt_tab')       # for tokenizers
nltk.download('stopwords')   # for stopwords

# Load the dataset
df = pd.read_csv(r'D:\Projects\SpamDetection\dataset\spam.csv', encoding='latin1')

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Rename columns for clarity
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target labels: ham -> 0, spam -> 1
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicates
df = df.drop_duplicates(keep='first')

# Ensure text column is string and fill NA if any
df['text'] = df['text'].astype(str).fillna('')

# Text length and structure analysis
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(word_tokenize(x)))
df['num_sentence'] = df['text'].apply(lambda x: len(sent_tokenize(x)))

# Display basic stats
print(df[['num_characters', 'num_words', 'num_sentence']].describe())

# Summary stats for ham (target=0)
print("Ham messages stats:")
print(df[df['target'] == 0][['num_characters', 'num_words', 'num_sentence']].describe())

# Summary stats for spam (target=1)
print("Spam messages stats:")
print(df[df['target'] == 1][['num_characters', 'num_words', 'num_sentence']].describe())

# Plot distribution of number of characters by target
plt.figure(figsize=(10, 6))
sns.histplot(df[df['target'] == 0]['num_characters'], color='blue', label='Ham (0)', kde=True)
sns.histplot(df[df['target'] == 1]['num_characters'], color='red', label='Spam (1)', kde=True)
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.title('Character Length Distribution by Target')
plt.legend()
plt.savefig("images/example_plot1.png")
plt.show()

# Plot distribution of number of words by target
plt.figure(figsize=(10, 6))
sns.histplot(df[df['target'] == 0]['num_words'], color='blue', label='Ham (0)', kde=True)
sns.histplot(df[df['target'] == 1]['num_words'], color='red', label='Spam (1)', kde=True)
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.title('Word Count Distribution by Target')
plt.legend()
plt.savefig("images/example_plot2.png")
plt.show()


# Pairplot visualization
sns.set(style='ticks', color_codes=True)
pairplot = sns.pairplot(df, hue='target', diag_kind='kde', markers=["o", "s"])
pairplot.fig.suptitle("Pairplot of Data by Target", fontsize=16, fontweight='bold')
plt.subplots_adjust(top=0.9)
plt.savefig("images/example_plot3.png")

plt.show()


# Correlation Heatmap
corr = df[['target', 'num_characters', 'num_words', 'num_sentence']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Text preprocessing function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    stems = [ps.stem(word) for word in tokens]
    return " ".join(stems)

# Apply preprocessing
df['transformed_text'] = df['text'].apply(transform_text)

# WordCloud for spam messages
spam_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(
    df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(spam_wc)
plt.axis('off')
plt.title('Word Cloud for Spam Messages')
plt.show()



# WordCloud for ham messages
ham_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(
    df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(ham_wc)
plt.axis('off')
plt.title('Word Cloud for Ham Messages')
plt.show()



# Top 25 words in spam messages
spam_words = []
for sentence in df[df['target'] == 1]['transformed_text']:
    spam_words.extend(sentence.split())

spam_word_freq = pd.DataFrame(Counter(spam_words).most_common(25), columns=['Word', 'Count'])

plt.figure(figsize=(12,6))
sns.barplot(data=spam_word_freq, x='Word', y='Count', palette='bright')
plt.xticks(rotation=90)
plt.title('Top 25 Words in Spam Messages')
plt.show()



# Top 25 words in ham messages
ham_words = []
for sentence in df[df['target'] == 0]['transformed_text']:
    ham_words.extend(sentence.split())

ham_word_freq = pd.DataFrame(Counter(ham_words).most_common(25), columns=['Word', 'Count'])

plt.figure(figsize=(12,6))
sns.barplot(data=ham_word_freq, x='Word', y='Count', palette='cool')
plt.xticks(rotation=90)
plt.title('Top 25 Words in Ham Messages')
plt.show()


# Feature Extraction using TF-IDF
tfid = TfidfVectorizer(max_features=3000)
X = tfid.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

# Initialize models
models = {
    'SVC': SVC(kernel="sigmoid", gamma=1.0),
    'KNN': KNeighborsClassifier(),
    'NB': MultinomialNB(),
    'DT': DecisionTreeClassifier(max_depth=5),
    'LR': LogisticRegression(solver='liblinear', penalty='l1'),
    'RF': RandomForestClassifier(n_estimators=50, random_state=2),
    'Adaboost': AdaBoostClassifier(n_estimators=50, random_state=2),
    'Bagging': BaggingClassifier(n_estimators=50, random_state=2),
    'ETC': ExtraTreesClassifier(n_estimators=50, random_state=2),
    'GBDT': GradientBoostingClassifier(n_estimators=50, random_state=2),
    'XGB': XGBClassifier(n_estimators=50, random_state=2, use_label_encoder=False, eval_metric='logloss')
}

# Training and evaluation function
def train_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    return acc, prec

# Train and evaluate all models
for name, model in models.items():
    accuracy, precision = train_evaluate(model, X_train, y_train, X_test, y_test)
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print('-'*30)

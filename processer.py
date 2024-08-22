import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

# Load your text data into a list
text_data = ["The quick brown fox jumps over the lazy dog",
    "The sun is shining brightly in the clear blue sky",
    "The cat purrs contentedly on my lap",
    "The baby laughs and plays with the colorful toys",
    "The flowers are blooming beautifully in the garden",
    "The dog runs quickly through the green grass",
    "The music is playing softly in the background",
    "The kids are playing happily in the park",
    "The food is delicious and flavorful at the restaurant",
    "The book is interesting and informative to read"]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to your text data and transform it into numerical features
tfidf = vectorizer.fit_transform(text_data)

# Create a Latent Dirichlet Allocation (LDA) model
lda = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tfidf)

# Get the topic weights for each feature
topic_weights = lda.components_

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Create a dictionary to store the topic words
topic_words = {}

# Iterate over the topics
for topic_idx, topic in enumerate(topic_weights):
    # Get the top 10 words for this topic
    top_words = np.argsort(topic)[::-1][:10]
    # Create a string of the top words
    topic_string = ' '.join([feature_names[i] for i in top_words])
    # Add the topic string to the dictionary
    topic_words['Topic {}'.format(topic_idx+1)] = topic_string

# Create a word cloud for each topic
for topic, words in topic_words.items():
    wordcloud = WordCloud(width = 800, height = 800, random_state=21, max_font_size = 110).generate(words)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(topic)
    plt.show()

# Use Seaborn to visualize the topic weights
sns.set()
sns.clustermap(topic_weights, cmap='viridis', figsize=(10, 10))
plt.show()
#کد برای پردازش قوی با کمک هوش مصنوعی
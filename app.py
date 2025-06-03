import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import altair as alt

# Set page config
st.set_page_config(page_title="Steam Review Sentiment", layout="wide")

# Title
st.title("'Stardew Valley' Steam Game Review Sentiment Dashboard")
st.markdown("This dashboard analyzes user reviews using Hugging Face transformers for sentiment classification and visualizes word frequency. Additional plots to analyze the efficacy of the sentiment classification are also shown.")

# Methodology
st.subheader("Methodology Overview")
st.markdown("""
### Review Collection via Steam JSON API

Instead of traditional web scraping, we retrieved game reviews using **Steam's JSON API**. This allowed for structured and efficient access to review data.

**Steps Taken:**
- Made HTTP requests to Steam’s review API using the game's `appID`.
- Extracted important fields such as:
  - `review`: The full review text.
  - `voted_up`: Whether the review was marked helpful by the user.
- Filtered out reviews with missing or short text (under 20 characters).

This provided a clean dataset ready for sentiment analysis.

---

### Sentiment Analysis with DistilBERT

We used a **pretrained transformer model** from Hugging Face to analyze sentiment.

**Model Used:**
- `distilbert-base-uncased-finetuned-sst-2-english`
  - A smaller, faster version of BERT.
  - Trained to classify text as either **POSITIVE** or **NEGATIVE**.

**Steps Taken:**
1. Tokenized each review using the model’s tokenizer.
2. Ran the model to get sentiment predictions and confidence scores.
3. Stored these results in the DataFrame for further analysis and visualization.

This approach allowed us to apply deep learning-based sentiment classification without needing to manually label any data.
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('stardew_valley_reviews.csv')  
df_clean = load_data()

# Sentiment Distribution
st.subheader("Interactive Sentiment Distribution")
sentiment_counts = df_clean['sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
chart = alt.Chart(sentiment_counts).mark_bar().encode(
    x='Sentiment',
    y='Count',
    color='Sentiment'
)
st.altair_chart(chart, use_container_width=True)

st.markdown("""The above chart displays the count of positive/negative sentiment scores identified by the model. It shows that the majority of user reviews are positive. This may reflect a general satisfaction among players, 
as Stardew Valley is generally well-received, or possibly a bias in review submission (players are more likely to write a review if they enjoyed the game).""")
# Prepare stopwords
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))
stop_words.extend(['game', 'steam', 'play', 'player', '10', 'stardew', 'valley', 'farm'])

# Word cloud generation
def tfidf_wordcloud(reviews):
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.asarray(X.sum(axis=0)).ravel()
    tfidf_dict = dict(zip(feature_names, tfidf_scores))
    return tfidf_dict

# Create side-by-side word clouds
st.subheader("Word Clouds by Sentiment")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Positive Reviews**")
    positive_reviews = df_clean[df_clean['sentiment'] == 'POSITIVE']['review']
    pos_tfidf = tfidf_wordcloud(positive_reviews)
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate_from_frequencies(pos_tfidf)
    fig1, ax1 = plt.subplots()
    ax1.imshow(wordcloud_pos, interpolation='bilinear')
    ax1.axis('off')
    st.pyplot(fig1)

with col2:
    st.markdown("**Negative Reviews**")
    negative_reviews = df_clean[df_clean['sentiment'] == 'NEGATIVE']['review']
    neg_tfidf = tfidf_wordcloud(negative_reviews)
    wordcloud_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate_from_frequencies(neg_tfidf)
    fig2, ax2 = plt.subplots()
    ax2.imshow(wordcloud_neg, interpolation='bilinear')
    ax2.axis('off')
    st.pyplot(fig2)

st.markdown("""The word clouds above visualize the top TF-IDF weighted words (in non-technical terms, loosely, the key words shared most across common sentiments) from positive and negative reviews, giving a sense of what topics users emphasize.
Note that the model's sentiment analysis flags words such as 'hours', and 'addiction' negatively even though players tend to use these terms in a positive context regarding how much time they have played the game for. We will see in future analyses
that this results in some thumbs-up reviews being flagged with negative sentiment.""")

st.subheader("Additional Visualizations")
# Interactive Sentiment Confidence Histogram
st.markdown("### Interactive: Sentiment Confidence by Class")

confidence_chart = alt.Chart(df_clean).mark_bar(opacity=0.7).encode(
    x=alt.X('confidence:Q', bin=alt.Bin(maxbins=30), title='Model Confidence Score'),
    y=alt.Y('count():Q', title='Number of Reviews'),
    color=alt.Color('sentiment:N', legend=alt.Legend(title="Sentiment")),
    tooltip=['sentiment:N', 'count():Q']
).properties(
    width=700,
    height=400
).interactive()

st.altair_chart(confidence_chart, use_container_width=True)
st.markdown("""This plot shows the distribution of sentiment confidence scores. Notice how the model is highly confident (> 0.95) for most predictions. However, it 
should be noted that lower confidence scores are assigned to proportionally more negative reviews than positive.""")

# Interactive Sentiment vs. Voted Up/Down
st.markdown("### Interactive: Sentiment vs. Voted Up/Down")

# Melt data for Altair compatibility
voted_df = df_clean.copy()
voted_df['voted_label'] = voted_df['voted_up'].replace({True: 'Voted Up', False: 'Voted Down'})

voted_chart = alt.Chart(voted_df).mark_bar().encode(
    x=alt.X('sentiment:N', title='Sentiment'),
    y=alt.Y('count():Q', title='Number of Reviews'),
    color=alt.Color('voted_label:N', title='Voted'),
    tooltip=['sentiment:N', 'voted_label:N', 'count():Q']
).properties(
    width=600,
    height=400
).interactive()

st.altair_chart(voted_chart, use_container_width=True)
st.markdown("""This plot shows the analyzed sentiment of each review against the actual thumbs-up or thumbs-down rating ascribed by the review. Although all of the thumbs-down reviews were
correctly identified as negative sentiment, this chart seems to imply that the model 'casts too wide of a net,' erroneously flagging some thumbs-up reviews with negative sentiment for the reasons outlined prior.""")

# Optional: Show some reviews
st.subheader("Sample Reviews")
n = st.slider("Select number of reviews to display", 5, 50, 10)
st.dataframe(df_clean.sample(n)[["review", "sentiment"]])

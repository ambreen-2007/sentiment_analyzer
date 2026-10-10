import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="📊", layout="wide")

st.title("📊 AI Customer Feedback Sentiment Analyzer")
st.write("Analyze customer reviews, feedback, or tweets instantly using Natural Language Processing (NLP).")

# Sidebar
st.sidebar.header("About Project")
st.sidebar.info("This app uses NLP and TextBlob to evaluate the emotional tone of customer feedback, helping businesses track user satisfaction.")

# Tabs for Single Text and Batch Processing
tab1, tab2 = st.tabs(["📝 Single Review Analysis", "📁 Bulk CSV Analysis"])

# TAB 1: Single Review
with tab1:
    user_review = st.text_area("Enter customer review or feedback below:", placeholder="Type something like: 'The product quality is amazing and delivery was super fast!'")

    if st.button("Analyze Sentiment"):
        if user_review.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            blob = TextBlob(user_review)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                st.success("Result: **Positive 😊**")
            elif polarity < -0.1:
                st.error("Result: **Negative 😞**")
            else:
                st.info("Result: **Neutral 😐**")
                
            st.write(f"**Polarity Score:** {polarity:.2f} (Ranges from -1.0 [Negative] to +1.0 [Positive])")

# TAB 2: Bulk CSV Analysis & Graphs
with tab2:
    st.subheader("Upload a CSV file containing reviews")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head())
        
        # Select Column
        text_column = st.selectbox("Select the column that contains reviews/feedback:", df.columns)
        
        if st.button("Process CSV"):
            def get_sentiment(text):
                analysis = TextBlob(str(text))
                if analysis.sentiment.polarity > 0.1:
                    return "Positive"
                elif analysis.sentiment.polarity < -0.1:
                    return "Negative"
                else:
                    return "Neutral"
            
            # Apply Sentiment Analysis
            df['Sentiment'] = df[text_column].apply(get_sentiment)
            
            st.success("Analysis Complete!")
            st.write("### Processed Results:")
            st.dataframe(df)
            
            # Plotting Graph
            st.write("### Sentiment Distribution Chart:")
            sentiment_counts = df['Sentiment'].value_counts()
            
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#2ca02c', '#d62728', '#1f77b4'], startangle=90)
            ax.axis('equal')
            
            st.pyplot(fig)
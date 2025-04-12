import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from wordcloud import WordCloud
import io
import base64
import PyPDF2
from collections import Counter

# Define the Mistral model we'll be using for the LLM tasks
# Note: this is just a reference, as we'll need HuggingFace access token for actual implementation
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# Download necessary NLTK data
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

def render_nlp_analysis(datasets):
    st.title("NLP & Large Language Models")
    
    st.markdown("""
    This section provides tools for natural language processing (NLP) and large language model (LLM) analysis.
    You can analyze text data, extract insights, and utilize state-of-the-art language models.
    """)
    
    # Data source options
    data_source = st.radio(
        "Select text data source:",
        ["From Dataset Column", "From PDF Document", "Manual Text Input"]
    )
    
    text_data = None
    
    if data_source == "From Dataset Column":
        # Check if data is loaded
        if datasets['current_df'] is None:
            st.error("No dataset loaded. Please upload or select a dataset in the 'Data Upload & Exploration' section.")
            return
        
        df = datasets['current_df']
        
        # Get text columns (string/object type)
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not text_columns:
            st.error("No text columns found in the current dataset.")
            return
        
        # Select text column
        text_column = st.selectbox("Select text column:", text_columns)
        
        # Preview text data
        st.write("Text data preview:")
        st.dataframe(df[text_column].head())
        
        # Get text data
        text_data = df[text_column].astype(str).tolist()
        
        # Filter out empty/NA values
        text_data = [text for text in text_data if text and text.strip() and text.lower() != 'nan']
        
        st.write(f"Total text entries: {len(text_data)}")
    
    elif data_source == "From PDF Document":
        # Check if budget data path is available
        if datasets['budget_data_path'] is None:
            st.error("Budget PDF document path not found.")
            
            # Offer upload option
            uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
            
            if uploaded_file is not None:
                try:
                    # Read PDF content
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text_content = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        text_content += pdf_reader.pages[page_num].extract_text()
                    
                    # Split by paragraphs
                    text_data = [p.strip() for p in text_content.split('\n\n') if p.strip()]
                    
                    st.success(f"Successfully extracted {len(text_data)} text segments from the PDF")
                    
                    # Preview extracted text
                    st.write("Extracted text preview:")
                    st.text(text_content[:1000] + "..." if len(text_content) > 1000 else text_content)
                    
                except Exception as e:
                    st.error(f"Error extracting text from PDF: {str(e)}")
        else:
            try:
                # Read from the budget PDF path
                with open(datasets['budget_data_path'], 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = ""
                    
                    # Extract text from the first 30 pages (for performance)
                    pages_to_extract = min(30, len(pdf_reader.pages))
                    
                    for page_num in range(pages_to_extract):
                        text_content += pdf_reader.pages[page_num].extract_text()
                
                # Split by paragraphs
                text_data = [p.strip() for p in text_content.split('\n\n') if p.strip()]
                
                st.success(f"Successfully extracted {len(text_data)} text segments from the budget PDF document")
                
                # Preview extracted text
                st.write("Extracted text preview:")
                st.text(text_content[:1000] + "..." if len(text_content) > 1000 else text_content)
                
            except Exception as e:
                st.error(f"Error extracting text from budget PDF: {str(e)}")
    
    elif data_source == "Manual Text Input":
        text_input = st.text_area(
            "Enter text for analysis:",
            height=200,
            help="Enter or paste the text you want to analyze"
        )
        
        if text_input:
            # Split into paragraphs for analysis
            text_data = [p.strip() for p in text_input.split('\n\n') if p.strip()]
            
            if not text_data:
                # If no paragraph breaks, use the whole text
                text_data = [text_input]
        else:
            st.warning("Please enter some text to analyze.")
            return
    
    # Check if we have text data
    if not text_data or len(text_data) == 0:
        st.warning("No text data available for analysis.")
        return
    
    # Display the number of text documents/entries
    st.write(f"Analyzing {len(text_data)} text documents/entries")
    
    # NLP Tools Selection
    st.subheader("Select NLP Analysis Tool")
    
    nlp_tools = [
        "Text Preprocessing & Statistics",
        "Word Frequency Analysis",
        "Topic Modeling",
        "Sentiment Analysis",
        "Text Classification",
        "Named Entity Recognition"
    ]
    
    selected_tool = st.selectbox("Choose analysis tool:", nlp_tools)
    
    # 1. Text Preprocessing & Statistics
    if selected_tool == "Text Preprocessing & Statistics":
        st.subheader("Text Preprocessing & Statistics")
        
        st.markdown("""
        This tool helps you preprocess text data and extract basic statistics such as word count,
        character count, and readability metrics.
        """)
        
        # Preprocessing options
        st.write("Select preprocessing options:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lowercase = st.checkbox("Convert to lowercase", value=True)
            remove_punctuation = st.checkbox("Remove punctuation", value=True)
            remove_numbers = st.checkbox("Remove numbers", value=False)
        
        with col2:
            remove_stopwords = st.checkbox("Remove stopwords", value=True)
            lemmatize = st.checkbox("Lemmatize words", value=False)
            stem = st.checkbox("Stem words", value=False)
        
        # Function to preprocess text
        def preprocess_text(text, lowercase=True, remove_punct=True, remove_nums=False,
                           remove_stops=True, lemmatize_words=False, stem_words=False):
            if lowercase:
                text = text.lower()
            
            if remove_punct:
                text = text.translate(str.maketrans('', '', string.punctuation))
            
            if remove_nums:
                text = re.sub(r'\d+', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            if remove_stops:
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stop_words]
            
            if lemmatize_words:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            if stem_words:
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(word) for word in tokens]
            
            return tokens
        
        # Process text data
        if st.button("Preprocess Text"):
            with st.spinner("Preprocessing text data..."):
                try:
                    processed_texts = []
                    text_stats = []
                    
                    for i, text in enumerate(text_data):
                        # Get original text stats
                        original_word_count = len(word_tokenize(text))
                        original_char_count = len(text)
                        
                        # Preprocess
                        tokens = preprocess_text(
                            text, 
                            lowercase=lowercase,
                            remove_punct=remove_punctuation,
                            remove_nums=remove_numbers,
                            remove_stops=remove_stopwords,
                            lemmatize_words=lemmatize,
                            stem_words=stem
                        )
                        
                        processed_text = ' '.join(tokens)
                        processed_texts.append(processed_text)
                        
                        # Calculate stats
                        word_count = len(tokens)
                        char_count = len(processed_text)
                        avg_word_length = char_count / word_count if word_count > 0 else 0
                        unique_words = len(set(tokens))
                        lexical_diversity = unique_words / word_count if word_count > 0 else 0
                        
                        # Add to stats list
                        text_stats.append({
                            'Document': i + 1,
                            'Original Word Count': original_word_count,
                            'Original Char Count': original_char_count,
                            'Processed Word Count': word_count,
                            'Processed Char Count': char_count,
                            'Avg Word Length': avg_word_length,
                            'Unique Words': unique_words,
                            'Lexical Diversity': lexical_diversity
                        })
                    
                    # Show overall statistics
                    st.subheader("Text Statistics")
                    
                    stats_df = pd.DataFrame(text_stats)
                    
                    # Overall stats
                    st.write("Overall Statistics:")
                    
                    total_docs = len(text_stats)
                    total_orig_words = stats_df['Original Word Count'].sum()
                    total_orig_chars = stats_df['Original Char Count'].sum()
                    total_proc_words = stats_df['Processed Word Count'].sum()
                    total_proc_chars = stats_df['Processed Char Count'].sum()
                    avg_words_per_doc = total_proc_words / total_docs
                    
                    st.write(f"Total Documents: {total_docs}")
                    st.write(f"Total Original Words: {total_orig_words}")
                    st.write(f"Total Original Characters: {total_orig_chars}")
                    st.write(f"Total Processed Words: {total_proc_words}")
                    st.write(f"Total Processed Characters: {total_proc_chars}")
                    st.write(f"Average Words per Document: {avg_words_per_doc:.2f}")
                    
                    # Display stats table
                    st.write("Document-level Statistics:")
                    st.dataframe(stats_df)
                    
                    # Visualize statistics
                    st.subheader("Text Statistics Visualization")
                    
                    # Word count distribution
                    fig = px.histogram(
                        stats_df,
                        x='Processed Word Count',
                        nbins=20,
                        title='Distribution of Word Counts'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Lexical diversity
                    fig = px.scatter(
                        stats_df,
                        x='Processed Word Count',
                        y='Lexical Diversity',
                        title='Lexical Diversity vs. Word Count',
                        hover_data=['Document']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display processed text samples
                    st.subheader("Processed Text Samples")
                    
                    for i in range(min(5, len(processed_texts))):
                        with st.expander(f"Document {i+1}"):
                            st.write("Original text:")
                            st.text(text_data[i][:500] + "..." if len(text_data[i]) > 500 else text_data[i])
                            st.write("Processed text:")
                            st.text(processed_texts[i][:500] + "..." if len(processed_texts[i]) > 500 else processed_texts[i])
                
                except Exception as e:
                    st.error(f"Error during text preprocessing: {str(e)}")
    
    # 2. Word Frequency Analysis
    elif selected_tool == "Word Frequency Analysis":
        st.subheader("Word Frequency Analysis")
        
        st.markdown("""
        Analyze the frequency of words in your text data. This tool helps identify the most common
        words and visualize them using word clouds and bar charts.
        """)
        
        # Preprocessing options
        st.write("Select preprocessing options:")
        
        lowercase = st.checkbox("Convert to lowercase", value=True)
        remove_punctuation = st.checkbox("Remove punctuation", value=True)
        remove_stopwords = st.checkbox("Remove stopwords", value=True)
        
        # Number of top words to show
        top_n = st.slider("Number of top words to show:", 10, 100, 30)
        
        # Custom stopwords
        custom_stopwords = st.text_input(
            "Add custom stopwords (comma-separated):",
            help="Enter additional words to exclude, separated by commas"
        )
        
        custom_stop_list = [word.strip() for word in custom_stopwords.split(',')] if custom_stopwords else []
        
        # Process text and analyze frequencies
        if st.button("Analyze Word Frequencies"):
            with st.spinner("Analyzing word frequencies..."):
                try:
                    # Combine all text data
                    all_text = ' '.join(text_data)
                    
                    # Preprocess
                    if lowercase:
                        all_text = all_text.lower()
                    
                    if remove_punctuation:
                        all_text = all_text.translate(str.maketrans('', '', string.punctuation))
                    
                    # Tokenize
                    tokens = word_tokenize(all_text)
                    
                    # Remove stopwords
                    if remove_stopwords:
                        stop_words = set(stopwords.words('english'))
                        
                        # Add custom stopwords
                        if custom_stop_list:
                            stop_words.update(custom_stop_list)
                        
                        tokens = [word for word in tokens if word not in stop_words]
                    
                    # Count word frequencies
                    word_freq = Counter(tokens)
                    
                    # Convert to dataframe
                    word_freq_df = pd.DataFrame(
                        word_freq.most_common(top_n),
                        columns=['Word', 'Frequency']
                    )
                    
                    # Display word frequency table
                    st.subheader("Word Frequency Table")
                    st.dataframe(word_freq_df)
                    
                    # Visualize word frequencies
                    st.subheader("Word Frequency Visualization")
                    
                    # Bar chart
                    fig = px.bar(
                        word_freq_df.head(30),
                        x='Word',
                        y='Frequency',
                        title='Top 30 Words by Frequency'
                    )
                    
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Word cloud
                    st.subheader("Word Cloud")
                    
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        max_words=100,
                        contour_width=3,
                        contour_color='steelblue'
                    ).generate(' '.join(tokens))
                    
                    # Convert to image
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    plt.tight_layout()
                    
                    # Display wordcloud
                    st.pyplot(fig)
                    
                    # N-gram analysis
                    st.subheader("N-gram Analysis")
                    
                    # Select n-gram size
                    n_gram_size = st.selectbox("Select n-gram size:", [2, 3, 4], index=0)
                    
                    # Function to generate n-grams
                    def generate_ngrams(tokens, n):
                        return zip(*[tokens[i:] for i in range(n)])
                    
                    # Generate and count n-grams
                    ngrams = list(generate_ngrams(tokens, n_gram_size))
                    ngram_freq = Counter([' '.join(gram) for gram in ngrams])
                    
                    # Convert to dataframe
                    ngram_freq_df = pd.DataFrame(
                        ngram_freq.most_common(top_n),
                        columns=[f'{n_gram_size}-gram', 'Frequency']
                    )
                    
                    # Display n-gram frequency table
                    st.write(f"Top {top_n} {n_gram_size}-grams:")
                    st.dataframe(ngram_freq_df)
                    
                    # Bar chart for n-grams
                    fig = px.bar(
                        ngram_freq_df.head(20),
                        x=f'{n_gram_size}-gram',
                        y='Frequency',
                        title=f'Top 20 {n_gram_size}-grams by Frequency'
                    )
                    
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error during word frequency analysis: {str(e)}")
    
    # 3. Topic Modeling
    elif selected_tool == "Topic Modeling":
        st.subheader("Topic Modeling")
        
        st.markdown("""
        Topic modeling helps discover abstract topics in a collection of documents.
        This tool uses algorithms like Latent Dirichlet Allocation (LDA) to identify topics.
        """)
        
        # Select vectorization method
        vectorization = st.radio(
            "Select vectorization method:",
            ["Count Vectorizer", "TF-IDF Vectorizer"]
        )
        
        # Select modeling algorithm
        algorithm = st.radio(
            "Select topic modeling algorithm:",
            ["Latent Dirichlet Allocation (LDA)", "Non-negative Matrix Factorization (NMF)"]
        )
        
        # Number of topics
        num_topics = st.slider("Number of topics:", 2, 20, 5)
        
        # Number of words per topic
        num_words = st.slider("Number of words per topic:", 5, 20, 10)
        
        # Preprocessing options
        st.write("Preprocessing options:")
        
        min_df = st.slider("Minimum document frequency:", 1, 20, 2, 
                          help="Ignore terms that appear in fewer than this number of documents")
        
        max_df = st.slider("Maximum document frequency (%):", 50, 100, 95, 
                          help="Ignore terms that appear in more than this percentage of documents")
        
        # Run topic modeling
        if st.button("Run Topic Modeling"):
            with st.spinner("Performing topic modeling..."):
                try:
                    # Create vectorizer
                    if vectorization == "Count Vectorizer":
                        vectorizer = CountVectorizer(
                            min_df=min_df,
                            max_df=max_df/100.0,
                            stop_words='english'
                        )
                    else:  # TF-IDF
                        vectorizer = TfidfVectorizer(
                            min_df=min_df,
                            max_df=max_df/100.0,
                            stop_words='english'
                        )
                    
                    # Fit vectorizer
                    X = vectorizer.fit_transform(text_data)
                    
                    # Get feature names
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Build model
                    if algorithm == "Latent Dirichlet Allocation (LDA)":
                        model = LatentDirichletAllocation(
                            n_components=num_topics,
                            random_state=42
                        )
                    else:  # NMF
                        # Import NMF
                        from sklearn.decomposition import NMF
                        
                        model = NMF(
                            n_components=num_topics,
                            random_state=42
                        )
                    
                    # Fit model
                    model.fit(X)
                    
                    # Extract topics
                    topics = []
                    for topic_idx, topic in enumerate(model.components_):
                        # Get top words for this topic
                        top_words_idx = topic.argsort()[:-num_words-1:-1]
                        top_words = [feature_names[i] for i in top_words_idx]
                        
                        topics.append({
                            'Topic': topic_idx + 1,
                            'Words': top_words,
                            'Weight': topic.sum()
                        })
                    
                    # Display topics
                    st.subheader("Discovered Topics")
                    
                    topics_df = pd.DataFrame(topics)
                    
                    # Format words as comma-separated strings
                    topics_df['Top Words'] = topics_df['Words'].apply(lambda x: ', '.join(x))
                    topics_df = topics_df.drop(columns=['Words'])
                    
                    # Normalize weights
                    total_weight = topics_df['Weight'].sum()
                    topics_df['Weight (%)'] = (topics_df['Weight'] / total_weight * 100).round(2)
                    
                    # Display topics table
                    st.dataframe(topics_df)
                    
                    # Visualize topic weights
                    st.subheader("Topic Weights")
                    
                    fig = px.pie(
                        topics_df,
                        values='Weight',
                        names='Topic',
                        title='Topic Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Get document-topic distributions
                    doc_topic_dist = model.transform(X)
                    
                    # Create document-topic dataframe
                    doc_topic_df = pd.DataFrame(
                        doc_topic_dist,
                        columns=[f'Topic {i+1}' for i in range(num_topics)]
                    )
                    
                    # Add document index
                    doc_topic_df.insert(0, 'Document', range(1, len(doc_topic_dist) + 1))
                    
                    # Display document-topic distribution
                    st.subheader("Document-Topic Distribution")
                    st.write("Each value represents the strength of each topic in a document.")
                    
                    # Display first few documents
                    st.dataframe(doc_topic_df.head(10))
                    
                    # Visualize document-topic distribution
                    st.subheader("Document-Topic Heatmap")
                    
                    # Create heatmap
                    fig = px.imshow(
                        doc_topic_dist[:30],  # Show first 30 documents for better visibility
                        labels=dict(x="Topic", y="Document", color="Weight"),
                        x=[f'Topic {i+1}' for i in range(num_topics)],
                        y=[f'Doc {i+1}' for i in range(min(30, len(doc_topic_dist)))],
                        color_continuous_scale="Viridis",
                        title="Document-Topic Distribution (First 30 documents)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Dominant topic per document
                    st.subheader("Dominant Topic per Document")
                    
                    # Get dominant topic for each document
                    doc_topic_df['Dominant Topic'] = doc_topic_df.iloc[:, 1:].idxmax(axis=1)
                    doc_topic_df['Confidence'] = doc_topic_df.iloc[:, 1:].max(axis=1)
                    
                    # Count documents per dominant topic
                    topic_counts = doc_topic_df['Dominant Topic'].value_counts().reset_index()
                    topic_counts.columns = ['Topic', 'Count']
                    
                    # Visualize document counts per topic
                    fig = px.bar(
                        topic_counts,
                        x='Topic',
                        y='Count',
                        title='Number of Documents per Dominant Topic'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Representative documents
                    st.subheader("Representative Documents per Topic")
                    
                    for topic_idx in range(num_topics):
                        topic_name = f'Topic {topic_idx+1}'
                        
                        with st.expander(f"{topic_name}: {topics_df.loc[topic_idx, 'Top Words']}"):
                            # Get top documents for this topic
                            topic_docs = doc_topic_df.sort_values(by=topic_name, ascending=False).head(3)
                            
                            for i, (idx, row) in enumerate(topic_docs.iterrows()):
                                doc_idx = row['Document'] - 1
                                confidence = row[topic_name]
                                
                                st.write(f"Document {row['Document']} (Confidence: {confidence:.4f}):")
                                st.text(text_data[doc_idx][:300] + "..." if len(text_data[doc_idx]) > 300 else text_data[doc_idx])
                                st.write("---")
                
                except Exception as e:
                    st.error(f"Error during topic modeling: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # 4. Sentiment Analysis
    elif selected_tool == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        
        st.markdown("""
        Sentiment analysis helps determine the emotional tone behind a text.
        This tool analyzes whether the text expresses positive, negative, or neutral sentiment.
        """)
        
        # Select sentiment analysis model
        model_option = st.radio(
            "Select sentiment analysis model:",
            ["Hugging Face Transformer Model"]
        )
        
        if model_option == "Hugging Face Transformer Model":
            with st.spinner("Loading sentiment analysis model..."):
                try:
                    # Initialize sentiment analysis pipeline
                    sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        max_length=512,
                        truncation=True
                    )
                    
                    st.success("Sentiment analysis model loaded successfully")
                    
                    # Run sentiment analysis
                    if st.button("Analyze Sentiment"):
                        with st.spinner("Analyzing sentiment..."):
                            # Limit to first 100 entries for performance
                            analysis_texts = text_data[:100]
                            
                            if len(text_data) > 100:
                                st.warning("For performance reasons, only the first 100 text entries will be analyzed.")
                            
                            # Analyze sentiment
                            results = []
                            
                            # Process in batches for better performance
                            batch_size = 8
                            for i in range(0, len(analysis_texts), batch_size):
                                batch = analysis_texts[i:i+batch_size]
                                
                                # Truncate long texts
                                truncated_batch = [text[:1000] for text in batch]
                                
                                # Get sentiment predictions
                                predictions = sentiment_analyzer(truncated_batch)
                                
                                for j, prediction in enumerate(predictions):
                                    text_idx = i + j
                                    results.append({
                                        'Document': text_idx + 1,
                                        'Text': analysis_texts[text_idx][:100] + "..." if len(analysis_texts[text_idx]) > 100 else analysis_texts[text_idx],
                                        'Sentiment': prediction['label'],
                                        'Score': prediction['score']
                                    })
                            
                            # Convert to dataframe
                            results_df = pd.DataFrame(results)
                            
                            # Display results
                            st.subheader("Sentiment Analysis Results")
                            st.dataframe(results_df)
                            
                            # Sentiment statistics
                            sentiment_counts = results_df['Sentiment'].value_counts().reset_index()
                            sentiment_counts.columns = ['Sentiment', 'Count']
                            
                            # Calculate percentages
                            total = sentiment_counts['Count'].sum()
                            sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total * 100).round(2)
                            
                            # Display sentiment distribution
                            st.subheader("Sentiment Distribution")
                            
                            fig = px.pie(
                                sentiment_counts,
                                values='Count',
                                names='Sentiment',
                                title='Sentiment Distribution',
                                color='Sentiment',
                                color_discrete_map={
                                    'POSITIVE': 'green',
                                    'NEGATIVE': 'red',
                                    'NEUTRAL': 'gray'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Score distribution
                            st.subheader("Confidence Score Distribution")
                            
                            fig = px.histogram(
                                results_df,
                                x='Score',
                                color='Sentiment',
                                nbins=20,
                                title='Confidence Score Distribution',
                                color_discrete_map={
                                    'POSITIVE': 'green',
                                    'NEGATIVE': 'red',
                                    'NEUTRAL': 'gray'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Most positive and most negative documents
                            st.subheader("Most Positive and Negative Documents")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("Most Positive Documents:")
                                positive_docs = results_df[results_df['Sentiment'] == 'POSITIVE'].sort_values('Score', ascending=False).head(3)
                                
                                for i, (idx, row) in enumerate(positive_docs.iterrows()):
                                    with st.expander(f"Document {row['Document']} (Score: {row['Score']:.4f})"):
                                        doc_idx = row['Document'] - 1
                                        st.text(text_data[doc_idx][:300] + "..." if len(text_data[doc_idx]) > 300 else text_data[doc_idx])
                            
                            with col2:
                                st.write("Most Negative Documents:")
                                negative_docs = results_df[results_df['Sentiment'] == 'NEGATIVE'].sort_values('Score', ascending=False).head(3)
                                
                                for i, (idx, row) in enumerate(negative_docs.iterrows()):
                                    with st.expander(f"Document {row['Document']} (Score: {row['Score']:.4f})"):
                                        doc_idx = row['Document'] - 1
                                        st.text(text_data[doc_idx][:300] + "..." if len(text_data[doc_idx]) > 300 else text_data[doc_idx])
                    
                except Exception as e:
                    st.error(f"Error loading or using sentiment analysis model: {str(e)}")
    
    # 5. Text Classification
    elif selected_tool == "Text Classification":
        st.subheader("Text Classification")
        
        st.markdown("""
        Text classification categorizes text into predefined classes or categories.
        This tool uses transformers models to classify text into various categories.
        """)
        
        # Select classification task
        task = st.selectbox(
            "Select classification task:",
            ["Topic Classification", "Zero-shot Classification"]
        )
        
        if task == "Topic Classification":
            with st.spinner("Loading topic classification model..."):
                try:
                    # Initialize topic classification pipeline
                    classifier = pipeline(
                        "text-classification",
                        model="facebook/bart-large-mnli",
                        max_length=512,
                        truncation=True
                    )
                    
                    st.success("Topic classification model loaded successfully")
                    
                    # Define topics/categories
                    default_categories = "economy, finance, politics, education, health, technology, sports, environment"
                    
                    categories = st.text_input(
                        "Enter categories (comma-separated):",
                        value=default_categories,
                        help="Enter categories for classification, separated by commas"
                    )
                    
                    # Convert to list
                    category_list = [c.strip() for c in categories.split(",")]
                    
                    # Run classification
                    if st.button("Classify Text"):
                        with st.spinner("Classifying text..."):
                            # Limit to first 50 entries for performance
                            analysis_texts = text_data[:50]
                            
                            if len(text_data) > 50:
                                st.warning("For performance reasons, only the first 50 text entries will be classified.")
                            
                            # Process each text
                            results = []
                            
                            for i, text in enumerate(analysis_texts):
                                # Truncate long texts
                                truncated_text = text[:1000]
                                
                                # Predict topics
                                prediction = classifier(truncated_text, category_list, multi_label=False)
                                
                                # Add to results
                                results.append({
                                    'Document': i + 1,
                                    'Text': text[:100] + "..." if len(text) > 100 else text,
                                    'Category': prediction[0]['label'],
                                    'Score': prediction[0]['score']
                                })
                            
                            # Convert to dataframe
                            results_df = pd.DataFrame(results)
                            
                            # Display results
                            st.subheader("Classification Results")
                            st.dataframe(results_df)
                            
                            # Category statistics
                            category_counts = results_df['Category'].value_counts().reset_index()
                            category_counts.columns = ['Category', 'Count']
                            
                            # Calculate percentages
                            total = category_counts['Count'].sum()
                            category_counts['Percentage'] = (category_counts['Count'] / total * 100).round(2)
                            
                            # Display category distribution
                            st.subheader("Category Distribution")
                            
                            fig = px.bar(
                                category_counts,
                                x='Category',
                                y='Count',
                                title='Topic Distribution',
                                color='Category'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Pie chart
                            fig = px.pie(
                                category_counts,
                                values='Count',
                                names='Category',
                                title='Topic Distribution',
                                color='Category'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Representative documents per category
                            st.subheader("Representative Documents per Category")
                            
                            for category in category_list:
                                if category in results_df['Category'].values:
                                    with st.expander(f"Category: {category}"):
                                        # Get top documents for this category
                                        category_docs = results_df[results_df['Category'] == category].sort_values('Score', ascending=False).head(2)
                                        
                                        for i, (idx, row) in enumerate(category_docs.iterrows()):
                                            doc_idx = row['Document'] - 1
                                            st.write(f"Document {row['Document']} (Score: {row['Score']:.4f}):")
                                            st.text(text_data[doc_idx][:300] + "..." if len(text_data[doc_idx]) > 300 else text_data[doc_idx])
                                            st.write("---")
                    
                except Exception as e:
                    st.error(f"Error loading or using classification model: {str(e)}")
        
        elif task == "Zero-shot Classification":
            with st.spinner("Loading zero-shot classification model..."):
                try:
                    # Initialize zero-shot classification pipeline
                    classifier = pipeline(
                        "zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        max_length=512,
                        truncation=True
                    )
                    
                    st.success("Zero-shot classification model loaded successfully")
                    
                    # Define custom labels
                    default_labels = "budget, economy, finance, policy, development, infrastructure, education, health"
                    
                    labels = st.text_input(
                        "Enter custom labels (comma-separated):",
                        value=default_labels,
                        help="Enter labels for classification, separated by commas"
                    )
                    
                    # Convert to list
                    label_list = [l.strip() for l in labels.split(",")]
                    
                    # Multi-label option
                    multi_label = st.checkbox("Multi-label classification", value=True)
                    
                    # Run classification
                    if st.button("Classify Text"):
                        with st.spinner("Classifying text..."):
                            # Limit to first 20 entries for performance
                            analysis_texts = text_data[:20]
                            
                            if len(text_data) > 20:
                                st.warning("For performance reasons, only the first 20 text entries will be classified.")
                            
                            # Process each text
                            results = []
                            
                            for i, text in enumerate(analysis_texts):
                                # Truncate long texts
                                truncated_text = text[:1000]
                                
                                # Predict labels
                                prediction = classifier(truncated_text, label_list, multi_label=multi_label)
                                
                                # Add to results
                                result_entry = {
                                    'Document': i + 1,
                                    'Text': text[:100] + "..." if len(text) > 100 else text
                                }
                                
                                # Add top 3 labels and scores
                                for j, (label, score) in enumerate(zip(prediction['labels'][:3], prediction['scores'][:3])):
                                    result_entry[f'Label {j+1}'] = label
                                    result_entry[f'Score {j+1}'] = score
                                
                                results.append(result_entry)
                            
                            # Convert to dataframe
                            results_df = pd.DataFrame(results)
                            
                            # Display results
                            st.subheader("Classification Results")
                            st.dataframe(results_df)
                            
                            # Aggregate label statistics (using Label 1)
                            label_counts = results_df['Label 1'].value_counts().reset_index()
                            label_counts.columns = ['Label', 'Count']
                            
                            # Display label distribution
                            st.subheader("Primary Label Distribution")
                            
                            fig = px.bar(
                                label_counts,
                                x='Label',
                                y='Count',
                                title='Primary Label Distribution',
                                color='Label'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Representative documents per label
                            st.subheader("Representative Documents per Label")
                            
                            for label in label_list:
                                matching_docs = results_df[results_df['Label 1'] == label]
                                
                                if not matching_docs.empty:
                                    with st.expander(f"Label: {label}"):
                                        # Get top documents for this label
                                        label_docs = matching_docs.sort_values('Score 1', ascending=False).head(2)
                                        
                                        for i, (idx, row) in enumerate(label_docs.iterrows()):
                                            doc_idx = row['Document'] - 1
                                            st.write(f"Document {row['Document']} (Score: {row['Score 1']:.4f}):")
                                            st.text(text_data[doc_idx][:300] + "..." if len(text_data[doc_idx]) > 300 else text_data[doc_idx])
                                            st.write("---")
                    
                except Exception as e:
                    st.error(f"Error loading or using zero-shot classification model: {str(e)}")
    
    # 6. Named Entity Recognition
    elif selected_tool == "Named Entity Recognition":
        st.subheader("Named Entity Recognition (NER)")
        
        st.markdown("""
        Named Entity Recognition identifies entities such as persons, organizations, locations, dates, 
        and amounts in text data. This tool extracts and visualizes these entities.
        """)
        
        with st.spinner("Loading NER model..."):
            try:
                # Initialize NER pipeline
                ner_pipeline = pipeline(
                    "ner",
                    aggregation_strategy="simple"
                )
                
                st.success("NER model loaded successfully")
                
                # Run NER
                if st.button("Extract Entities"):
                    with st.spinner("Extracting entities..."):
                        # Limit to first 10 entries for performance
                        analysis_texts = text_data[:10]
                        
                        if len(text_data) > 10:
                            st.warning("For performance reasons, only the first 10 text entries will be analyzed.")
                        
                        # Process each text
                        all_entities = []
                        
                        for i, text in enumerate(analysis_texts):
                            # Truncate long texts
                            truncated_text = text[:1000]
                            
                            # Extract entities
                            entities = ner_pipeline(truncated_text)
                            
                            # Add to results
                            for entity in entities:
                                all_entities.append({
                                    'Document': i + 1,
                                    'Text': truncated_text[entity['start']:entity['end']],
                                    'Type': entity['entity_group'],
                                    'Score': entity['score'],
                                    'Start': entity['start'],
                                    'End': entity['end']
                                })
                        
                        # Convert to dataframe
                        entities_df = pd.DataFrame(all_entities)
                        
                        if entities_df.empty:
                            st.warning("No entities found in the analyzed text.")
                            return
                        
                        # Display results
                        st.subheader("Extracted Entities")
                        st.dataframe(entities_df)
                        
                        # Entity type statistics
                        entity_counts = entities_df['Type'].value_counts().reset_index()
                        entity_counts.columns = ['Entity Type', 'Count']
                        
                        # Display entity type distribution
                        st.subheader("Entity Type Distribution")
                        
                        fig = px.bar(
                            entity_counts,
                            x='Entity Type',
                            y='Count',
                            title='Entity Type Distribution',
                            color='Entity Type'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Most frequent entities per type
                        st.subheader("Most Frequent Entities by Type")
                        
                        entity_types = entities_df['Type'].unique()
                        
                        for entity_type in entity_types:
                            with st.expander(f"Entity Type: {entity_type}"):
                                # Get entities of this type
                                type_entities = entities_df[entities_df['Type'] == entity_type]
                                
                                # Count occurrences
                                entity_freq = type_entities['Text'].value_counts().reset_index()
                                entity_freq.columns = ['Entity', 'Frequency']
                                
                                # Display top entities
                                st.write(f"Top {min(10, len(entity_freq))} most frequent {entity_type} entities:")
                                st.dataframe(entity_freq.head(10))
                                
                                # Plot top entities
                                if len(entity_freq) > 1:
                                    fig = px.bar(
                                        entity_freq.head(10),
                                        x='Entity',
                                        y='Frequency',
                                        title=f'Top {entity_type} Entities',
                                        color='Frequency'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Visualize entities in context
                        st.subheader("Entities in Context")
                        
                        for i, doc_idx in enumerate(range(min(3, len(analysis_texts)))):
                            with st.expander(f"Document {doc_idx + 1}"):
                                doc_text = analysis_texts[doc_idx]
                                
                                # Get entities for this document
                                doc_entities = entities_df[entities_df['Document'] == doc_idx + 1]
                                
                                if doc_entities.empty:
                                    st.write("No entities found in this document.")
                                    continue
                                
                                # Show text with highlighted entities
                                st.write("Entities found:")
                                st.dataframe(doc_entities[['Text', 'Type', 'Score']])
                                
                                # Truncate text for display
                                display_text = doc_text[:1000] + "..." if len(doc_text) > 1000 else doc_text
                                st.text(display_text)
            
            except Exception as e:
                st.error(f"Error loading or using NER model: {str(e)}")

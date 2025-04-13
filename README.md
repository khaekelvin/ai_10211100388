# ai_10211100388
# Kelvin Sungzie Duobu
# Data Analysis - ML/AI Explorer
### https://ai10211100388.streamlit.app/

A comprehensive Streamlit application for exploring and solving various machine learning and AI problems with interactive interfaces, visualizations, and practical use cases using Ghana election and budget data.

## Features

- **Regression Analysis**: Build and evaluate linear regression models on your data
- **Clustering Analysis**: Perform K-means clustering to identify patterns and segments
- **Neural Networks**: Design, train and evaluate neural network models
- **NLP & Large Language Models**: Analyze text data and utilize LLM models for Q&A
- **Election Data Analysis**: Explore Ghana's election results with interactive visualizations
- **Budget Data Analysis**: Extract insights from Ghana's budget documents

## Requirements

This application requires Python 3.8+ and the following libraries:
- streamlit
- pandas
- numpy

## Installation & Setup

1. Clone or download this repository to your local machine
2. Navigate to the project directory in your terminal

```bash
cd path/to/data-analysis
```

3. Install the required packages

```bash
pip install streamlit pandas numpy
```

You can also install additional packages if you want to enhance functionality:

```bash
pip install matplotlib seaborn scikit-learn
```

4. Ensure the data files are in the correct location:
   - Place the Ghana Election CSV file in the `attached_assets` folder
   - Place the Budget Statement PDF in the `attached_assets` folder

## Running the Application

1. Create a `.streamlit` directory in the project root (if it doesn't exist):

```bash
mkdir -p .streamlit
```

2. Create a configuration file at `.streamlit/config.toml` with the following content:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 8501

[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#212121"
```

3. Run the Streamlit application with the following command:

```bash
streamlit run app.py
```

This will start the application and automatically open it in your default web browser. If it doesn't open automatically, you can access it at [http://localhost:8501](http://localhost:8501).

## Using External APIs (Optional)

For enhanced LLM functionality, you can provide your custom API keys for:
- Mistral AI

## Data Sources

The application uses the following data sources:
- Ghana Election Results: CSV data with historical election results
- Ghana 2025 Budget Statement: PDF document with budget information
- User-uploaded datasets: Custom datasets can be uploaded for analysis

## Application Structure

- `app.py`: Main application entry point
- `data_processing.py`: Functions for data preparation and exploration
- `regression.py`: Linear regression implementation
- `clustering.py`: K-means clustering implementation
- `neural_networks.py`: Neural network simulation
- `nlp.py`: Natural language processing utilities
- `election_analysis.py`: Election data analysis functions
- `budget_analysis.py`: Budget document analysis functions
- `utils.py`: Utility functions used across the application

## Detailed Documentation

### How to Use Each Feature

#### Regression Analysis
1. **Upload Data**: Upload a regression-related CSV file or use the preloaded datasets
2. **Select Target Variable**: Choose the continuous variable you want to predict
3. **Choose Features**: Select the independent variables for your model
4. **Configure and Train**: Set parameters and click "Build Regression Model"
5. **Evaluate Results**: Review metrics like R², MAE, and RMSE
6. **Make Predictions**: Enter values to get predictions from your trained model
7. **Download Results**: Save model outputs and predictions as CSV

#### Clustering Analysis
1. **Select/Upload Data**: Choose from preloaded data or upload a dataset
2. **Choose Features**: Select numerical columns to use for clustering
3. **Set Cluster Count**: Use the interactive slider to select number of clusters
4. **Run K-Means**: Perform clustering and visualize results
5. **Explore Clusters**: View statistics for each identified group
6. **Download Results**: Save the clustered dataset with cluster labels
7. **Adjust and Repeat**: Fine-tune parameters for optimal segmentation

#### Neural Networks
1. **Prepare Data**: Upload or select a classification dataset
2. **Configure Network**: Set hidden layers, neurons, and activation functions
3. **Train Model**: Set epochs and batch size then train the network
4. **Monitor Progress**: View real-time training metrics and loss curves
5. **Evaluate Performance**: Check accuracy, precision, recall, and F1-score
6. **Test with New Data**: Make predictions on custom inputs

#### LLM Analysis with Mistral-7B
1. **Choose Document**: Select Ghana Budget or Election dataset to analyze
2. **Ask Questions**: Enter questions or select from provided examples
3. **Review Responses**: See the generated answers with source citations
4. **Compare Results**: View side-by-side comparison with and without RAG
5. **Explore Methodology**: Learn how Retrieval-Augmented Generation works

### Dataset and Model Details

#### Datasets
1. **Ghana Election Results (CSV)**
   - Contains historical election data from Ghana
   - Features: Year, Region, Party, Candidate, Votes, Percentage, etc.
   - Use cases: Electoral trend analysis, regional voting pattern analysis

2. **Ghana 2025 Budget Statement (PDF)**
   - Official budget document for Ghana's 2025 fiscal year
   - Contents: Financial projections, policy statements, sector allocations
   - Use cases: Policy analysis, financial figure extraction, budget allocation analysis

3. **User-Uploaded Datasets**
   - Support for CSV and Excel formats
   - Automatic data type detection and preprocessing
   - Compatible with all analysis modules

#### Models Used
1. **Regression Models**:
   - Linear Regression, Ridge, Lasso, ElasticNet
   - Random Forest, Gradient Boosting
   - SVR, KNN, Decision Tree

2. **Clustering Models**:
   - K-Means with interactive cluster selection
   - Options for data standardization and dimensionality reduction
   - Visualization in 2D and 3D when applicable

3. **Neural Network Models**:
   - Feed-forward networks with customizable architecture
   - Support for various activation functions
   - Configurable training parameters

4. **LLM Implementation**:
   - Mistral-7B-Instruct-v0.1 as primary model
   - Optional variants: Mistral-7B-v0.2, Mistral-7B-OpenOrca
   - Accessed via HuggingFace API

### LLM Architecture and Methodology

#### Retrieval-Augmented Generation (RAG) Architecture
Our application implements a comprehensive RAG architecture to enhance Mistral-7B's capabilities for document question-answering:

```
[User Question] → [Document Retrieval] → [Context Selection] → [Mistral-7B Model] → [Response]
                     ↑                       ↑
           [Document Indexing] ←  [Ghana Budget & Election Documents]
```

#### Document Processing Pipeline
1. **Data Extraction**: Convert PDF/CSV documents to plain text
2. **Text Cleaning**: Remove noise, normalize formatting, handle special characters
3. **Chunking**: Split documents into semantic chunks of ~500 tokens
4. **Embedding Generation**: Create vector embeddings for each chunk
5. **Vector Storage**: Store embeddings in a vector database for efficient retrieval

#### Retrieval System
1. **Query Processing**: Analyze user question and convert to embedding
2. **Semantic Search**: Find relevant document chunks based on embedding similarity
3. **Relevance Ranking**: Sort chunks by relevance score
4. **Context Assembly**: Select top chunks and combine into context window
5. **Confidence Scoring**: Assign confidence scores to each retrieved passage

#### Generation Process
1. **Prompt Creation**: Construct prompt with retrieved context and user question
2. **Model Invocation**: Send to Mistral-7B-Instruct-v0.1 for answer generation
3. **Response Formatting**: Structure response with citations and confidence indicators
4. **Explanation Generation**: Provide reasoning for the generated answer

#### Evaluation Framework
1. **Result Comparison**: Contrast RAG responses with direct LLM output
2. **Confidence Visualization**: Display confidence scores for each source passage
3. **Reference Tracking**: Link answer components to source document sections
4. **User Feedback Collection**: Gather ratings on response accuracy and relevance

### Performance Comparison with ChatGPT

#### Advantages of Our Mistral-7B RAG Implementation

1. **Domain Specificity**
   - Our system: Specifically trained on Ghana's budget and election documents
   - ChatGPT: Broad training without specific focus on Ghana data
   - Result: More accurate and relevant answers for Ghana-specific queries

2. **Data Recency**
   - Our system: Can answer questions about 2025 budget (beyond ChatGPT training cutoff)
   - ChatGPT: Limited by training cutoff date
   - Result: Our system provides current information from recent documents

3. **Source Transparency**
   - Our system: Provides precise document citations with confidence scores
   - ChatGPT: Often unable to cite specific sources
   - Result: Users can verify information accuracy and provenance

4. **Query Precision**
   - Our system: Retrieves and focuses on relevant document sections
   - ChatGPT: May mix information or provide generalized responses
   - Result: Highly specific and targeted answers to complex queries

5. **Local Deployment Potential**
   - Our system: Uses Mistral-7B which can run on consumer hardware
   - ChatGPT: Requires remote API calls to OpenAI
   - Result: Potential for fully on-premise deployment for sensitive data

6. **Cost Efficiency**
   - Our system: Leverages open-source Mistral-7B models
   - ChatGPT: Requires commercial API usage
   - Result: More sustainable long-term solution with lower operational costs

Overall, our Mistral-7B RAG implementation delivers more accurate, relevant, and verifiable responses on Ghana-specific queries while maintaining source transparency and potential for local deployment.

Kelvin Sungzie Duobu - 10211100388

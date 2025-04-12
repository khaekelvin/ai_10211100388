import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import io
import PyPDF2
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string

# Download necessary NLTK data
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def render_budget_analysis(datasets):
    st.title("Budget Data Analysis")
    
    st.markdown("""
    This section provides analysis tools for Ghana's budget documents. 
    Extract key figures, analyze spending patterns, and identify budget priorities.
    """)
    
    # Check if budget data path is available
    if datasets['budget_data_path'] is None:
        st.error("Budget PDF document path not found.")
        return
    
    # Attempt to extract data from the PDF
    try:
        # Read from the budget PDF
        with st.spinner("Extracting text from budget document..."):
            with open(datasets['budget_data_path'], 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get basic document info
                num_pages = len(pdf_reader.pages)
                
                # Extract text from a limited number of pages for faster processing
                pages_to_extract = min(50, num_pages)  # Limit to first 50 pages for performance
                text_content = ""
                
                for page_num in range(pages_to_extract):
                    text_content += pdf_reader.pages[page_num].extract_text()
                
        st.success(f"Successfully extracted text from {pages_to_extract} of {num_pages} pages")
        
        # Display document info
        st.subheader("Document Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Pages", num_pages)
            
            # Extract title from the first page if available
            title_match = re.search(r"THEME:\s+(.+?)(?:\n|$)", text_content)
            if title_match:
                document_title = title_match.group(1).strip()
                st.write(f"**Document Title:** {document_title}")
            else:
                st.write("**Document Title:** Ghana 2025 Budget Statement")
        
        with col2:
            # Extract year from text
            year_match = re.search(r"(\d{4})(?:\s+Financial Year|\s+Budget)", text_content)
            if year_match:
                budget_year = year_match.group(1)
                st.metric("Budget Year", budget_year)
            else:
                st.metric("Budget Year", "2025")
            
            # Extract minister name
            minister_match = re.search(r"Presented.*\n(DR\.\s+\w+\s+\w+\s+\w+)", text_content)
            if minister_match:
                minister_name = minister_match.group(1).strip()
                st.write(f"**Presented by:** {minister_name}")
        
        # Analysis tools tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Text Analysis", "Key Figures", "Sectoral Analysis", "Policy Priorities"])
        
        with tab1:
            st.subheader("Budget Document Text Analysis")
            
            # Text statistics
            st.write("### Text Statistics")
            
            # Basic text stats
            words = word_tokenize(text_content)
            sentences = sent_tokenize(text_content)
            paragraphs = text_content.split('\n\n')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Words", len(words))
            
            with col2:
                st.metric("Total Sentences", len(sentences))
            
            with col3:
                st.metric("Total Paragraphs", len(paragraphs))
            
            # Word frequency analysis
            st.write("### Word Frequency Analysis")
            
            # Preprocessing options
            show_options = st.checkbox("Show preprocessing options", value=False)
            
            if show_options:
                col1, col2 = st.columns(2)
                
                with col1:
                    remove_stopwords = st.checkbox("Remove common words", value=True)
                    remove_numbers = st.checkbox("Remove numbers", value=True)
                
                with col2:
                    remove_short_words = st.checkbox("Remove short words (< 3 letters)", value=True)
                    custom_words = st.text_input("Additional words to exclude (comma-separated)")
            else:
                remove_stopwords = True
                remove_numbers = True
                remove_short_words = True
                custom_words = ""
            
            # Custom stopwords
            custom_stopwords = [word.strip() for word in custom_words.split(',')] if custom_words else []
            
            # Process text for word frequency
            stop_words = set(stopwords.words('english'))
            stop_words.update(['fiscal', 'government', 'year', 'ghana', 'ghana\'s', 'million', 'billion', 'ghc'])
            stop_words.update(custom_stopwords)
            
            # Tokenize and clean
            words = [word.lower() for word in words if word.isalpha() or (not remove_numbers and word.isalnum())]
            
            if remove_stopwords:
                words = [word for word in words if word not in stop_words]
            
            if remove_short_words:
                words = [word for word in words if len(word) > 2]
            
            # Count word frequencies
            word_freq = Counter(words)
            
            # Get top words
            top_n = st.slider("Number of top words to display:", 10, 50, 20)
            top_words = word_freq.most_common(top_n)
            
            # Create dataframe for plotting
            word_freq_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            
            # Plot word frequencies
            fig = px.bar(
                word_freq_df,
                x='Word',
                y='Frequency',
                title=f'Top {top_n} Words in Budget Document',
                color='Frequency'
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Word cloud (if available)
            try:
                from wordcloud import WordCloud
                
                st.write("### Word Cloud")
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=100,
                    contour_width=3,
                    contour_color='steelblue'
                ).generate(' '.join(words))
                
                # Display word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                plt.tight_layout()
                
                st.pyplot(fig)
                
            except ImportError:
                st.info("Word cloud visualization requires the wordcloud package. Please install it to enable this feature.")
            
            # Keyword search tool
            st.write("### Keyword Search Tool")
            
            search_query = st.text_input("Enter keywords to search in the budget document:")
            
            if search_query:
                search_terms = [term.strip().lower() for term in search_query.split(',')]
                
                # Search for terms in paragraphs
                results = []
                
                for i, paragraph in enumerate(paragraphs):
                    paragraph_lower = paragraph.lower()
                    
                    for term in search_terms:
                        if term in paragraph_lower:
                            # Get a snippet of the paragraph containing the term
                            start_idx = max(0, paragraph_lower.find(term) - 50)
                            end_idx = min(len(paragraph_lower), paragraph_lower.find(term) + len(term) + 50)
                            
                            # Create snippet with highlighted term
                            snippet = paragraph[start_idx:end_idx]
                            
                            results.append({
                                'Term': term,
                                'Paragraph': i + 1,
                                'Context': f"...{snippet}..."
                            })
                
                if results:
                    st.write(f"Found {len(results)} instances of your search terms:")
                    
                    # Convert to dataframe for display
                    results_df = pd.DataFrame(results)
                    
                    # Display results with expandable contexts
                    for i, (index, row) in enumerate(results_df.iterrows()):
                        with st.expander(f"Result {i+1}: {row['Term']} (Paragraph {row['Paragraph']})"):
                            st.write(row['Context'])
                else:
                    st.warning("No matches found for your search terms.")
        
        with tab2:
            st.subheader("Key Budget Figures")
            
            # Extract key financial figures
            st.write("### Financial Highlights")
            
            # Function to extract financial figures
            def extract_financial_figures(text):
                # Pattern for monetary values in billions
                billion_pattern = r"GH[¢C]\s*(\d+(?:\.\d+)?)\s*billion"
                # Pattern for monetary values in millions
                million_pattern = r"GH[¢C]\s*(\d+(?:\.\d+)?)\s*million"
                # Pattern for percentage values
                percentage_pattern = r"(\d+(?:\.\d+)?)\s*(?:percent|per cent|%)"
                # Pattern for GDP related figures
                gdp_pattern = r"(\d+(?:\.\d+)?)\s*(?:percent|per cent|%)\s*of\s*GDP"
                
                # Extract values
                billions = re.findall(billion_pattern, text, re.IGNORECASE)
                millions = re.findall(million_pattern, text, re.IGNORECASE)
                percentages = re.findall(percentage_pattern, text, re.IGNORECASE)
                gdp_figures = re.findall(gdp_pattern, text, re.IGNORECASE)
                
                # Convert to float
                billions = [float(val) for val in billions]
                millions = [float(val) for val in millions]
                percentages = [float(val) for val in percentages]
                gdp_figures = [float(val) for val in gdp_figures]
                
                return {
                    'billions': billions,
                    'millions': millions,
                    'percentages': percentages,
                    'gdp_figures': gdp_figures
                }
            
            figures = extract_financial_figures(text_content)
            
            # Display statistics on extracted figures
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Billion GH¢ Figures", len(figures['billions']))
                
                if figures['billions']:
                    avg_billion = sum(figures['billions']) / len(figures['billions'])
                    st.metric("Average Billion Figure", f"GH¢ {avg_billion:.2f} billion")
                
                st.metric("Number of Million GH¢ Figures", len(figures['millions']))
                
                if figures['millions']:
                    avg_million = sum(figures['millions']) / len(figures['millions'])
                    st.metric("Average Million Figure", f"GH¢ {avg_million:.2f} million")
            
            with col2:
                st.metric("Number of Percentage Figures", len(figures['percentages']))
                
                if figures['percentages']:
                    avg_pct = sum(figures['percentages']) / len(figures['percentages'])
                    st.metric("Average Percentage", f"{avg_pct:.2f}%")
                
                st.metric("Number of GDP-related Figures", len(figures['gdp_figures']))
                
                if figures['gdp_figures']:
                    avg_gdp = sum(figures['gdp_figures']) / len(figures['gdp_figures'])
                    st.metric("Average GDP Percentage", f"{avg_gdp:.2f}% of GDP")
            
            # Display extracted figures
            st.write("### Extracted Financial Figures")
            
            show_all_figures = st.checkbox("Show all extracted figures", value=False)
            
            if show_all_figures:
                col1, col2 = st.columns(2)
                
                with col1:
                    if figures['billions']:
                        st.write("**Billion GH¢ Figures:**")
                        for i, val in enumerate(figures['billions'][:20]):  # Limit to first 20 for clarity
                            st.write(f"- GH¢ {val} billion")
                        
                        if len(figures['billions']) > 20:
                            st.write(f"... and {len(figures['billions']) - 20} more")
                    
                    if figures['millions']:
                        st.write("**Million GH¢ Figures:**")
                        for i, val in enumerate(figures['millions'][:20]):  # Limit to first 20 for clarity
                            st.write(f"- GH¢ {val} million")
                        
                        if len(figures['millions']) > 20:
                            st.write(f"... and {len(figures['millions']) - 20} more")
                
                with col2:
                    if figures['percentages']:
                        st.write("**Percentage Figures:**")
                        for i, val in enumerate(figures['percentages'][:20]):  # Limit to first 20 for clarity
                            st.write(f"- {val}%")
                        
                        if len(figures['percentages']) > 20:
                            st.write(f"... and {len(figures['percentages']) - 20} more")
                    
                    if figures['gdp_figures']:
                        st.write("**GDP-related Figures:**")
                        for i, val in enumerate(figures['gdp_figures'][:20]):  # Limit to first 20 for clarity
                            st.write(f"- {val}% of GDP")
                        
                        if len(figures['gdp_figures']) > 20:
                            st.write(f"... and {len(figures['gdp_figures']) - 20} more")
            
            # Find specific budget figures
            st.write("### Key Budget Indicators")
            
            # Function to find specific budget metrics
            def find_budget_metric(text, keyword, pattern):
                # Find paragraph containing keyword
                keyword_lower = keyword.lower()
                paragraphs = text.split('\n\n')
                relevant_paragraphs = [p for p in paragraphs if keyword_lower in p.lower()]
                
                # Try to extract value using pattern
                for paragraph in relevant_paragraphs:
                    matches = re.findall(pattern, paragraph, re.IGNORECASE)
                    if matches:
                        return matches[0]
                
                return None
            
            # Try to extract key metrics
            gdp_growth = find_budget_metric(
                text_content, 
                "GDP growth", 
                r"GDP growth of (\d+(?:\.\d+)?)\s*(?:percent|per cent|%)"
            )
            
            inflation = find_budget_metric(
                text_content, 
                "inflation", 
                r"inflation(?:[^.]*?)(?:of|at|to)\s*(\d+(?:\.\d+)?)\s*(?:percent|per cent|%)"
            )
            
            fiscal_deficit = find_budget_metric(
                text_content, 
                "fiscal deficit", 
                r"fiscal deficit(?:[^.]*?)(?:of|at|to)\s*(\d+(?:\.\d+)?)\s*(?:percent|per cent|%)"
            )
            
            revenue = find_budget_metric(
                text_content, 
                "total revenue", 
                r"total revenue(?:[^.]*?)(?:of|at|to)\s*GH[¢C]\s*(\d+(?:\.\d+)?)\s*(?:billion|million)"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if gdp_growth:
                    st.metric("Projected GDP Growth", f"{gdp_growth}%")
                else:
                    st.metric("Projected GDP Growth", "N/A")
                
                if inflation:
                    st.metric("Target Inflation", f"{inflation}%")
                else:
                    st.metric("Target Inflation", "N/A")
            
            with col2:
                if fiscal_deficit:
                    st.metric("Fiscal Deficit", f"{fiscal_deficit}% of GDP")
                else:
                    st.metric("Fiscal Deficit", "N/A")
                
                if revenue:
                    st.metric("Total Revenue", f"GH¢ {revenue} billion")
                else:
                    st.metric("Total Revenue", "N/A")
            
            # Custom metric search
            st.write("### Search for Budget Metrics")
            
            metric_search = st.text_input("Enter budget metric to search for (e.g., 'public debt', 'tax revenue'):")
            
            if metric_search:
                # Look for metric in text
                metric_paragraphs = [p for p in text_content.split('\n\n') if metric_search.lower() in p.lower()]
                
                if metric_paragraphs:
                    st.write(f"Found {len(metric_paragraphs)} paragraphs mentioning '{metric_search}':")
                    
                    for i, paragraph in enumerate(metric_paragraphs[:5]):  # Limit to first 5 for clarity
                        with st.expander(f"Reference {i+1}"):
                            st.write(paragraph)
                    
                    if len(metric_paragraphs) > 5:
                        st.write(f"... and {len(metric_paragraphs) - 5} more references")
                else:
                    st.warning(f"No paragraphs found containing '{metric_search}'.")
        
        with tab3:
            st.subheader("Sectoral Budget Analysis")
            
            # Define key sectors to look for
            sectors = [
                "Education", "Health", "Agriculture", "Energy", "Infrastructure",
                "Transportation", "Security", "Social Protection", "Tourism",
                "Environment", "Technology", "Employment"
            ]
            
            # Function to count sector mentions and extract budget allocations
            def analyze_sector(text, sector):
                sector_lower = sector.lower()
                mentions = text.lower().count(sector_lower)
                
                # Try to find budget allocation for sector
                allocation_pattern = r"(?:" + sector + r"[^.]*?)(?:allocation|budget|spending)(?:[^.]*?)(?:of|at|to)\s*GH[¢C]\s*(\d+(?:\.\d+)?)\s*(?:billion|million)"
                allocation_matches = re.findall(allocation_pattern, text, re.IGNORECASE)
                
                allocation = None
                unit = None
                
                if allocation_matches:
                    allocation = allocation_matches[0]
                    
                    # Determine if it's billion or million
                    context = re.search(r"GH[¢C]\s*" + re.escape(allocation) + r"\s*(billion|million)", text, re.IGNORECASE)
                    unit = context.group(1) if context else "unknown"
                
                return {
                    'sector': sector,
                    'mentions': mentions,
                    'allocation': allocation,
                    'unit': unit
                }
            
            # Analyze each sector
            sector_analysis = []
            
            for sector in sectors:
                result = analyze_sector(text_content, sector)
                sector_analysis.append(result)
            
            # Convert to dataframe
            sector_df = pd.DataFrame(sector_analysis)
            
            # Sort by mentions
            sector_df = sector_df.sort_values('mentions', ascending=False)
            
            # Display sector mentions
            st.write("### Sector Mentions in Budget")
            
            fig = px.bar(
                sector_df,
                x='sector',
                y='mentions',
                title='Sector Mentions in Budget Document',
                color='mentions'
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display allocation data if available
            allocations = sector_df[sector_df['allocation'].notnull()]
            
            if not allocations.empty:
                st.write("### Sector Budget Allocations")
                
                # Create formatted allocation column
                formatted_allocations = []
                
                for i, row in allocations.iterrows():
                    if row['allocation'] and row['unit']:
                        formatted_allocations.append(f"GH¢ {row['allocation']} {row['unit']}")
                    else:
                        formatted_allocations.append("Not found")
                
                # Add to dataframe
                allocations_display = allocations[['sector', 'mentions']].copy()
                allocations_display['allocation'] = formatted_allocations
                
                st.dataframe(allocations_display)
            
            # Sector-specific analysis
            st.write("### Detailed Sector Analysis")
            
            selected_sector = st.selectbox("Select sector for detailed analysis:", sectors)
            
            if selected_sector:
                # Search for paragraphs mentioning the sector
                sector_lower = selected_sector.lower()
                sector_paragraphs = [p for p in text_content.split('\n\n') if sector_lower in p.lower()]
                
                if sector_paragraphs:
                    st.write(f"Found {len(sector_paragraphs)} paragraphs mentioning '{selected_sector}':")
                    
                    for i, paragraph in enumerate(sector_paragraphs[:5]):  # Limit to first 5 for clarity
                        with st.expander(f"Reference {i+1}"):
                            st.write(paragraph)
                    
                    if len(sector_paragraphs) > 5:
                        st.write(f"... and {len(sector_paragraphs) - 5} more references")
                    
                    # Look for specific metrics related to the sector
                    st.write(f"### {selected_sector} Metrics")
                    
                    # Try to find specific metrics based on sector
                    metrics = []
                    
                    if selected_sector == "Education":
                        metrics = [
                            ("Schools to be built", r"(\d+)(?:\s+new)?\s+schools"),
                            ("Teacher recruitment", r"(?:recruit|employment of)(?:[^.]*?)(\d+(?:,\d+)?)\s+(?:new\s+)?teachers"),
                            ("Education budget", r"education(?:[^.]*?)(?:allocation|budget|spending)(?:[^.]*?)(?:of|at|to)\s*GH[¢C]\s*(\d+(?:\.\d+)?)")
                        ]
                    elif selected_sector == "Health":
                        metrics = [
                            ("Hospitals to be built", r"(\d+)(?:\s+new)?\s+hospitals"),
                            ("Health workers recruitment", r"(?:recruit|employment of)(?:[^.]*?)(\d+(?:,\d+)?)\s+(?:new\s+)?health\s+workers"),
                            ("Health budget", r"health(?:[^.]*?)(?:allocation|budget|spending)(?:[^.]*?)(?:of|at|to)\s*GH[¢C]\s*(\d+(?:\.\d+)?)")
                        ]
                    elif selected_sector == "Infrastructure":
                        metrics = [
                            ("Roads to be constructed", r"(\d+(?:,\d+)?)\s+(?:kilometers|km)\s+of\s+(?:roads|highways)"),
                            ("Bridges to be built", r"(\d+)\s+(?:new\s+)?bridges"),
                            ("Infrastructure budget", r"infrastructure(?:[^.]*?)(?:allocation|budget|spending)(?:[^.]*?)(?:of|at|to)\s*GH[¢C]\s*(\d+(?:\.\d+)?)")
                        ]
                    
                    # Search for metrics
                    metric_results = []
                    
                    for metric_name, pattern in metrics:
                        match = None
                        for paragraph in sector_paragraphs:
                            matches = re.findall(pattern, paragraph, re.IGNORECASE)
                            if matches:
                                match = matches[0]
                                break
                        
                        metric_results.append({
                            'Metric': metric_name,
                            'Value': match if match else "Not found"
                        })
                    
                    # Display metrics
                    if metric_results:
                        st.dataframe(pd.DataFrame(metric_results))
                    else:
                        st.write("No specific metrics extracted for this sector.")
                else:
                    st.warning(f"No paragraphs found mentioning '{selected_sector}'.")
        
        with tab4:
            st.subheader("Policy Priorities Analysis")
            
            # Extract policy statements
            st.write("### Policy Statements and Initiatives")
            
            # Look for policy keywords
            policy_keywords = [
                "policy", "initiative", "programme", "program", "strategy", 
                "project", "reform", "action plan", "investment"
            ]
            
            # Function to find policy statements
            def extract_policy_statements(text):
                sentences = sent_tokenize(text)
                
                policy_statements = []
                
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in policy_keywords):
                        policy_statements.append(sentence)
                
                return policy_statements
            
            policy_statements = extract_policy_statements(text_content)
            
            st.write(f"Found {len(policy_statements)} potential policy statements in the document.")
            
            # Display sample of policy statements
            with st.expander("View sample policy statements"):
                for i, statement in enumerate(policy_statements[:10]):  # Limit to first 10 for clarity
                    st.write(f"{i+1}. {statement}")
                
                if len(policy_statements) > 10:
                    st.write(f"... and {len(policy_statements) - 10} more statements")
            
            # Policy priorities based on frequency
            st.write("### Policy Priorities by Frequency")
            
            # Define policy areas to scan for
            policy_areas = {
                "Economic Growth": ["economic growth", "economy", "gdp", "economic development"],
                "Fiscal Management": ["fiscal", "deficit", "debt", "revenue", "expenditure", "tax"],
                "Education": ["education", "school", "student", "teacher", "learning"],
                "Health": ["health", "hospital", "medical", "disease", "patient"],
                "Infrastructure": ["infrastructure", "construction", "road", "bridge", "building"],
                "Agriculture": ["agriculture", "farming", "crop", "livestock", "food security"],
                "Energy": ["energy", "electricity", "power", "renewable", "solar"],
                "Employment": ["employment", "job", "unemployment", "labor", "work"],
                "Social Protection": ["social protection", "welfare", "poverty", "vulnerable", "safety net"],
                "Digital Transformation": ["digital", "technology", "innovation", "ict", "e-government"]
            }
            
            # Count mentions of each policy area
            policy_counts = {}
            
            for area, keywords in policy_areas.items():
                count = 0
                for keyword in keywords:
                    count += text_content.lower().count(keyword)
                policy_counts[area] = count
            
            # Convert to dataframe
            policy_counts_df = pd.DataFrame({
                'Policy Area': list(policy_counts.keys()),
                'Mentions': list(policy_counts.values())
            }).sort_values('Mentions', ascending=False)
            
            # Plot policy priorities
            fig = px.bar(
                policy_counts_df,
                x='Policy Area',
                y='Mentions',
                title='Policy Priorities Based on Frequency',
                color='Mentions'
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # New initiatives and flagship programs
            st.write("### New Initiatives and Flagship Programs")
            
            # Keywords for new initiatives
            new_initiative_keywords = [
                "new initiative", "new program", "flagship", "launch", "introduce", 
                "announce", "unveil", "implement"
            ]
            
            # Find sentences containing new initiatives
            new_initiatives = []
            
            for sentence in sent_tokenize(text_content):
                if any(keyword in sentence.lower() for keyword in new_initiative_keywords):
                    new_initiatives.append(sentence)
            
            if new_initiatives:
                st.write(f"Found {len(new_initiatives)} potential new initiatives in the document.")
                
                # Display new initiatives
                with st.expander("View potential new initiatives"):
                    for i, initiative in enumerate(new_initiatives[:15]):  # Limit to first 15 for clarity
                        st.write(f"{i+1}. {initiative}")
                    
                    if len(new_initiatives) > 15:
                        st.write(f"... and {len(new_initiatives) - 15} more initiatives")
            else:
                st.write("No clear new initiatives found in the analyzed text.")
            
            # Custom policy area search
            st.write("### Custom Policy Area Search")
            
            custom_policy = st.text_input("Enter a policy area or keyword to search for:")
            
            if custom_policy:
                # Search for mentions
                custom_count = text_content.lower().count(custom_policy.lower())
                st.write(f"'{custom_policy}' is mentioned {custom_count} times in the document.")
                
                # Find relevant sentences
                relevant_sentences = []
                
                for sentence in sent_tokenize(text_content):
                    if custom_policy.lower() in sentence.lower():
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    st.write(f"Found {len(relevant_sentences)} sentences mentioning '{custom_policy}':")
                    
                    # Display sentences
                    with st.expander(f"View sentences mentioning '{custom_policy}'"):
                        for i, sentence in enumerate(relevant_sentences[:10]):  # Limit to first 10 for clarity
                            st.write(f"{i+1}. {sentence}")
                        
                        if len(relevant_sentences) > 10:
                            st.write(f"... and {len(relevant_sentences) - 10} more sentences")
                else:
                    st.warning(f"No sentences found mentioning '{custom_policy}'.")
        
    except Exception as e:
        st.error(f"Error analyzing budget document: {str(e)}")


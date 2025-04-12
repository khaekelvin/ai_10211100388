import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io

def render_election_analysis(datasets):
    st.title("Election Data Analysis")
    
    st.markdown("""
    This section provides comprehensive analysis of Ghana's election data, 
    allowing you to explore trends, regional differences, and party performance 
    across years and regions.
    """)
    
    # Check if election data is loaded
    if datasets['election_data'] is None:
        st.error("Ghana election data is not loaded. Please check the data source.")
        return
    
    df = datasets['election_data']
    
    # Data preview
    st.subheader("Election Data Preview")
    st.dataframe(df.head())
    
    # Basic statistics
    st.subheader("Basic Election Statistics")
    
    # Clean percentage data - extract numerical values
    if 'Votes(%)' in df.columns:
        df['Votes_Percentage'] = df['Votes(%)'].str.rstrip('%').astype('float') / 100.0
    
    # Election overview
    years = sorted(df['Year'].unique())
    selected_year = st.selectbox("Select Election Year", years)
    
    # Filter data by selected year
    year_data = df[df['Year'] == selected_year]
    
    # Display overall statistics for the selected year
    st.write(f"### {selected_year} Election Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_votes = year_data['Votes'].sum()
        st.metric("Total Votes Cast", f"{total_votes:,}")
    
    with col2:
        num_candidates = year_data['Candidate'].nunique()
        st.metric("Number of Candidates", num_candidates)
    
    with col3:
        num_regions = year_data['New Region'].nunique()
        st.metric("Number of Regions", num_regions)
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Party Performance", "Regional Analysis", "Candidate Analysis", "Custom Analysis"])
    
    with tab1:
        st.subheader("Party Performance Analysis")
        
        # Get unique parties
        parties = df['Party'].unique()
        
        # Display party vote shares for selected year
        party_votes = year_data.groupby('Party')['Votes'].sum().reset_index()
        total_votes = party_votes['Votes'].sum()
        party_votes['Vote Share'] = party_votes['Votes'] / total_votes
        party_votes = party_votes.sort_values('Votes', ascending=False)
        
        # Plot party vote shares
        fig = px.pie(
            party_votes, 
            values='Votes', 
            names='Party', 
            title=f'Party Vote Shares in {selected_year}',
            color='Party',
            color_discrete_map={
                'NPP': '#FF0000',  # Red for NPP
                'NDC': '#00A000',  # Green for NDC
                'Others': '#808080',  # Grey for Others
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display vote share table
        party_votes['Vote Share'] = party_votes['Vote Share'].apply(lambda x: f"{x:.2%}")
        party_votes['Votes'] = party_votes['Votes'].apply(lambda x: f"{x:,}")
        st.dataframe(party_votes.reset_index(drop=True))
        
        # Comparison between NPP and NDC
        if 'NPP' in parties and 'NDC' in parties:
            st.subheader("NPP vs NDC Comparison")
            
            # Filter for NPP and NDC data across years and regions
            major_parties_data = df[df['Party'].isin(['NPP', 'NDC'])]
            
            # Group by Year and Party
            year_party_votes = major_parties_data.groupby(['Year', 'Party'])['Votes'].sum().reset_index()
            
            # Create a pivot table for better visualization
            year_party_pivot = year_party_votes.pivot(index='Year', columns='Party', values='Votes').reset_index()
            
            # Calculate total votes per year
            year_total_votes = year_party_votes.groupby('Year')['Votes'].sum().reset_index()
            year_total_votes.columns = ['Year', 'Total Votes']
            
            # Merge with pivot table
            year_party_pivot = pd.merge(year_party_pivot, year_total_votes, on='Year')
            
            # Calculate vote shares
            year_party_pivot['NPP Share'] = year_party_pivot['NPP'] / year_party_pivot['Total Votes']
            year_party_pivot['NDC Share'] = year_party_pivot['NDC'] / year_party_pivot['Total Votes']
            
            # Plot NPP vs NDC vote shares over time
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=year_party_pivot['Year'],
                y=year_party_pivot['NPP Share'],
                name='NPP',
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                x=year_party_pivot['Year'],
                y=year_party_pivot['NDC Share'],
                name='NDC',
                marker_color='green'
            ))
            
            fig.update_layout(
                title='NPP vs NDC Vote Share Over Time',
                xaxis_title='Election Year',
                yaxis_title='Vote Share',
                barmode='group',
                yaxis=dict(tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Regional performance comparison for NPP and NDC
            st.subheader(f"NPP vs NDC Regional Performance in {selected_year}")
            
            # Filter for selected year and major parties
            year_major_parties = year_data[year_data['Party'].isin(['NPP', 'NDC'])]
            
            # Group by Region and Party
            region_party_votes = year_major_parties.groupby(['New Region', 'Party'])['Votes'].sum().reset_index()
            
            # Create a pivot table for better visualization
            region_party_pivot = region_party_votes.pivot(index='New Region', columns='Party', values='Votes').reset_index()
            
            # Calculate total votes per region
            region_total_votes = region_party_votes.groupby('New Region')['Votes'].sum().reset_index()
            region_total_votes.columns = ['New Region', 'Total Votes']
            
            # Merge with pivot table
            region_party_pivot = pd.merge(region_party_pivot, region_total_votes, on='New Region')
            
            # Calculate vote shares
            region_party_pivot['NPP Share'] = region_party_pivot['NPP'] / region_party_pivot['Total Votes']
            region_party_pivot['NDC Share'] = region_party_pivot['NDC'] / region_party_pivot['Total Votes']
            
            # Calculate winning margin (positive for NPP, negative for NDC)
            region_party_pivot['Winning Margin'] = region_party_pivot['NPP Share'] - region_party_pivot['NDC Share']
            
            # Sort by winning margin
            region_party_pivot = region_party_pivot.sort_values('Winning Margin', ascending=False)
            
            # Plot regional performance
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=region_party_pivot['New Region'],
                x=region_party_pivot['NPP Share'],
                name='NPP',
                orientation='h',
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                y=region_party_pivot['New Region'],
                x=region_party_pivot['NDC Share'],
                name='NDC',
                orientation='h',
                marker_color='green'
            ))
            
            fig.update_layout(
                title=f'NPP vs NDC Regional Performance in {selected_year}',
                xaxis_title='Vote Share',
                yaxis_title='Region',
                barmode='group',
                xaxis=dict(tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table with vote shares and winning margins
            display_df = region_party_pivot[['New Region', 'NPP Share', 'NDC Share', 'Winning Margin']].copy()
            display_df['NPP Share'] = display_df['NPP Share'].apply(lambda x: f"{x:.2%}")
            display_df['NDC Share'] = display_df['NDC Share'].apply(lambda x: f"{x:.2%}")
            display_df['Winning Margin'] = display_df['Winning Margin'].apply(lambda x: f"{x:.2%} {'(NPP)' if x > 0 else '(NDC)'}")
            st.dataframe(display_df)
    
    with tab2:
        st.subheader("Regional Analysis")
        
        # Select region
        regions = sorted(df['New Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
        
        # Filter data by selected region
        region_data = df[df['New Region'] == selected_region]
        
        # Display overall statistics for the selected region
        st.write(f"### Analysis for {selected_region}")
        
        # Split by year and party
        region_year_party = region_data.groupby(['Year', 'Party'])['Votes'].sum().reset_index()
        
        # Create a pivot table for better visualization
        region_year_party_pivot = region_year_party.pivot(index='Year', columns='Party', values='Votes').fillna(0).reset_index()
        
        # Calculate total votes per year
        region_year_total = region_year_party.groupby('Year')['Votes'].sum().reset_index()
        region_year_total.columns = ['Year', 'Total Votes']
        
        # Merge with pivot table
        region_year_party_pivot = pd.merge(region_year_party_pivot, region_year_total, on='Year')
        
        # Calculate vote shares for major parties
        if 'NPP' in region_year_party_pivot.columns:
            region_year_party_pivot['NPP Share'] = region_year_party_pivot['NPP'] / region_year_party_pivot['Total Votes']
        
        if 'NDC' in region_year_party_pivot.columns:
            region_year_party_pivot['NDC Share'] = region_year_party_pivot['NDC'] / region_year_party_pivot['Total Votes']
        
        # Plot vote shares over time for selected region
        fig = go.Figure()
        
        if 'NPP' in region_year_party_pivot.columns:
            fig.add_trace(go.Line(
                x=region_year_party_pivot['Year'],
                y=region_year_party_pivot['NPP Share'],
                name='NPP',
                marker_color='red'
            ))
        
        if 'NDC' in region_year_party_pivot.columns:
            fig.add_trace(go.Line(
                x=region_year_party_pivot['Year'],
                y=region_year_party_pivot['NDC Share'],
                name='NDC',
                marker_color='green'
            ))
        
        fig.update_layout(
            title=f'Vote Share Trends in {selected_region}',
            xaxis_title='Election Year',
            yaxis_title='Vote Share',
            yaxis=dict(tickformat='.0%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed party performance for the selected region in the selected year
        st.write(f"### Party Performance in {selected_region} ({selected_year})")
        
        # Filter for selected year and region
        year_region_data = df[(df['Year'] == selected_year) & (df['New Region'] == selected_region)]
        
        # Group by party
        year_region_party = year_region_data.groupby('Party')['Votes'].sum().reset_index()
        total_votes = year_region_party['Votes'].sum()
        year_region_party['Vote Share'] = year_region_party['Votes'] / total_votes
        year_region_party = year_region_party.sort_values('Votes', ascending=False)
        
        # Plot party performance
        fig = px.bar(
            year_region_party,
            x='Party',
            y='Vote Share',
            title=f'Party Performance in {selected_region} ({selected_year})',
            color='Party',
            color_discrete_map={
                'NPP': '#FF0000',  # Red for NPP
                'NDC': '#00A000',  # Green for NDC
            }
        )
        
        fig.update_layout(
            xaxis_title='Party',
            yaxis_title='Vote Share',
            yaxis=dict(tickformat='.0%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display performance table
        year_region_party['Vote Share'] = year_region_party['Vote Share'].apply(lambda x: f"{x:.2%}")
        year_region_party['Votes'] = year_region_party['Votes'].apply(lambda x: f"{x:,}")
        st.dataframe(year_region_party.reset_index(drop=True))
        
        # Regional vote share map (if available)
        st.subheader("Regional Vote Share Map (Work in progress)")
        st.info("This feature will be added in a future update to visualize regional vote shares on a map of Ghana.")
    
    with tab3:
        st.subheader("Candidate Analysis")
        
        # Get unique candidates
        candidates = df['Candidate'].unique()
        
        # Allow selection of candidates
        selected_candidates = st.multiselect(
            "Select Candidates to Compare",
            candidates,
            default=['Nana Akufo Addo', 'John Dramani Mahama'] if 'Nana Akufo Addo' in candidates and 'John Dramani Mahama' in candidates else []
        )
        
        if not selected_candidates:
            st.warning("Please select at least one candidate to analyze.")
        else:
            # Filter for selected candidates
            candidates_data = df[df['Candidate'].isin(selected_candidates)]
            
            # Group by Year and Candidate
            year_candidate_votes = candidates_data.groupby(['Year', 'Candidate'])['Votes'].sum().reset_index()
            
            # Create a pivot table for better visualization
            year_candidate_pivot = year_candidate_votes.pivot(index='Year', columns='Candidate', values='Votes').reset_index()
            
            # Calculate total votes per year (need to use original dataframe for this)
            year_total = df.groupby('Year')['Votes'].sum().reset_index()
            year_total.columns = ['Year', 'Total Votes']
            
            # Merge with pivot table
            year_candidate_pivot = pd.merge(year_candidate_pivot, year_total, on='Year')
            
            # Calculate vote shares for each candidate
            for candidate in selected_candidates:
                if candidate in year_candidate_pivot.columns:
                    year_candidate_pivot[f'{candidate} Share'] = year_candidate_pivot[candidate] / year_candidate_pivot['Total Votes']
            
            # Plot vote shares over time
            fig = go.Figure()
            
            for candidate in selected_candidates:
                if f'{candidate} Share' in year_candidate_pivot.columns:
                    fig.add_trace(go.Line(
                        x=year_candidate_pivot['Year'],
                        y=year_candidate_pivot[f'{candidate} Share'],
                        name=candidate
                    ))
            
            fig.update_layout(
                title='Candidate Vote Share Trends',
                xaxis_title='Election Year',
                yaxis_title='Vote Share',
                yaxis=dict(tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Regional performance comparison for selected candidates
            st.subheader(f"Regional Performance in {selected_year}")
            
            # Filter for selected year and candidates
            year_candidates = df[(df['Year'] == selected_year) & (df['Candidate'].isin(selected_candidates))]
            
            # Group by Region and Candidate
            region_candidate_votes = year_candidates.groupby(['New Region', 'Candidate'])['Votes'].sum().reset_index()
            
            # Create a pivot table for better visualization
            region_candidate_pivot = region_candidate_votes.pivot(index='New Region', columns='Candidate', values='Votes').reset_index()
            
            # Calculate total votes per region for the selected year
            region_total = df[df['Year'] == selected_year].groupby('New Region')['Votes'].sum().reset_index()
            region_total.columns = ['New Region', 'Total Votes']
            
            # Merge with pivot table
            region_candidate_pivot = pd.merge(region_candidate_pivot, region_total, on='New Region')
            
            # Calculate vote shares for each candidate
            for candidate in selected_candidates:
                if candidate in region_candidate_pivot.columns:
                    region_candidate_pivot[f'{candidate} Share'] = region_candidate_pivot[candidate] / region_candidate_pivot['Total Votes']
            
            # Plot regional performance
            fig = px.bar(
                region_candidate_pivot,
                x='New Region',
                y=[f'{candidate} Share' for candidate in selected_candidates if f'{candidate} Share' in region_candidate_pivot.columns],
                title=f'Regional Performance in {selected_year}',
                barmode='group'
            )
            
            fig.update_layout(
                xaxis_title='Region',
                yaxis_title='Vote Share',
                yaxis=dict(tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Custom Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Vote Distribution by Region", "Party Performance Trend", "Stronghold Analysis"]
        )
        
        if analysis_type == "Vote Distribution by Region":
            st.write("### Vote Distribution by Region")
            
            # Select year
            year_for_analysis = st.selectbox("Select Year", years, key="region_dist_year")
            
            # Filter for selected year
            year_data_for_regions = df[df['Year'] == year_for_analysis]
            
            # Group by region and party
            region_party_data = year_data_for_regions.groupby(['New Region', 'Party'])['Votes'].sum().reset_index()
            
            # Create pivot table
            region_party_pivot = region_party_data.pivot(index='New Region', columns='Party', values='Votes').fillna(0)
            
            # Calculate total votes per region
            region_party_pivot['Total'] = region_party_pivot.sum(axis=1)
            
            # Calculate percentages
            for party in region_party_pivot.columns:
                if party != 'Total':
                    region_party_pivot[f'{party} %'] = region_party_pivot[party] / region_party_pivot['Total'] * 100
            
            # Reset index for display
            region_party_pivot = region_party_pivot.reset_index()
            
            # Select visualization type
            viz_type = st.radio("Select Visualization", ["Bar Chart", "Heatmap"])
            
            if viz_type == "Bar Chart":
                # Select parties to include
                parties_in_data = [col for col in region_party_pivot.columns if ' %' in col]
                parties_to_show = st.multiselect(
                    "Select Parties to Show",
                    parties_in_data,
                    default=parties_in_data[:2]
                )
                
                if parties_to_show:
                    fig = px.bar(
                        region_party_pivot,
                        x='New Region',
                        y=parties_to_show,
                        title=f'Vote Distribution by Region ({year_for_analysis})',
                        barmode='group'
                    )
                    
                    fig.update_layout(
                        xaxis_title='Region',
                        yaxis_title='Vote Percentage',
                        xaxis={'categoryorder':'total descending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one party to display.")
            
            else:  # Heatmap
                # Select parties for heatmap
                major_parties = ['NPP %', 'NDC %'] if 'NPP %' in region_party_pivot.columns and 'NDC %' in region_party_pivot.columns else []
                parties_for_heatmap = st.multiselect(
                    "Select Parties for Heatmap",
                    [col for col in region_party_pivot.columns if ' %' in col],
                    default=major_parties
                )
                
                if parties_for_heatmap:
                    # Prepare data for heatmap
                    heatmap_data = region_party_pivot[['New Region'] + parties_for_heatmap].copy()
                    
                    # Convert to long format for heatmap
                    heatmap_data_long = pd.melt(
                        heatmap_data,
                        id_vars=['New Region'],
                        value_vars=parties_for_heatmap,
                        var_name='Party',
                        value_name='Vote Percentage'
                    )
                    
                    # Clean party names
                    heatmap_data_long['Party'] = heatmap_data_long['Party'].str.replace(' %', '')
                    
                    # Create heatmap
                    fig = px.density_heatmap(
                        heatmap_data_long,
                        x='New Region',
                        y='Party',
                        z='Vote Percentage',
                        title=f'Vote Distribution Heatmap ({year_for_analysis})',
                        color_continuous_scale='RdBu_r'
                    )
                    
                    fig.update_layout(
                        xaxis_title='Region',
                        yaxis_title='Party',
                        xaxis={'categoryorder':'total descending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one party for the heatmap.")
        
        elif analysis_type == "Party Performance Trend":
            st.write("### Party Performance Trend Analysis")
            
            # Select parties
            available_parties = df['Party'].unique()
            selected_parties_trend = st.multiselect(
                "Select Parties",
                available_parties,
                default=['NPP', 'NDC'] if 'NPP' in available_parties and 'NDC' in available_parties else []
            )
            
            if not selected_parties_trend:
                st.warning("Please select at least one party.")
            else:
                # Group by year and party
                year_party_trend = df[df['Party'].isin(selected_parties_trend)].groupby(['Year', 'Party'])['Votes'].sum().reset_index()
                
                # Calculate total votes by year
                year_total_trend = df.groupby('Year')['Votes'].sum().reset_index()
                year_total_trend.columns = ['Year', 'Total Votes']
                
                # Merge with party data
                year_party_trend = pd.merge(year_party_trend, year_total_trend, on='Year')
                
                # Calculate vote shares
                year_party_trend['Vote Share'] = year_party_trend['Votes'] / year_party_trend['Total Votes']
                
                # Plot trend
                fig = px.line(
                    year_party_trend,
                    x='Year',
                    y='Vote Share',
                    color='Party',
                    title='Party Performance Trend',
                    markers=True
                )
                
                fig.update_layout(
                    xaxis_title='Election Year',
                    yaxis_title='Vote Share',
                    yaxis=dict(tickformat='.0%')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data table
                display_trend = year_party_trend.copy()
                display_trend['Vote Share'] = display_trend['Vote Share'].apply(lambda x: f"{x:.2%}")
                display_trend['Votes'] = display_trend['Votes'].apply(lambda x: f"{x:,}")
                st.dataframe(display_trend[['Year', 'Party', 'Votes', 'Vote Share']])
        
        elif analysis_type == "Stronghold Analysis":
            st.write("### Party Stronghold Analysis")
            
            # Select year and party
            year_for_stronghold = st.selectbox("Select Year", years, key="stronghold_year")
            
            party_for_stronghold = st.selectbox(
                "Select Party",
                ['NPP', 'NDC'],
                index=0 if 'NPP' in df['Party'].unique() else 1
            )
            
            if party_for_stronghold:
                # Filter data
                stronghold_data = df[(df['Year'] == year_for_stronghold) & (df['Party'] == party_for_stronghold)]
                
                # Group by region
                region_stronghold = stronghold_data.groupby('New Region')['Votes'].sum().reset_index()
                
                # Get total votes per region
                region_total_votes = df[df['Year'] == year_for_stronghold].groupby('New Region')['Votes'].sum().reset_index()
                region_total_votes.columns = ['New Region', 'Total Votes']
                
                # Merge
                region_stronghold = pd.merge(region_stronghold, region_total_votes, on='New Region')
                
                # Calculate vote share
                region_stronghold['Vote Share'] = region_stronghold['Votes'] / region_stronghold['Total Votes']
                
                # Sort by vote share
                region_stronghold = region_stronghold.sort_values('Vote Share', ascending=False)
                
                # Classify regions
                threshold_strong = 0.6  # 60% or more
                threshold_lean = 0.5    # 50-60%
                
                region_stronghold['Classification'] = 'Battleground'
                region_stronghold.loc[region_stronghold['Vote Share'] >= threshold_strong, 'Classification'] = 'Stronghold'
                region_stronghold.loc[(region_stronghold['Vote Share'] >= threshold_lean) & 
                                     (region_stronghold['Vote Share'] < threshold_strong), 'Classification'] = 'Leaning'
                
                # Assign colors based on classification
                color_map = {
                    'Stronghold': '#0343DF',  # Dark blue
                    'Leaning': '#97BBFF',     # Light blue
                    'Battleground': '#BBBBBB'  # Grey
                }
                
                fig = px.bar(
                    region_stronghold,
                    x='New Region',
                    y='Vote Share',
                    title=f'{party_for_stronghold} Regional Support ({year_for_stronghold})',
                    color='Classification',
                    color_discrete_map=color_map,
                    text='Vote Share'
                )
                
                fig.update_traces(texttemplate='%{text:.1%}', textposition='inside')
                
                fig.update_layout(
                    xaxis_title='Region',
                    yaxis_title='Vote Share',
                    yaxis=dict(tickformat='.0%')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display classification table
                display_stronghold = region_stronghold.copy()
                display_stronghold['Vote Share'] = display_stronghold['Vote Share'].apply(lambda x: f"{x:.2%}")
                display_stronghold['Votes'] = display_stronghold['Votes'].apply(lambda x: f"{x:,}")
                st.dataframe(display_stronghold[['New Region', 'Votes', 'Vote Share', 'Classification']])
                
                # Show classifications
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    stronghold_count = len(region_stronghold[region_stronghold['Classification'] == 'Stronghold'])
                    st.metric("Stronghold Regions", stronghold_count)
                
                with col2:
                    leaning_count = len(region_stronghold[region_stronghold['Classification'] == 'Leaning'])
                    st.metric("Leaning Regions", leaning_count)
                
                with col3:
                    battleground_count = len(region_stronghold[region_stronghold['Classification'] == 'Battleground'])
                    st.metric("Battleground Regions", battleground_count)
    
    # Download options
    st.subheader("Download Analysis Data")
    
    # Prepare data for download
    download_options = st.radio(
        "Select data to download:",
        ["Complete Election Data", "Selected Year Data", "Regional Data"]
    )
    
    if download_options == "Complete Election Data":
        download_data = df
        filename = "ghana_election_data_complete.csv"
    elif download_options == "Selected Year Data":
        download_data = df[df['Year'] == selected_year]
        filename = f"ghana_election_data_{selected_year}.csv"
    else:  # Regional Data
        region_for_download = st.selectbox("Select Region for Download", regions)
        download_data = df[df['New Region'] == region_for_download]
        filename = f"ghana_election_data_{region_for_download.replace(' ', '_')}.csv"
    
    # Create download button
    csv = download_data.to_csv(index=False)
    st.download_button(
        label=f"Download {download_options}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

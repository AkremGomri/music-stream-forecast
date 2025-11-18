import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from nixtla import NixtlaClient
import os

# Configure the page
st.set_page_config(page_title="TimeGPT Music Forecast", layout="wide")
st.title("TimeGPT Music Streaming Forecast")

# Initialize Nixtla client
nixtla_client = NixtlaClient(api_key=st.secrets["NIXTLA_API_KEY"])

# Load saved model artifacts
@st.cache_data
def load_model_artifacts():
    with open('./data/timegpt_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    with open('./data/artist_time_series.json', 'r') as f:
        artist_data = json.load(f)
    return metadata, artist_data

# Initialize ALL session state variables at the top
if 'show_dataset_overview' not in st.session_state:
    st.session_state.show_dataset_overview = False
if 'show_historical_details' not in st.session_state:
    st.session_state.show_historical_details = False
if 'show_forecast_details' not in st.session_state:
    st.session_state.show_forecast_details = False
if 'show_forecast' not in st.session_state:
    st.session_state.show_forecast = False
if 'selected_artist' not in st.session_state:
    st.session_state.selected_artist = None
if 'global_stats' not in st.session_state:
    st.session_state.global_stats = None
if 'artist_stats' not in st.session_state:
    st.session_state.artist_stats = {}
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = {}
if 'forecast_triggered' not in st.session_state:
    st.session_state.forecast_triggered = False
if 'back_triggered' not in st.session_state:
    st.session_state.back_triggered = False

try:
    metadata, artist_data = load_model_artifacts()
    
    # Pre-calculate global statistics ONCE
    if st.session_state.global_stats is None:
        with st.spinner("Preparing dataset..."):
            total_artists = len(metadata['artists_list'])
            total_data_points = sum(len(artist_data[artist]) for artist in metadata['artists_list'])
            
            # Calculate date range across all artists
            all_dates = []
            for artist in metadata['artists_list']:
                artist_df = pd.DataFrame(artist_data[artist])
                artist_df['ds'] = pd.to_datetime(artist_df['ds'])
                all_dates.extend(artist_df['ds'].tolist())
            
            date_range_text = "N/A"
            if all_dates:
                min_date = min(all_dates).strftime('%Y/%m/%d')
                max_date = max(all_dates).strftime('%Y/%m/%d')
                date_range_text = f"{min_date} : {max_date}"
            
            avg_points = total_data_points / total_artists if total_artists > 0 else 0
            
            st.session_state.global_stats = {
                'total_artists': total_artists,
                'total_data_points': total_data_points,
                'date_range_text': date_range_text,
                'avg_points': avg_points
            }
    
    # Handle triggers BEFORE any view rendering
    if st.session_state.forecast_triggered:
        st.session_state.show_forecast = True
        st.session_state.show_forecast_details = False
        st.session_state.forecast_triggered = False
        
    if st.session_state.back_triggered:
        st.session_state.show_forecast = False
        st.session_state.back_triggered = False
    
    # Single searchable dropdown in main view
    st.header("Artist Selection")
    
    artist_list = metadata['artists_list']
    
    # Create a single searchable dropdown
    selected_artist = st.selectbox(
        "Search and select artist:",
        options=artist_list,
        index=None,
        placeholder="Start typing to search artists...",
        key="artist_search_select"
    )
    
    # Update session state when artist is selected
    if selected_artist and selected_artist != st.session_state.selected_artist:
        st.session_state.selected_artist = selected_artist
        st.session_state.show_forecast = False
        st.session_state.forecast_triggered = False
        st.session_state.back_triggered = False
        st.session_state.show_historical_details = False
        
        # Pre-calculate artist statistics
        if selected_artist in artist_data and selected_artist not in st.session_state.artist_stats:
            artist_df = pd.DataFrame(artist_data[selected_artist])
            artist_df['ds'] = pd.to_datetime(artist_df['ds'])
            
            # Pre-calculate everything for this artist
            display_df = artist_df.copy()
            display_df['ds'] = display_df['ds'].dt.strftime('%Y-%m-%d')
            display_df = display_df.rename(columns={'ds': 'Date', 'y': 'Streams'})
            
            st.session_state.artist_stats[selected_artist] = {
                'artist_df': artist_df,
                'display_df': display_df,
                'total_points': len(artist_df),
                'avg_streams': artist_df['y'].mean(),
                'max_streams': artist_df['y'].max(),
                'min_streams_val': artist_df['y'].min(),
                'start_date': artist_df['ds'].min().strftime('%Y-%m-%d'),
                'end_date': artist_df['ds'].max().strftime('%Y-%m-%d')
            }
    
    # Sidebar for prediction parameters only
    st.sidebar.header("Prediction Settings")
    
    # Prediction parameters
    prediction_horizon = st.sidebar.slider(
        "Weeks to Predict", 
        min_value=4, 
        max_value=52, 
        value=12,
        help="Number of weeks to forecast"
    )
    
    include_history = st.sidebar.checkbox(
        "Include History in Plot", 
        value=True,
        help="Show historical data along with predictions"
    )
    
    # NEW: Single confidence level selector with fixed 20% and 99%
    selected_confidence = st.sidebar.slider(
        "Confidence Level",
        min_value=20,
        max_value=99,
        value=70,
        help="Select the confidence level for evaluation (20% and 99% are fixed for reference)"
    )
    
    confidence_levels = [selected_confidence]
    
    # Artist evaluation settings
    st.sidebar.header("Artist Evaluation Criteria")
    
    min_annual_streams = st.sidebar.number_input(
        "Minimum Annual Streams (in millions)",
        min_value=0.0,
        max_value=1000.0,
        value=0.5,
        step=0.1,
        help="Minimum streams per year to be considered interesting (in millions)"
    )
    
    max_annual_streams = st.sidebar.number_input(
        "Maximum Annual Streams (in millions)",
        min_value=0.0,
        max_value=1000.0,
        value=20.0,
        step=1.0,
        help="Maximum streams per year to be considered interesting (in millions)"
    )
    
    # Convert to actual stream counts (millions to units)
    min_streams = min_annual_streams * 1_000_000
    max_streams = max_annual_streams * 1_000_000
    
    st.divider()
    # Main content area
    if not st.session_state.selected_artist:
        # Initial state - no artist selected
        st.info("👆 Please select an artist above to begin analysis")
        
        st.write("### How to use:")
        st.write("1. Use the dropdown above to search and select an artist")
        st.write("2. View the artist's historical data and statistics")
        st.write("3. Adjust prediction settings in the sidebar")
        st.write("4. Click 'Generate Forecast' to see predictions")
        
        # Show/Hide Details button for overview
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Show/hide data", 
                        key="dataset_overview_btn"):
                st.session_state.show_dataset_overview = not st.session_state.show_dataset_overview
        
        # Show statistics about the dataset (only if details are shown)
        if st.session_state.show_dataset_overview:
            st.header("📊 Dataset Overview")
            
            # Display pre-calculated metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Artists", st.session_state.global_stats['total_artists'])
            with col2:
                st.metric("Total Data Points", st.session_state.global_stats['total_data_points'])
            with col3:
                st.markdown(f"**Date Range**")
                st.markdown(st.session_state.global_stats['date_range_text'])
            with col4:
                st.metric("Avg Data per Artist", f"{st.session_state.global_stats['avg_points']:.1f}")
        
    elif st.session_state.selected_artist and not st.session_state.show_forecast:
        # Show historical data for selected artist
        if st.session_state.selected_artist in artist_data:
            artist_stats = st.session_state.artist_stats[st.session_state.selected_artist]
            artist_df = artist_stats['artist_df']
            
            st.header(f"Historical Data")

            # Plot historical data (always shown)
            fig_historical = go.Figure()
            fig_historical.add_trace(go.Scatter(
                x=artist_df['ds'], 
                y=artist_df['y'],
                mode='lines+markers',
                name='Historical Streams',
                line=dict(color='#1f77b4')
            ))
            
            fig_historical.update_layout(
                title=dict(
                    text="Historical Streaming Data",
                    x=0.5,  # center the title
                    xanchor="center",
                    font=dict(size=20)
                ),
                xaxis_title="Date",
                yaxis_title="Streams",
                height=400,
                showlegend=True
            )

            fig_historical.add_annotation(
                xref="paper", yref="paper",
                x=0, y=1.05,  # position above the plot
                showarrow=False,
                text=f"Artist: {st.session_state.selected_artist}",
                font=dict(size=14)
            )
            
            st.plotly_chart(fig_historical, use_container_width=True)

            # Prediction button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🎯 Generate Forecast", type="primary", use_container_width=True, key="generate_forecast_btn"):
                    st.session_state.forecast_triggered = True
                    st.rerun()
            
            # Show/Hide Details button
            if st.button("Show/hide data", 
                        key="historical_details_btn"):
                st.session_state.show_historical_details = not st.session_state.show_historical_details

            # Show details only if toggled
            if st.session_state.show_historical_details:
                # Show the pre-calculated data table
                st.subheader("📋 Data Table")
                st.dataframe(artist_stats['display_df'].style.format({'Streams': '{:.0f}'}), use_container_width=True)

                # Statistics with improved date display
                st.subheader("📈 Statistics Summary")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Data Points", artist_stats['total_points'])
                with col2:
                    st.markdown(f"**Date Range**")
                    st.markdown(f"**Start:** {artist_stats['start_date']}")
                    st.markdown(f"**End:** {artist_stats['end_date']}")
                with col3:
                    st.metric("Average Streams", f"{artist_stats['avg_streams']:.0f}")
                with col4:
                    st.metric("Max Streams", f"{artist_stats['max_streams']:.0f}")
                with col5:
                    st.metric("Min Streams", f"{artist_stats['min_streams_val']:.0f}")
    
    elif st.session_state.show_forecast and st.session_state.selected_artist:
        # Show forecast results
        if st.session_state.selected_artist in artist_data:
            artist_stats = st.session_state.artist_stats[st.session_state.selected_artist]
            artist_df = artist_stats['artist_df']
            
            # Generate forecast if not already done
            forecast_key = f"{st.session_state.selected_artist}_{prediction_horizon}_{selected_confidence}"
            
            if forecast_key not in st.session_state.forecast_data:
                with st.spinner("Generating TimeGPT forecast..."):
                    try:
                        forecast_df = nixtla_client.forecast(
                            df=artist_df,
                            h=prediction_horizon,
                            finetune_steps=metadata['fine_tune_epochs'],
                            time_col='ds',
                            target_col='y',
                            level=confidence_levels,
                            add_history=include_history
                        )
                        
                        # Pre-calculate ALL forecast data
                        future_mask = forecast_df['ds'] > artist_df['ds'].max()
                        forecast_only = forecast_df[future_mask]
                        total_predicted_streams = forecast_only['TimeGPT'].sum()
                        weeks_per_year = 52
                        annualized_streams = total_predicted_streams * (weeks_per_year / prediction_horizon)
                        
                        # NEW: Get confidence bounds for selected confidence level
                        lower_bound_col = f'TimeGPT-lo-{selected_confidence}'
                        upper_bound_col = f'TimeGPT-hi-{selected_confidence}'
                        
                        # Calculate annualized confidence bounds
                        annualized_lower = forecast_only[lower_bound_col].sum() * (weeks_per_year / prediction_horizon)
                        annualized_upper = forecast_only[upper_bound_col].sum() * (weeks_per_year / prediction_horizon)
                        
                        # NEW: Calculate precision score
                        if (annualized_upper + annualized_lower) == 0:
                            precision_score = 0
                        else:
                            precision_score = (annualized_upper - annualized_lower) / ((annualized_upper + annualized_lower) / 2)
                        
                        # NEW: Determine evaluation tier
                        lower_in_range = min_streams <= annualized_lower <= max_streams
                        upper_in_range = min_streams <= annualized_upper <= max_streams
                        point_in_range = min_streams <= annualized_streams <= max_streams
                        
                        if lower_in_range and upper_in_range:
                            evaluation_tier = "HIGH_CONFIDENCE_DEAL"
                            tier_color = "green"
                            tier_icon = "✅"
                            tier_name = "High Confidence Deal"
                            recommendation = "Safe investment with predictable returns.",
                            justification = f"The **{selected_confidence}% confidence interval** [{annualized_lower/1_000_000:.1f}M, {annualized_upper/1_000_000:.1f}M] is fully **within target range** [{min_streams/1_000_000:.1f}M, {max_streams/1_000_000:.1f}M]."
                        elif not lower_in_range and upper_in_range:
                            evaluation_tier = "GROWTH_POTENTIAL_DEAL"
                            tier_color = "orange"
                            tier_icon = "📈"
                            tier_name = "Growth Potential Deal"
                            recommendation = f"A risky yet potentially interesting opportunity.",
                            justification = f"** The Lower bound {annualized_lower/1_000_000:.1f}M below target **{min_streams/1_000_000:.1f}M**, but upper bound **{annualized_upper/1_000_000:.1f}M** within range **[{min_streams/1_000_000:.1f}M, {max_streams/1_000_000:.1f}M]**."
                        elif lower_in_range and not upper_in_range:
                            evaluation_tier = "COMPETITIVE_PREMIUM_DEAL"
                            tier_color = "blue"
                            tier_icon = "🏆"
                            tier_name = "Competitive Premium Deal"
                            recommendation = f"High-demand artist, expect competition.",
                            justification = f"**Upper bound {annualized_upper/1_000_000:.1f}M above target {max_streams/1_000_000:.1f}M, but lower bound {annualized_lower/1_000_000:.1f}M within range[{min_streams/1_000_000:.1f}M, {max_streams/1_000_000:.1f}M]."
                        else:
                            evaluation_tier = "OUTSIDE_TARGET_RANGE"
                            tier_color = "red"
                            tier_icon = "❌"
                            tier_name = "Outside Target Range"
                            recommendation = f"Too low expectations.",
                            justification = f"Entire confidence interval [{annualized_lower/1_000_000:.1f}M, {annualized_upper/1_000_000:.1f}M] outside target range [{min_streams/1_000_000:.1f}M, {max_streams/1_000_000:.1f}M]."
                        
                        # Pre-calculate forecast display data
                        display_forecast = forecast_only[['ds', 'TimeGPT', lower_bound_col, upper_bound_col]].copy()
                        display_forecast['ds'] = display_forecast['ds'].dt.strftime('%Y-%m-%d')
                        display_forecast = display_forecast.rename(columns={
                            'ds': 'Date', 
                            'TimeGPT': 'Streams',
                            lower_bound_col: f'Lower {selected_confidence}%',
                            upper_bound_col: f'Upper {selected_confidence}%'
                        })
                        display_forecast['Type'] = 'Forecast'
                        
                        # Pre-calculate forecast stats
                        avg_forecast = forecast_only['TimeGPT'].mean()
                        max_forecast = forecast_only['TimeGPT'].max()
                        min_forecast = forecast_only['TimeGPT'].min()
                        
                        st.session_state.forecast_data[forecast_key] = {
                            'forecast_df': forecast_df,
                            'forecast_only': forecast_only,
                            'total_predicted_streams': total_predicted_streams,
                            'annualized_streams': annualized_streams,
                            'annualized_lower': annualized_lower,
                            'annualized_upper': annualized_upper,
                            'precision_score': precision_score,
                            'evaluation_tier': evaluation_tier,
                            'tier_name': tier_name,
                            'tier_icon': tier_icon,
                            'tier_color': tier_color,
                            'recommendation': recommendation,
                            'justification': justification,
                            'display_forecast': display_forecast,
                            'avg_forecast': avg_forecast,
                            'max_forecast': max_forecast,
                            'min_forecast': min_forecast,
                            'lower_bound_col': lower_bound_col,
                            'upper_bound_col': upper_bound_col
                        }
                        
                        st.success("✅ Forecast generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        if st.button("← Back to Historical Data", key="error_back_btn"):
                            st.session_state.back_triggered = True
                            st.rerun()
                        st.stop()
            else:
                # Use pre-calculated forecast data
                forecast_data = st.session_state.forecast_data[forecast_key]
                forecast_df = forecast_data['forecast_df']
                forecast_only = forecast_data['forecast_only']
                total_predicted_streams = forecast_data['total_predicted_streams']
                annualized_streams = forecast_data['annualized_streams']
                annualized_lower = forecast_data['annualized_lower']
                annualized_upper = forecast_data['annualized_upper']
                precision_score = forecast_data['precision_score']
                evaluation_tier = forecast_data['evaluation_tier']
                tier_name = forecast_data['tier_name']
                tier_icon = forecast_data['tier_icon']
                tier_color = forecast_data['tier_color']
                recommendation = forecast_data['recommendation']
                justification = forecast_data['justification']
                display_forecast = forecast_data['display_forecast']
                avg_forecast = forecast_data['avg_forecast']
                max_forecast = forecast_data['max_forecast']
                min_forecast = forecast_data['min_forecast']
                lower_bound_col = forecast_data['lower_bound_col']
                upper_bound_col = forecast_data['upper_bound_col']
            
            # Back button to return to historical data
            if st.button("← Back to Historical Data", type="secondary", key="back_btn"):
                st.session_state.back_triggered = True
                st.rerun()

            st.header(f"Forecast Results")
            
            # Create interactive forecast plot (always shown)
            fig = go.Figure()
            
            # Always show complete historical data
            fig.add_trace(go.Scatter(
                x=artist_df['ds'], 
                y=artist_df['y'],
                mode='lines',
                name='Historical Data',
                line=dict(color='#1f77b4')
            ))
            
            # Add forecast data
            fig.add_trace(go.Scatter(
                x=forecast_only['ds'],
                y=forecast_only['TimeGPT'],
                mode='lines+markers',
                name='Future Forecast',
                line=dict(color='#ff7f0e', dash='dash')
            ))
            
            # Add confidence intervals for future (show all three levels)
            for level in confidence_levels:
                lo_col = f'TimeGPT-lo-{level}'
                hi_col = f'TimeGPT-hi-{level}'
                if lo_col in forecast_only.columns and hi_col in forecast_only.columns:
                    # Use different opacity for selected confidence level
                    opacity = 0.5 if level != selected_confidence else 0.3
                    name_suffix = " (Evaluation)" if level == selected_confidence else f" ({level}%)"
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_only['ds'],
                        y=forecast_only[lo_col],
                        fill=None,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_only['ds'],
                        y=forecast_only[hi_col],
                        fill='tonexty',
                        mode='lines',
                        name=f'{level}% Confidence{name_suffix}',
                        line=dict(width=0),
                        opacity=opacity
                    ))
            
            fig.update_layout(
                title=dict(
                    text=f"{prediction_horizon} weeks Streaming forecast Data",
                    x=0.5,  # center the title
                    xanchor="center",
                    font=dict(size=20)
                ),
                xaxis_title="Date",
                yaxis_title="Streams",
                height=500,
                showlegend=True
            )

            fig.add_annotation(
                xref="paper", yref="paper",
                x=0, y=1.05,  # position above the plot
                showarrow=False,
                text=f"Artist: {st.session_state.selected_artist}",
                font=dict(size=14)
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Display confidence interval analysis
            st.subheader("Analysis and Interpretation")

            col1, col2, col3 = st.columns(3)

            with col1:
                delta_lower = (annualized_lower - min_streams) / 1_000_000
                st.metric(
                    f"Lower Bound ({selected_confidence}%)", 
                    f"{annualized_lower/1_000_000:.1f}M",
                    delta=f"{abs(delta_lower):.1f}M" if delta_lower != 0 else None,
                    delta_color="normal" if delta_lower >= 0 else "inverse"
                )
                lower_status = "✅ Within range" if min_streams <= annualized_lower <= max_streams else "❌ Below target"
                st.write(lower_status)
                st.caption(f"Worst-case scenario")

            with col2:
                delta_point = (annualized_streams - min_streams) / 1_000_000
                st.metric(
                    "Point Forecast", 
                    f"{annualized_streams/1_000_000:.1f}M",
                    delta=f"{abs(delta_point):.1f}M" if delta_point != 0 else None,
                    delta_color="normal" if delta_point >= 0 else "inverse"
                )
                point_status = "✅ Within range" if min_streams <= annualized_streams <= max_streams else "❌ Outside range"
                st.write(point_status)
                st.caption(f"Most likely outcome")

            with col3:
                delta_upper = (annualized_upper - max_streams) / 1_000_000
                st.metric(
                    f"Upper Bound ({selected_confidence}%)", 
                    f"{annualized_upper/1_000_000:.1f}M",
                    delta=f"{abs(delta_upper):.1f}M" if delta_upper != 0 else None,
                    delta_color="normal" if delta_upper <= 0 else "inverse"
                )
                upper_status = "✅ Within range" if min_streams <= annualized_upper <= max_streams else "❌ Above target"
                st.write(upper_status)
                st.caption(f"Best-case scenario")

            st.write(f"This means there's a **{selected_confidence}% probability** that the actual **annual streams will fall between** [{annualized_lower/1_000_000:.1f}M, {annualized_upper/1_000_000:.1f}M].")
            st.write(f"Notice how {justification.lower()}")

            # ENHANCED ARTIST EVALUATION SECTION
            st.subheader("Action / Recommendation")

            # Display tier evaluation with color coding
            if tier_color == "green":
                st.success(f"{tier_icon} **{recommendation[0]}**")
            elif tier_color == "orange":
                st.warning(f"{tier_icon} **{recommendation[0]}**")
            elif tier_color == "blue":
                st.info(f"{tier_icon} **{recommendation[0]}**")
            else:
                st.error(f"{tier_icon} **{recommendation[0]}**")

            st.divider()

            # Confidence interval probability explanatio

            # st.write(f"**Recommendation:** {recommendation}")
            # # Display Dispersion Score
            # st.subheader("Forecast Precision")
            
            # precision_col1, precision_col2 = st.columns([1, 2])
            
            # with precision_col1:
            #     st.metric("Dispersion Score", f"{precision_score:.2f}")
                
            #     if precision_score < 0.3:
            #         precision_rating = "🟢 High Precision"
            #     elif precision_score < 0.6:
            #         precision_rating = "🟡 Medium Precision"
            #     else:
            #         precision_rating = "🔴 Low Precision"
                    
            #     st.write(f"**Rating:** {precision_rating}")
            
            # with precision_col2:
            #     st.write("**Interpretation:**")
            #     st.write(f"Interval width: {annualized_upper - annualized_lower:,.0f} streams")
            #     st.write(f"Relative to lower bound: {precision_score:.1%}")
            #     st.info("Lower scores indicate more precise forecasts")

            # Show/Hide Details button for forecast
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Show/hide data", 
                            key="forecast_details_btn"):
                    st.session_state.show_forecast_details = not st.session_state.show_forecast_details
            
            # Show details only if toggled
            if st.session_state.show_forecast_details:
                # Show pre-calculated forecast data table
                st.subheader("📋 Forecast Data Table")
                st.dataframe(display_forecast.style.format({
                    'Streams': '{:.0f}',
                    f'Lower {selected_confidence}%': '{:.0f}',
                    f'Upper {selected_confidence}%': '{:.0f}'
                }), use_container_width=True)

                # Forecast summary metrics
                st.subheader("Forecast Summary")

                # Show pre-calculated forecast stats
                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Forecast Periods", len(forecast_only))
                col2.metric("Average Forecast", f"{avg_forecast:.0f}")
                col3.metric("Max Forecast", f"{max_forecast:.0f}")
                col4.metric("Min Forecast", f"{min_forecast:.0f}")

                # Export to Excel functionality
                import io
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    display_forecast.to_excel(writer, sheet_name='Forecast', index=False)
                    
                writer.close()

                st.download_button(
                    label="📥 Download Forecast as Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"{st.session_state.selected_artist}_forecast.xlsx",
                    mime="application/vnd.ms-excel",
                    key="download-excel"
                )


except FileNotFoundError:
    st.error("Model artifacts not found. Please run the Colab notebook first to generate the required files.")
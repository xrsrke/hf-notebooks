import streamlit as st
import wandb
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("The internal states of neural networks")

# Initialize wandb API
api = wandb.Api()

# Replace with your project name
project_name = "fp8_for_nanotron"

@st.cache_data
def fetch_data(metric_filters, x_axis_options, name_filter):
    runs = api.runs(
        project_name,
        filters={"display_name": {"$regex": f"^{name_filter}"}},
        per_page=20
    )

    data = []
    for run in runs:
        config = run.config
        
        dp = config.get('parallelism', {}).get('dp', 1)
        batch_accumulation = config.get('tokens', {}).get('batch_accumulation_per_replica', 1)
        micro_batch_size = config.get('tokens', {}).get('micro_batch_size', 1)
        
        batch_size = dp * batch_accumulation * micro_batch_size

        # Fetch all history for lm_loss and iteration_step        
        for row in run.scan_history(keys=["lm_loss", "iteration_step"], page_size=1000):
            run_data = {
                'batch_size': batch_size,
                'name': run.name,
                'iteration_step': row.get('iteration_step'),
                'lm_loss': row.get('lm_loss')
            }
            
            for filter in metric_filters + x_axis_options:
                if filter in row:
                    run_data[filter] = row[filter]
            
            data.append(run_data)

    df = pd.DataFrame(data).drop_duplicates()
    return df

# Create two main tabs
summary_tab, internal_state_tab = st.tabs(["Summary", "Internal State"])

# Sidebar
st.sidebar.header(f"Project: {project_name}")

# Add name filter input and submit button to sidebar
name_filter = st.sidebar.text_input("Filter runs (name starts with)", value="exp704ba")
submit_button = st.sidebar.button("Submit")

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None

# Load data when submit button is clicked
if submit_button:
    with st.spinner('Loading data...'):
        st.session_state.df = fetch_data(st.session_state.metric_filters, st.session_state.x_axis_options, name_filter)
    st.sidebar.success(f"Data loaded: {len(st.session_state.df['name'].unique())} runs")

# Display number of runs in sidebar
if st.session_state.df is not None:
    st.sidebar.markdown(f"### Data Statistics")
    st.sidebar.write(f"Total number of runs: {len(st.session_state.df['name'].unique())}")


with summary_tab:
    if st.session_state.df is not None:
        df = st.session_state.df
        
        if 'iteration_step' in df.columns and 'lm_loss' in df.columns:
            st.subheader("LM Loss vs Iteration Step")
            
            fig = go.Figure()
            
            for run in df['name'].unique():
                run_data = df[df['name'] == run].sort_values('iteration_step')
                fig.add_trace(go.Scatter(
                    x=run_data['iteration_step'],
                    y=run_data['lm_loss'],
                    mode='lines',
                    name=run
                ))
            
            fig.update_layout(
                xaxis_title="Iteration Step",
                yaxis_title="LM Loss",
                height=600,
                margin=dict(l=50, r=50, t=30, b=50),
                autosize=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Description
            This line plot shows the Language Model (LM) Loss versus the Iteration Step for each run in the dataset. 
            Each line represents a different run, allowing you to compare the loss progression across different runs over time.
            
            - X-axis: Iteration Step
            - Y-axis: LM Loss
            
            This plot now shows all logged LM Loss values for each iteration step, providing a more detailed view of the loss progression throughout each run.
            
            You can hover over the lines to see exact values and use the legend below the graph to toggle specific runs on or off.
            """)
        else:
            st.warning("Required columns 'iteration_step' or 'lm_loss' not found in the data.")
    else:
        st.warning("Please enter a name filter and click Submit to load data.")

with internal_state_tab:
    # Initialize session state for metric filters and x-axis options if they don't exist
    if 'metric_filters' not in st.session_state:
        st.session_state.metric_filters = ['fp32_new_changes_in_p:rms']

    if 'x_axis_options' not in st.session_state:
        st.session_state.x_axis_options = ['batch_size']

    # Function to add a new filter or x-axis option
    def add_item(item_type):
        new_item = st.session_state[f'new_{item_type}']
        if new_item and new_item not in st.session_state[f'{item_type}s']:
            st.session_state[f'{item_type}s'].append(new_item)
        st.session_state[f'new_{item_type}'] = ""

    55# Display current filters and x-axis options and allow removal
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Metric Filters")
        for i, item in enumerate(st.session_state.metric_filters):
            cols = st.columns([3, 1])
            cols[0].text(item)
            if cols[1].button('Remove', key=f'remove_metric_filter_{i}'):
                st.session_state.metric_filters.pop(i)
                st.experimental_rerun()
        
        st.text_input("Add new filter", key="new_metric_filter")
        st.button("Add Filter", on_click=add_item, args=('metric_filter',))

    with col2:
        st.subheader("X-Axis Options")
        for i, item in enumerate(st.session_state.x_axis_options):
            cols = st.columns([3, 1])
            cols[0].text(item)
            if cols[1].button('Remove', key=f'remove_x_axis_option_{i}'):
                st.session_state.x_axis_options.pop(i)
                st.experimental_rerun()
        
        st.text_input("Add new x-axis option", key="new_x_axis_option")
        st.button("Add X-Axis Option", on_click=add_item, args=('x_axis_option',))

    if st.session_state.df is not None:
        df = st.session_state.df

        # Create tabs for each x-axis option
        tabs = st.tabs(st.session_state.x_axis_options)

        for i, x_axis in enumerate(st.session_state.x_axis_options):
            with tabs[i]:
                st.header(f"Graphs for X-Axis: {x_axis}")
                
                if x_axis not in df.columns:
                    st.warning(f"No data found for x-axis option '{x_axis}'")
                    continue
                
                # Filter out non-numeric values and sort
                x_values = pd.to_numeric(df[x_axis], errors='coerce').dropna().sort_values()
                
                # Calculate range limits (10th percentile to 90th percentile)
                x_min, x_max = x_values.quantile([0.1, 0.9])
                
                # Allow user to adjust range
                col1, col2 = st.columns(2)
                with col1:
                    user_x_min = st.number_input(f"Minimum {x_axis}", value=float(x_min), key=f"min_{x_axis}")
                with col2:
                    user_x_max = st.number_input(f"Maximum {x_axis}", value=float(x_max), key=f"max_{x_axis}")
                
                # Filter x_values based on user-defined range
                x_values = x_values[(x_values >= user_x_min) & (x_values <= user_x_max)].unique()

                for filter in st.session_state.metric_filters:
                    st.subheader(f"Heatmap for metrics containing '{filter}'")

                    matching_metrics = [col for col in df.columns if filter in col]
                    
                    if not matching_metrics:
                        st.warning(f"No metrics found containing '{filter}'")
                        continue

                    heatmap_data = np.zeros((len(matching_metrics), len(x_values)))

                    for i, metric in enumerate(matching_metrics):
                        for j, x_value in enumerate(x_values):
                            value = df[df[x_axis] == x_value][metric].mean()
                            heatmap_data[i, j] = value if not np.isnan(value) else np.nan

                    heatmap_data = np.log10(heatmap_data)

                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data,
                        x=x_values,
                        y=matching_metrics,
                        colorscale='Viridis',
                        hoverongaps=False,
                        colorbar=dict(
                            title='log10(Value)',
                            titleside='right'
                        )
                    ))

                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title='Metric',
                        xaxis_type='log',
                        height=min(600, max(300, 30 * len(matching_metrics))),
                        margin=dict(l=50, r=50, t=30, b=50),
                        autosize=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show Raw Data"):
            st.subheader("Raw Data")
            st.dataframe(df)

        st.markdown("### Description")
        st.write("""
        These heatmaps display the logarithm (base 10) of the metrics matching the specified filters across different x-axis values. 
        For each x-axis option and filter combination, a separate heatmap is generated showing all matching metrics.
        The x-axis represents the selected x-axis option (on a logarithmic scale), while the y-axis shows the matching metric names. 
        The color intensity indicates the log10 value of the metric, with darker colors representing lower values and brighter colors representing higher values.
        Use the Metric Filters and X-Axis Options sections to add or remove filters and x-axis options.
        You can adjust the range of x-axis values using the input fields above each graph.
        Use the sidebar to filter runs by the start of their name and click the Submit button to load the data.
        The number of runs loaded is displayed in the sidebar.
        """)
    else:
        st.warning("Please enter a name filter and click Submit to load data.")
from functools import lru_cache
from girder_client import GirderClient
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import os
from periodictable import Al, V, Cr, Mn, Fe, Co, Ni, Cu
import logging
import plotly.graph_objects as go

query_terms = ["AAA", "AAB", "AAC", "AAD", "AAE", "BAA", "BBA", "BBB", "BBC"]

# Configure the logging
logging.basicConfig(level=logging.DEBUG,  # Adjust the level as needed
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def total_molar_mass(composition):
    molar_mass = 0
    for element, value in composition.items():
        molar_mass += value / getattr(globals()[element], 'mass')
    return molar_mass


def percent_composition(composition, total):
    return {element: value / total for element, value in composition.items()}

@lru_cache
def query(campaign):
    client = GirderClient(apiUrl=os.environ['GIRDER_API_URL'])
    client.token = os.environ['GIRDER_TOKEN']
    raw_data = client.get(
        'entry/search', parameters={'query': f'^{campaign}.._VAM-.', 'limit': 1000}
    )
    data = {}
    for entry in raw_data:
        entry = entry['data']
        sample_id = entry['sampleId'].split('_')
        # logging.debug(sample_id)
        if len(sample_id) > 2:
            subsample_id = sample_id[-1]
        else:
            subsample_id = None
        sample_id = '_'.join(sample_id[:2])

        if sample_id not in data:
            data[sample_id] = {}

        if entry['suffix'] == 'Tensile':
            for key in entry['Results']:
                data[sample_id][f'{key}.{subsample_id}'] = entry['Results'][key]
        elif entry['suffix'] == 'Syn' and 'Material Preparation' in entry:
            target_mass = entry['Material Preparation']['Target Mass']
            total_mass = round(float(target_mass.pop('Total')))
            sum_molar = total_molar_mass(target_mass)
            data[sample_id].update({
                f'Target Composition (%).{element}': round(
                    (value / getattr(globals()[element], 'mass')) / sum_molar * 100
                )
                for element, value in target_mass.items()
            })
        elif entry['suffix'] == 'EDS':
            for element, value in entry['Results']['Measured Composition (%)'].items():
                data[sample_id][f'Measured Composition (%).{element}'] = value
        elif entry['suffix'] == 'XRD':
            try:
                results = entry['XRD Process Overview']['TAMU Instrument Details']['Results']
                phase = results['Phase Information'][0]
                data[sample_id]['XRD.Phase'] = phase['Structure']
                data[sample_id]['XRD.Lattice Parameters'] = phase['a']
            except: 
                continue

    types = {
        'UTS/YS Ratio.a': float,
        'UTS/YS Ratio.b': float
    }

    # Create DataFrame from the data
    df = pd.DataFrame.from_dict(data, orient='index')

    # Check that all required columns are present in the DataFrame
    missing_columns = [col for col in types if col not in df.columns]

    if missing_columns:
        logging.debug(f"Warning: Missing columns: {', '.join(missing_columns)}")
        return df
    else:
        df = df.astype(types)
        
    # logging.debug("Done.")
    return df



@callback(
    [Output('indicator-graphic', 'figure'),
     Output('error-message', 'children')],
    [Input('campaign-column', 'value'),
     Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('color-column', 'value'),
     Input('size-column', 'value')]
)
def update_graph(campaign, xaxis_column_name, yaxis_column_name,
                color_column_name, size_column_name):    

    try:
        df = query(campaign)
        
        # Check if all required columns are in the dataframe
        required_columns = [xaxis_column_name, yaxis_column_name, color_column_name, size_column_name]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if df.empty:
            return go.Figure(), 'Data cannot be found. The graph cannot be generated.'
        elif missing_columns:
            fig = go.Figure()
            fig.update_layout(
                title='Missing Columns',
                annotations=[
                    dict(
                        text=f'Error: Missing columns: {", ".join(missing_columns)}',
                        x=0.5,
                        y=0.5,
                        font_size=20,
                        showarrow=False,
                        xref='paper',
                        yref='paper'
                    )
                ],
                xaxis_title=xaxis_column_name,
                yaxis_title=yaxis_column_name
            )
            return fig, f'The following columns are missing: {", ".join(missing_columns)}'

        fig = px.scatter(df, x=xaxis_column_name,
                         y=yaxis_column_name,
                         size=size_column_name,
                         color=color_column_name,
                         hover_data={  # Here, you can add more columns from `df` for hover data
                                'sample': df.index   # Example: Add Material Name to hover
                            }
                        )
        return fig, ''

    except Exception as e:
        print(df)
        print(df.isna().sum())
        return go.Figure(), f'An error occurred: {str(e)}'

def serve_layout():
    default_campaign = 'BAA'    
    df = query(default_campaign)
    
    numeric_df = df.select_dtypes(include=[int, float])
    cols = sorted(df.columns)
    numeric_cols = sorted(numeric_df)
    logging.debug(cols)
    
    return html.Div([
        html.Div([
            html.Label(  
                'Campaign', 
                style={'font-weight': 'bold', 'text-align': 'right', 'offset':1}
            ),
            dcc.Dropdown(
                query_terms,
                default_campaign,
                id='campaign-column',
                style={ 'width': '100px'}
            ), 
        ], style={'margin-bottom': '10px'}),
        html.Div([                 
            html.Div([         
                html.Label(  
                    'X-Axis', 
                    style={'font-weight': 'bold', 'text-align': 'right','offset':1}
                ),
                dcc.Dropdown(
                    numeric_cols,
                    'Yield Strength.a',
                    id='xaxis-column',
                    style={ 'width': '80%', 'margin-bottom': '10px'}
                ),
                html.Label(  
                    'Color', 
                    style={'font-weight': 'bold', 'text-align': 'right','offset':1}
                ),             
                dcc.Dropdown(
                    cols,
                    'Target Composition (%).Fe',
                    id='color-column',
                    style={ 'width': '80%'}
                ),

            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                html.Label(  
                    'Y-Axis', 
                    style={'font-weight': 'bold', 'text-align': 'right','offset':1}
                ),            
                dcc.Dropdown(
                    numeric_cols,
                    'Ultimate Tensile Strength.a',
                    id='yaxis-column',
                    style={ 'width': '80%', 'margin-bottom': '10px'}
                ),
                html.Label(  
                    'Size', 
                    style={'font-weight': 'bold', 'text-align': 'right','offset':1}
                ),              
                dcc.Dropdown(
                    numeric_cols,
                    'Target Composition (%).Co',
                    id='size-column',
                    style={ 'width': '80%'}
                )
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),     

        ]),

        dcc.Graph(id='indicator-graphic'),
        html.Div(id='error-message')
    ], style={'margin': '20px'})

def show_plot():
    app = Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
               requests_pathname_prefix='/proxy/8050/')

    app.layout = serve_layout
 
    app.run(debug=False, jupyter_mode='jupyterlab', host='0.0.0.0', jupyter_server_url=f"https://{os.environ['TMP_URL']}/")
    return app

def report():

    # Dictionary to store the NaN counts for each query term
    na_counts = {}

    # Loop over each query term
    for term in query_terms:
        # Run the query
        result_df = birdshot.query(f'{term}')

        # Count the number of NaN values for each column
        na_counts[term] = result_df.isna().sum()

    # Convert the dictionary to a DataFrame for better presentation
    na_counts_df = pd.DataFrame(na_counts)
    # Replace NaN values with 16
    na_counts_df_filled = na_counts_df.fillna(16)
    # Show the NaN counts per column for each query term
    print(na_counts_df_filled)

    na_counts_df_filled.to_csv('report.csv', index=True)
    
   
# def main():
#     show_plot()
    
    
# if __name__ == "__main__":
#     main()
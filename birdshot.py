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
import re
import json
from collections import defaultdict

query_terms = ["AAA", "AAB", "AAC", "AAD", "AAE", "BAA", "BBA", "BBB", "BBC", "CBA"]

# ========= Utility Functions =========

def total_molar_mass(composition):
    molar_mass = 0
    for element, value in composition.items():
        molar_mass += value / getattr(globals()[element], "mass")
    return molar_mass

# ========= Query Function =========

def query(campaign, client, raw=False):
    raw_data = client.get(
        "entry", parameters={"query": f"^{campaign}.._VAM-.", "limit": 1000}
    )
    if raw:
        return raw_data

    data = {}
    for entry in raw_data:
        entry = entry["data"]
        sample_id = entry["sampleId"].split("_")
        if len(sample_id) > 2:
            subsample_id = sample_id[-1]
        else:
            subsample_id = None
        sample_id = "_".join(sample_id[:2])

        if sample_id not in data:
            data[sample_id] = {}

        if entry["suffix"] == "Tensile":
            for key in entry["Results"]:
                data[sample_id][f"{key}.{subsample_id}"] = entry["Results"][key]
        elif entry["suffix"] == "Syn" and "Material Preparation" in entry:
            target_mass = entry["Material Preparation"]["Target Mass"]
            target_mass.pop("Total", None)
            sum_molar = total_molar_mass(target_mass)
            data[sample_id].update(
                {
                    f"Target Composition (%).{element}": round(
                        (value / getattr(globals()[element], "mass")) / sum_molar * 100
                    )
                    for element, value in target_mass.items()
                }
            )
        elif entry["suffix"] == "EDS":
            for element, value in entry["results"]["eds"].items():
                data[sample_id][f"Measured Composition (%).{element}"] = value
        elif entry["suffix"] == "XRD":
            try:
                results = entry["XRD Process Overview"]["TAMU Instrument Details"][
                    "Results"
                ]
                phase = results["Phase Information"][0]
                data[sample_id]["XRD.Phase"] = phase["Structure"]
                data[sample_id]["XRD.Lattice Parameters"] = phase["a"]
            except Exception:
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
    # print(df)

    return df

def average_replicate_columns(df):

    measurements_to_average = [
        'Elastic Modulus', 'Elongation', 'Maximum ∂2σ/∂ε2',
        'UTS/YS Ratio', 'Ultimate Tensile Strength', 'Yield Strength'
    ]

    df_new = df.copy()

    for measurement in measurements_to_average:
        # Find columns matching this measurement with .b/.c/.d suffix
        pattern = re.compile(rf'^{re.escape(measurement)}\.[a-zA-Z]$')
        replicate_cols = [col for col in df.columns if pattern.match(col)]

        if replicate_cols:
            df_new[f'{measurement} (Average)'] = df[replicate_cols].mean(axis=1)
            df_new.drop(columns=replicate_cols, inplace=True)
    # print(df)
    # print(df_new)
    # exit()

    return df_new

# ========= Plot Server =========

def return_filtered_df(df):
    numeric_df = df.select_dtypes(include=[int, float])
    cols = sorted(df.columns)
    numeric_cols = sorted(numeric_df)
    numeric_cols_without_nan = numeric_df.columns[numeric_df.notna().all()].tolist()
    return cols, numeric_cols, numeric_df, numeric_cols_without_nan

def serve_layout(client):
    default_campaign = 'BBC'    

    df = query(default_campaign, client)

    if average:
        df = average_replicate_columns(df)

    cols, numeric_cols, numeric_df, numeric_cols_without_nan = return_filtered_df(df)

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
            ], 
            style={'margin-bottom': '10px'
        }),
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

            ], 
            style={'width': '48%', 'display': 'inline-block'
        }),
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
                numeric_cols_without_nan,
                'Target Composition (%).Co',
                id='size-column',
                style={ 'width': '80%'}
            )
            ], 
            style={'width': '48%', 'float': 'right', 'display': 'inline-block'
        }),     
        ]),
        dcc.Graph(id='indicator-graphic'),
        html.Div(id='error-message'),
        html.Div([
            html.Label('Number of points plotted:'),
            html.Div(id='num-points')
        ]),
        html.Div([
            html.Label('Missing data indices:'),
            html.Div(id='missing-indices')
        ]),
        ], 
        style={'margin': '20px'
    })

@callback(
    [Output('indicator-graphic', 'figure'),
     Output('missing-indices', 'children'),
     Output('num-points', 'children'),
     Output('error-message', 'children'),
     Output("xaxis-column", "options"),
     Output("yaxis-column", "options"),
     Output("color-column", "options"),
     Output("size-column", "options"),
     ],
    [Input('campaign-column', 'value'),
     Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('color-column', 'value'),
     Input('size-column', 'value')]
)
def update_graph(campaign, xaxis_column_name, yaxis_column_name,
                color_column_name, size_column_name):    

    try:
        # print(campaign)
        df = query(campaign, client)

        if average:
            df = average_replicate_columns(df)

        cols, numeric_cols, numeric_df, numeric_cols_without_nan = return_filtered_df(df)
        
        # Check if all required columns are in the dataframe
        required_columns = [xaxis_column_name, yaxis_column_name, color_column_name, size_column_name]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if df.empty:
            return go.Figure(), 'N/A', 'N/A', 'Data cannot be found. The graph cannot be generated.', numeric_cols, numeric_cols, cols, numeric_cols_without_nan
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
            return fig, 'N/A', 'N/A', f'The following columns are missing: {", ".join(missing_columns)}', numeric_cols, numeric_cols, cols, numeric_cols_without_nan
        
        
        filtered_df = df[[xaxis_column_name, yaxis_column_name, color_column_name, size_column_name]].copy().dropna()

        # Count the number of points to be plotted
        num_points = len(filtered_df)

        # Get indices of rows where X or Y are missing (without altering df)
        missing_indices = df.index[df[[xaxis_column_name, yaxis_column_name]].isna().any(axis=1)].tolist()
        missing_indices = ', '.join(map(str, missing_indices)) + ','

        fig = px.scatter(df, x=xaxis_column_name,
                        y=yaxis_column_name,
                        size=size_column_name,
                        color=color_column_name,
                        hover_data={  #
                                'sample': df.index   
                            }
                        )
    
        
        return fig, missing_indices, num_points, 'No Error', numeric_cols, numeric_cols, cols, numeric_cols_without_nan

    except Exception as e:
        return go.Figure(), 'N/A', 'N/A', f'An error occurred: {str(e)}', numeric_cols, numeric_cols, cols, numeric_cols_without_nan

# ========= 02: Main Launch Function =========
def show_plot(external_client):
    global client 
    global average
    average = True
    client = external_client
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    )
    app.layout = serve_layout(client)

    # --- Run server ---
    port = 8050
    while True:
        try:
            app.run(debug=False, host="localhost", port=port)
            break
        except OSError as e:
            if "already in use" in str(e):
                port += 1
            else:
                raise

    return app

# ========= 04_Campaign_Status =========

def summarize_presence_by_sample(df: pd.DataFrame, campaign: str, group_prefixes=None) -> pd.DataFrame:
    if group_prefixes is None:
        group_prefixes = [
            'Elastic Modulus', 'Elongation', 'Maximum ∂2σ/∂ε2',
            'UTS/YS Ratio', 'Ultimate Tensile Strength', 'Yield Strength'
        ]

    grouped_columns = defaultdict(list)
    single_columns = []

    for col in df.columns:
        for prefix in group_prefixes:
            pattern = re.compile(rf'^{re.escape(prefix)}\.([a-zA-Z])$')
            match = pattern.match(col)
            if match:
                suffix = match.group(1)
                grouped_columns[prefix].append((col, suffix))
                break
        else:
            single_columns.append(col)

    summary_rows = []

    for idx, row in df.iterrows():
        sample_id = row.name if df.index.name else idx
        entry = {"Sample": sample_id}

        # Grouped columns: summarize which subsamples exist
        for prefix, cols in grouped_columns.items():
            present_suffixes = [suffix for col, suffix in cols if pd.notna(row[col])]
            entry[prefix] = ", ".join(sorted(present_suffixes)) if present_suffixes else "None"

        # Non-grouped columns
        for col in single_columns:
            entry[col] = "Yes" if pd.notna(row[col]) else "No"

        summary_rows.append(entry)

    # Add summary row per campaign
    summary = {"Sample": campaign}
    total_rows = df.shape[0]

    for prefix, cols in grouped_columns.items():
        total_possible = total_rows * len(cols)
        total_present = sum(pd.notna(df[col]).sum() for col, _ in cols)
        summary[prefix] = f"{total_present}/{total_possible}"

    for col in single_columns:
        total_present = pd.notna(df[col]).sum()
        summary[col] = f"{total_present}/{total_rows}"

    summary_rows.append(summary)

    return pd.DataFrame(summary_rows)
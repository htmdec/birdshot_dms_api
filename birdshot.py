from functools import lru_cache
from girder_client import GirderClient
import pandas as pd
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import os
from periodictable import Al, V, Cr, Mn, Fe, Co, Ni, Cu


def total_molar_mass(composition):
    molar_mass = 0
    for element, value in composition.items():
        molar_mass += value / getattr(globals()[element], "mass")
    return molar_mass


@lru_cache
def query(campaign, raw=False):
    client = GirderClient(apiUrl=os.environ["GIRDER_API_URL"])
    client.authenticate(apiKey=os.environ["GIRDER_API_KEY"])

    raw_data = client.get(
        "entry/search", parameters={"query": f"^{campaign}.._VAM-.", "limit": 1000}
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
            for element, value in entry["Results"]["Measured Composition (%)"].items():
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

    return pd.DataFrame.from_dict(data, orient="index")


def serve_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Campaign", style={"font-weight": "bold"}),
                    dcc.Dropdown(
                        [
                            "AAA",
                            "AAB",
                            "AAC",
                            "AAD",
                            "AAE",
                            "BAA",
                            "BBA",
                            "BBB",
                            "BZZ",
                            "CBA",
                            "ZZZ",
                        ],
                        "AAA",
                        id="campaign-column",
                        style={"width": "150px"},
                    ),
                ],
                style={"margin-bottom": "10px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("X-Axis", style={"font-weight": "bold"}),
                            dcc.Dropdown(id="xaxis-column"),
                            html.Label("Color", style={"font-weight": "bold"}),
                            dcc.Dropdown(id="color-column"),
                        ],
                        style={"width": "48%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            html.Label("Y-Axis", style={"font-weight": "bold"}),
                            dcc.Dropdown(id="yaxis-column"),
                            html.Label("Size", style={"font-weight": "bold"}),
                            dcc.Dropdown(id="size-column"),
                        ],
                        style={
                            "width": "48%",
                            "float": "right",
                            "display": "inline-block",
                        },
                    ),
                ]
            ),
            dcc.Graph(id="indicator-graphic"),
        ],
        style={"margin": "20px"},
    )


app = Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]
)
app.layout = serve_layout


@app.callback(
    Output("xaxis-column", "options"),
    Output("yaxis-column", "options"),
    Output("color-column", "options"),
    Output("size-column", "options"),
    Output("xaxis-column", "value"),
    Output("yaxis-column", "value"),
    Output("color-column", "value"),
    Output("size-column", "value"),
    Input("campaign-column", "value"),
)
def update_dropdown_columns(campaign):
    df = query(campaign)
    cols = sorted(df.columns)
    defaults = [cols[i] if len(cols) > i else cols[0] for i in range(4)]
    return cols, cols, cols, cols, *defaults


@app.callback(
    Output("indicator-graphic", "figure"),
    Input("campaign-column", "value"),
    Input("xaxis-column", "value"),
    Input("yaxis-column", "value"),
    Input("color-column", "value"),
    Input("size-column", "value"),
)
def update_graph(campaign, x_col, y_col, color_col, size_col):
    df = query(campaign)
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col)
    return fig


def show_plot():
    port = 8050
    while True:
        try:
            app.run(debug=False, host="localhost", port=port, use_reloader=False)
            break
        except OSError as e:
            if "already in use" in str(e):
                port += 1
            else:
                raise

    return app

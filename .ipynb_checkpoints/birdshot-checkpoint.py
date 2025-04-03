from functools import lru_cache
from girder_client import GirderClient
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import os
from periodictable import Al, V, Cr, Mn, Fe, Co, Ni, Cu


def total_molar_mass(composition):
    molar_mass = 0
    for element, value in composition.items():
        molar_mass += value / getattr(globals()[element], "mass")
    return molar_mass


def percent_composition(composition, total):
    return {element: value / total for element, value in composition.items()}


@lru_cache
def query(campaign, raw=False):
    client = GirderClient(apiUrl=os.environ["GIRDER_API_URL"])
    client.token = os.environ["GIRDER_TOKEN"]
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
            total_mass = round(float(target_mass.pop("Total")))
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

    df = pd.DataFrame.from_dict(data, orient="index")

    # Safe type casting only if the columns exist
    for col in ["UTS/YS Ratio.a", "UTS/YS Ratio.b"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


@callback(
    Output("indicator-graphic", "figure"),
    Input("campaign-column", "value"),
    Input("xaxis-column", "value"),
    Input("yaxis-column", "value"),
    Input("color-column", "value"),
    Input("size-column", "value"),
)
def update_graph(
    campaign, xaxis_column_name, yaxis_column_name, color_column_name, size_column_name
):
    df = query(campaign)
    fig = px.scatter(
        df,
        x=xaxis_column_name,
        y=yaxis_column_name,
        size=size_column_name,
        color=color_column_name,
    )
    return fig


def serve_layout():
    default_campaign = "AAA"
    df = query(default_campaign)
    cols = sorted(df.columns)

    return html.Div(
        [
            html.Div(
                [
                    html.Label(
                        "Campaign", style={"font-weight": "bold", "text-align": "right"}
                    ),
                    dcc.Dropdown(
                        ["AAA", "AAB", "AAC", "AAD", "AAE"],
                        default_campaign,
                        id="campaign-column",
                        style={"width": "100px"},
                    ),
                ],
                style={"margin-bottom": "10px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "X-Axis",
                                style={"font-weight": "bold", "text-align": "right"},
                            ),
                            dcc.Dropdown(
                                cols,
                                "Yield Strength.a",
                                id="xaxis-column",
                                style={"width": "80%", "margin-bottom": "10px"},
                            ),
                            html.Label(
                                "Color",
                                style={"font-weight": "bold", "text-align": "right"},
                            ),
                            dcc.Dropdown(
                                cols,
                                "Target Composition (%).Fe",
                                id="color-column",
                                style={"width": "80%"},
                            ),
                        ],
                        style={"width": "48%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Y-Axis",
                                style={"font-weight": "bold", "text-align": "right"},
                            ),
                            dcc.Dropdown(
                                cols,
                                "Ultimate Tensile Strength.a",
                                id="yaxis-column",
                                style={"width": "80%", "margin-bottom": "10px"},
                            ),
                            html.Label(
                                "Size",
                                style={"font-weight": "bold", "text-align": "right"},
                            ),
                            dcc.Dropdown(
                                cols,
                                "Target Composition (%).Co",
                                id="size-column",
                                style={"width": "80%"},
                            ),
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


def show_plot():
    from dash import Dash
    import dash_bootstrap_components as dbc

    port = 8050
    while True:
        try:
            app = Dash(
                __name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                requests_pathname_prefix=f"/proxy/{port}/",
            )
            app.layout = serve_layout
            app.run(
                debug=False,
                jupyter_mode="external",  # This avoids jupyter proxy complications
                host="0.0.0.0",
                port=port,
                jupyter_server_url=f"http://{os.environ.get('TMP_URL', 'localhost:8888')}/",
            )
            break
        except OSError as e:
            if "Address" in str(e) and "already in use" in str(e):
                port += 1  # Try next port
            else:
                raise

    return app

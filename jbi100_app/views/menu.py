# jb100_app/views/menu.py
from dash import html, dcc
from .. import config


def layout():
    """Top-level UI: title + global filters + layer switch + global store."""
    return html.Div(
        children=[
            # ---- Title ----
            html.H1(
                "Hospital Beds Management Dashboard",
                className="page-title"
            ),

            # ---- Global Filters ----
            html.Div(
                children=[
                    html.Div(
                        [
                            html.Label("Week range"),
                            dcc.RangeSlider(
                                id="week-range",
                                min=1,
                                max=52,
                                step=1,
                                value=[1, 52],
                                marks={1: "1", 13:"13", 26: "26", 39: "39", 52: "52"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        className="filter-item",
                    ),

                    html.Div(
                        [
                            html.Label("Service"),
                            dcc.Dropdown(
                                id="service-filter",
                                multi=True,
                                options=[{"label": s, "value": s} for s in config.SERVICES_ORDER],
                                value=list(config.SERVICES_ORDER),  # 默认全选
                                placeholder="Select service(s)",
                                clearable=False,
                            ),
                        ],
                        className="filter-item",
                    ),

                    html.Div(
                        [
                            html.Label("Event"),
                            dcc.Dropdown(
                                id="event-filter",
                                multi=True,
                                options=[{"label": e, "value": e} for e in config.EVENT_ORDER],
                                value=list(config.EVENT_ORDER),  # 默认全选（包含 none）
                                placeholder="Select event(s)",
                                clearable=False,
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-row",
            ),

            # ---- Layer Switch ----
            html.Div(
                children=[
                    dcc.RadioItems(
                        id="layer-selector",
                        options=[
                            {"label": "Overview", "value": "overview"},
                            {"label": "Incidents", "value": "incidents"},
                            {"label": "Impact", "value": "impact"},
                            {"label": "Diagnosis", "value": "diagnosis"},
                        ],
                        value="overview",
                        inline=True,
                        className="layer-selector",
                    )
                ],
                className="layer-row",
            ),

            # ---- Global Store ----
            dcc.Store(
                id="global-store",
                data={
                    "filters": {
                        "weekRange": [1, 52],
                        "services": list(config.SERVICES_ORDER),
                        "events": list(config.EVENT_ORDER),
                        "shortageThreshold": getattr(config, "DEFAULT_SHORTAGE_THRESHOLD", 0.2),
                    },
                    "selection": {"service": None, "week": None},
                    "layer": "overview",
                },
            ),
        ],
        className="page-shell",
    )
from dash import Input, Output, callback

@callback(
    Output("event-filter", "disabled"),
    Output("event-filter", "style"),
    Input("layer-selector", "value")
)
def toggle_event_filter_by_layer(selected_layer):
    """
    Disables the Event filter interface when the Diagnosis layer is active.
    """
    if selected_layer == "diagnosis":
        # Disable the dropdown and dim it visually
        return True, {"opacity": "0.5", "pointerEvents": "none"}
    
    # Re-enable for Overview, Incidents, or Impact layers
    return False, {"opacity": "1", "pointerEvents": "auto"}
# jbi100_app/main.py
from dash import Dash, html, dcc
from dash import Input, Output, State, callback

from .views import menu
from .views import overview
from .views import incidents 
from .views import impact
from .views import diagnosis

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Hospital Operations Analytics Dashboard"

app.layout = html.Div(
    className="app-container",
    children=[
        # Shared global state for cross-view synchronization
        dcc.Store(id='shared-selected-week', data=1),

        # ===== Sidebar =====
        html.Div(
            id="sidebar",
            className="sidebar expanded",  
            children=[
                html.Div(
                    className="sidebar-toggle-row",
                    children=[
                        html.Button("◀", id="toggle-sidebar", n_clicks=0, className="toggle-btn"),
                    ],
                ),
                menu.layout(),
            ],
        ),

        # ===== Main content =====
        html.Div(
            id="main-content",
            className="main-content expanded",
            children=[
                html.Div(
                    id="layer-container",
                    className="layer-container",
                    children=overview.layout(),  # default view
                ),
            ],
        ),
    ],
)


# --- Sidebar toggle logic ---
@callback(
    Output("sidebar", "className"),
    Output("main-content", "className"),
    Output("toggle-sidebar", "children"),
    Input("toggle-sidebar", "n_clicks"),
    State("sidebar", "className"),
)
def toggle_sidebar(n_clicks, sidebar_class):
    if not sidebar_class:
        sidebar_class = "sidebar expanded"
    if n_clicks and n_clicks % 2 == 1:
        return "sidebar collapsed", "main-content collapsed", "▶"
    return "sidebar expanded", "main-content expanded", "◀"


# --- Layer switch logic (render selected view) ---
@callback(
    Output("layer-container", "children"),
    Input("layer-selector", "value"),
)
def switch_layer(layer_value: str):
    # Defensive default
    layer_value = layer_value or "overview"

    if layer_value == "incidents":
        return incidents.layout()
    
    if layer_value == "impact":
        return impact.layout()

    if layer_value == "diagnosis":
        return diagnosis.layout()

    return overview.layout()

if __name__ == "__main__":
    app.run_server(debug=True)
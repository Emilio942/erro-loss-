import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Angenommen, 'results' ist ein Dictionary, das deine Daten enthält
results = {
    'optimizer1': {'train_loss': [0.6, 0.4, 0.3], 'val_loss': [0.5, 0.45, 0.35]},
    'optimizer2': {'train_loss': [0.65, 0.5, 0.4], 'val_loss': [0.55, 0.5, 0.45]}
}

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='optimizer-selector',
        options=[{'label': k, 'value': k} for k in results.keys()],
        value='optimizer1'
    ),
    dcc.Graph(id='loss-graph')
])

@app.callback(
    Output('loss-graph', 'figure'),
    [Input('optimizer-selector', 'value')]
)
def update_graph(selected_optimizer):
    data = results[selected_optimizer]
    trace1 = go.Scatter(y=data['train_loss'], mode='lines+markers', name='Train Loss')
    trace2 = go.Scatter(y=data['val_loss'], mode='lines+markers', name='Validation Loss')
    return {'data': [trace1, trace2], 'layout': go.Layout(title=f'Loss Curves for {selected_optimizer}')}

if __name__ == '__main__':
    app.run_server(debug=True)


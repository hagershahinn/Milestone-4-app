import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import io
import base64
import numpy as np

app = dash.Dash(__name__)
app.title = "XGBoost Regression App"

data = None
encoded_data = None
model = None
categorical_columns = []
numerical_columns = []
trained_features = None  

app.layout = html.Div([
    html.H1("XGBoost Regression App", style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or Click to Upload File']),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='dataset-instructions', style={'margin': '20px', 'color': 'blue'}),
    html.Div([
        html.Label("Choose the Target Variable:"),
        dcc.Dropdown(id='target-variable-dropdown', style={'marginBottom': '20px'}),
    ]),
    dcc.Graph(id='correlation-graph', style={'height': '400px'}),
    html.Div([
        html.Label("Select a Categorical Variable for Analysis:"),
        dcc.RadioItems(id='categorical-variable-radio', inline=True),
        dcc.Graph(id='categorical-analysis-graph', style={'height': '400px'}),
    ], style={'margin': '20px'}),
    html.Div([
        html.Label("Pick Features for Training:"),
        dcc.Checklist(id='feature-selection-checklist', inline=True),
        html.Button("Train the Model", id='train-button', n_clicks=0),
        html.Div(id='model-results', style={'marginTop': '20px'}),
    ], style={'margin': '20px'}),
    html.Div([
        html.Label("Enter Feature Values for Prediction:"),
        dcc.Input(id='prediction-input', type='text', style={'width': '50%', 'marginBottom': '10px'}),
        html.Button("Predict!", id='predict-button', n_clicks=0),
        html.Div(id='prediction-result', style={'marginTop': '20px'}),
    ], style={'margin': '20px'}),
])

@app.callback(
    [Output('target-variable-dropdown', 'options'),
     Output('target-variable-dropdown', 'value'),
     Output('feature-selection-checklist', 'options'),
     Output('categorical-variable-radio', 'options'),
     Output('dataset-instructions', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_file(contents, filename):
    global data, encoded_data, categorical_columns, numerical_columns, trained_features
    if contents is None:
        return [], None, [], [], "Please upload a CSV file to get started."

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        encoded_data = data.copy()
        categorical_columns = []
        numerical_columns = []
        trained_features = None  

        for col in data.columns:
            if data[col].dtype == 'object' or (data[col].nunique() < 10 and data[col].dtype in ['int64', 'float64']):
                categorical_columns.append(col)
                encoded_data[col] = encoded_data[col].astype('category').cat.codes
            else:
                numerical_columns.append(col)
                encoded_data[col] = data[col]

        options = [{'label': col, 'value': col} for col in data.columns]
        categorical_options = [{'label': col, 'value': col} for col in categorical_columns]
        instructions = [
            html.Div(f"Dataset Loaded: {filename}", style={'fontWeight': 'bold'}),
            html.Div(f"Detected {len(categorical_columns)} categorical variables: {', '.join(categorical_columns)}"),
            html.Div(f"Detected {len(numerical_columns)} numerical variables."),
            html.Div("You can select a target variable, explore categorical relationships, and train a model."),
            html.Div("For predictions, ensure the input matches the trained dataset."),
        ]
        return options, None, options, categorical_options, instructions
    except Exception as e:
        return [], None, [], [], f"Error loading file: {e}"

@app.callback(
    Output('correlation-graph', 'figure'),
    Input('target-variable-dropdown', 'value')
)
def update_correlation_graph(target):
    if encoded_data is None or target is None:
        return {}
    try:
        correlation = encoded_data[numerical_columns].corr()[target].sort_values(ascending=False)
        fig = px.bar(
            x=correlation.index, 
            y=correlation.values, 
            labels={'x': 'Features', 'y': 'Correlation'},
            title=f"Correlation of Features with '{target}'",
            color_discrete_sequence=['black']
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        return fig
    except Exception as e:
        return {}

@app.callback(
    Output('categorical-analysis-graph', 'figure'),
    [Input('target-variable-dropdown', 'value'),
     Input('categorical-variable-radio', 'value')]
)
def update_categorical_analysis(target, categorical_var):
    if data is None or target is None or categorical_var is None:
        return {}
    try:
        avg_values = data.groupby(categorical_var)[target].mean().reset_index()
        fig = px.bar(
            avg_values, 
            x=categorical_var, 
            y=target,
            title=f"Average '{target}' by '{categorical_var}'", 
            labels={categorical_var: "Category", target: "Average Value"},
            color_discrete_sequence=['black']
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        return fig
    except Exception as e:
        return {}

@app.callback(
    Output('model-results', 'children'),
    Input('train-button', 'n_clicks'),
    [State('target-variable-dropdown', 'value'),
     State('feature-selection-checklist', 'value')]
)
def train_model(n_clicks, target, selected_features):
    global model, trained_features
    if n_clicks == 0 or encoded_data is None or target is None or not selected_features:
        return "Please select the target variable and features to train the model."

    trained_features = selected_features  # Save trained features
    X = encoded_data[selected_features]
    y = encoded_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return [
        html.Div("The model has been trained successfully!"),
        html.Div(f"R² Score: {r2:.4f}"),
        html.Div(f"RMSE: {rmse:.4f}")
    ]

@app.callback(
    Output('prediction-result', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('prediction-input', 'value'),
     State('feature-selection-checklist', 'value')]
)
def predict(n_clicks, input_values, selected_features):
    global trained_features
    if n_clicks == 0 or model is None or not input_values or not selected_features:
        return "Please ensure the model is trained and input values are provided."

    if trained_features != selected_features:
        return "Error: The selected features do not match the features used for training."

    try:
        input_values = list(map(float, input_values.split(',')))
        if len(input_values) != len(selected_features):
            return "The number of input values does not match the selected features."

        input_df = pd.DataFrame([input_values], columns=selected_features)
        prediction = model.predict(input_df)[0]
        return f"The predicted value is: {prediction:.4f}"
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run_server(debug=True)
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop

import pandas as pd
import numpy as np

import plotly.graph_objects as go

app = dash.Dash(__name__)
app.scripts.config.serve_locally = True

from keras.models import load_model
from keras.optimizers import RMSprop
import random

model = load_model('LSTM.hdf5',compile=False)
model_RNN = load_model('RNN-GRU.hdf5',compile=False)

lookback = 3*24*7 # no. of hours to use to forecast next hour
step = 1 # 1 for hourly, 24 for daily, etc.
delay = 1 # no. of hour(s) to forecast
batch_size = 64 # no of records per epoch, each record contains "lookback" hours of input and "delay" hours of output, 
### 2 ^ n batch_size. i.e.2,4,8,16

# Compile model (required to make predictions)
model.compile(optimizer=RMSprop(), loss='mae')

import os
fname = 'Hourly Demand_201301-201906_ALL_PH_MOD.csv'
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:len(lines)-1]

selected_columns = ['MONTH', 'WEEKDAY', ' HR ', 'PH']

data2 = pd.read_csv('Hourly Demand_201301-201906_ALL_PH_MOD.csv',usecols=selected_columns)
dummy=data2

import os
fname = 'Hourly Demand_201301-201906_ALL_PH_MOD.csv'
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:len(lines)-1]

float_data = np.zeros((len(lines), len(header)-1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

NN_data=float_data

train_count = int(len(NN_data)*.7)
validate_count = int((len(NN_data)-train_count)*0.5)
test_count = len(NN_data)-train_count-validate_count

mean = NN_data[:train_count].mean(axis=0)
NN_data -= mean
std = NN_data[:train_count].std(axis=0)
NN_data /= std

target_column = len(dummy.columns)-1

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][target_column]
        yield samples, targets

train_gen = generator(NN_data,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=train_count,
    shuffle=True,
    step=step,
    batch_size=batch_size)

val_gen = generator(NN_data,
    lookback=lookback,
    delay=delay,
    min_index=train_count+1,
    max_index=train_count+validate_count,
    step=step,
    batch_size=batch_size)

test_gen = generator(NN_data,
    lookback=lookback,
    delay=delay,
    min_index=train_count+validate_count+1,
    max_index=None,
    step=step,
    batch_size=batch_size)

val_steps = (validate_count - 1 - lookback)  #How many steps to draw from val_gen in order to see the entire 
                                        #validation set --- 
                                        #This is normally a problem in keras so let's manually set this to just 1000
test_steps = (test_count - 1 - lookback)  #How many steps to draw from test_gen in order to see the 
                                        #entire test set 

########

hours_needed = 60
traing = []
for i in range(hours_needed):
    traing.append(next(train_gen))

vg = []
for i in range(hours_needed):
    vg.append(next(val_gen))
    
testg = []
for i in range(hours_needed):
    testg.append(next(test_gen))

batch_test = random.randint(0,50)
    
# pred1 = model.predict(vg[0][0]) #prediction for first batch containing batch_size 64
# pred_RNN = model_RNN.predict(vg[0][0]) #prediction for first batch containing batch_size 64
pred1 = model.predict(vg[batch_test][0]) #prediction for first batch containing batch_size 64
pred_RNN = model_RNN.predict(vg[batch_test][0]) #prediction for first batch containing batch_size 64


# actual_output=vg[0][1][0:hours_needed]*std[3]+mean[3]
actual_output=vg[batch_test][1][0:hours_needed]*std[3]+mean[3]

predicted_output=pred1[0:hours_needed]*std[3]+mean[3]
predicted_output_RNN=pred_RNN[0:hours_needed]*std[3]+mean[3]

data_actual=[]

data_predict=[]
data_predict_RNN=[]

data_results=[]
data_results_RNN=[]

for i in range(0,hours_needed):
    data_actual.append(actual_output[i])
    
    data_predict.append(predicted_output[i])
    data_predict_RNN.append(predicted_output_RNN[i])
    
    results=(predicted_output[i][0]-actual_output[i])
    results=(predicted_output_RNN[i][0]-actual_output[i])
    
    data_results.append([actual_output[i],
                        predicted_output[i][0],
                        results,
                        100*results/actual_output[i],
                        abs(100*results/actual_output[i],)])
    
    data_results_RNN.append([actual_output[i],
                        predicted_output_RNN[i][0],
                        results,
                        100*results/actual_output[i],
                        abs(100*results/actual_output[i],)])
    
#print(results,actual_output[i],100*results/actual_output[i])

#LSTM Results
data_results_df = pd.DataFrame(data=data_results,columns=['Actual MW', 'Predicted MW',
                                                        'Error in MW','Error in %',
                                                        'Absolute Error in %'])
dfx = pd.DataFrame(data_results_df['Actual MW'])
dfx['Label']='Actual_MW'
dfx = dfx.reset_index()
dfx.columns=['Hours','Megawatts','Label']
dfx2 = pd.DataFrame(data_results_df['Predicted MW'])
dfx2['Label']='Predicted_MW'
dfx2 = dfx2.reset_index()
dfx2.columns=['Hours','Megawatts','Label']
dfz = pd.concat([dfx,dfx2])
dfz.to_csv('dfz_LSTM.csv',index=False)

#RNN Results
data_results_df_RNN = pd.DataFrame(data=data_results_RNN,columns=['Actual MW', 'Predicted MW',
                                                        'Error in MW','Error in %',
                                                        'Absolute Error in %'])
dfx = pd.DataFrame(data_results_df_RNN['Actual MW'])
dfx['Label']='Actual_MW'
dfx = dfx.reset_index()
dfx.columns=['Hours','Megawatts','Label']
dfx2 = pd.DataFrame(data_results_df_RNN['Predicted MW'])
dfx2['Label']='Predicted_MW'
dfx2 = dfx2.reset_index()
dfx2.columns=['Hours','Megawatts','Label']
dfz = pd.concat([dfx,dfx2])
dfz.to_csv('dfz_RNN.csv',index=False)

#####


app.layout = html.Div([
  #  html.Div(children='   ',style={'backgroundColor':'black','background-image': 'logo.'}),
    html.Center(html.H1(children='PowerPH: Predicting the Country\'s Electricity Demand using Neural Networks',style={'color': 'yellow'})),
    html.Div(html.Center(html.H2(children='Machine Learning 2.0 Project',style={'color':'yellow'}))),
    html.Div(html.Center(html.H3(children='Learning Team 2',style={'color':'yellow'}))),
    html.Div(html.Center(html.H4(children='MSDS 2020',style={'color':'yellow'}))),

    html.Div(html.H3(children='Abstract',style={'color':'white'})),
    html.Div(html.H4(children='Predicting energy demand is crucial for power supply planning and policymaking. In this study, we used Neural Network Models to perform energy demand prediction to supplement traditional forecasting models, such as ARIMA and SARIMA, which have been used for decades. Using 6.5 years of hourly demand data from the National Grid Corporation of the Philippines (NGCP), we explored two different neural network models: Long Short-Term Memory (LSTM) and RNN-Gated Recurrent Unit (RNN-GRU). Each model is tested on three different lookbacks: 3 days, 3 weeks, and 3 months. Results show that LSTM with a lookback of 3 weeks achieves the highest prediction performance with a mean absolute error (MAE) of 0.0062. This means that given a mean hourly demand of 8,911.76 MW, the model can give a forecast that has an average accuracy that is within 55.34 MW.',style={'color':'white'})),


    html.Div(html.P([html.Br(style={'color': 'yellow'})])),
    html.Div(html.H3(children='View Prediction Below:',style={'color':'white'})),
    
    html.Div(html.H4(children='Choose the Neural Network Model',style={'color':'white'})),
    dcc.Dropdown(
        id='nn_model',
        options=[
            {'label': 'LSTM', 'value': 'dfz_LSTM.csv'},
            {'label': 'RNN-GRU', 'value': 'dfz_RNN.csv'}
        ],
        value='dfz_LSTM.csv',
        clearable=False
        ),
    
    html.Div(html.H4(children='Choose the number of hours to predict the nationwide electricity demand',style={'color':'white'})),
    dcc.Dropdown(
        id='Hours',
        options=[
            {'label': 'Next 12 Hours', 'value': 12},
            {'label': 'Next 24 Hours', 'value': 24},
            {'label': 'Next 36 Hours', 'value': 36},
            {'label': 'Next 48 Hours', 'value': 48},
            {'label': 'Next 60 Hours', 'value': 60}
        ],
        value=12,
        clearable=False
        ),
    html.Br(),
    html.Button(children='Refresh Power Demand Graph', style={'color':'white'}, id='btn'),
    html.Div(html.P([html.Br(style={'color': 'yellow'})])),

    dcc.Graph(id='Plot'),


    html.Div(html.H3(children='Conclusion and Recommendations',style={'color':'white'})),
    html.Div(html.H4(children='The study focused on applying neural network models to predict power demand using hourly demand data of the Philippines. The two neural networks explored, combined with three different lookback values, outperformed the baseline neural network models. LSTM with a 3-week lookback showed the highest forecast performance in terms of mean absolute error (MAE).',style={'color':'white'})),
    html.Div(html.H4(children='These models can be trained on more granular data (e.g. per franchise area or station) and is expected to perform at a commensurate accuracy level.',style={'color':'white'})),
    html.Div(html.H4(children='High forecast accuracy on power demand translates to huge business value for utility and power generation companies alike. These forecasts can augment in decision support tools for purchasing electricity and power supply planning, both in the short and the long term. These can also guide decisions on urban planning and policymaking in the energy and adjacent sectors.',style={'color':'white'})),
    html.Div(html.H4(children='The choice of hyperparameters run were limited by the amount of computing resources and corresponding running time. Extensions of this study can explore more lookback values and other hyperparameters to find the optimal architecture. While LSTM is currently seen as the best forecasting model for time series data, other neural network models can still be experimented.',style={'color':'white'})),


#    html.Center(html.H4(id='Answer')),
#    html.Center(html.H4(id='Aver'))
    ],
      style={
        'borderBottom': 'black',
        'backgroundColor': 'black',
        'padding': '30px',
        'background-image': 'url(https://docs.google.com/uc?export=view&id=1RiOT1LeSKAAICa0gvlCvIDxhJAgUgA6l)'}
    ) 

@app.callback(
    dash.dependencies.Output('Plot', 'figure'),
    [dash.dependencies.Input('btn','n_clicks')],
    [dash.dependencies.State('Hours','value'),
    dash.dependencies.State('nn_model','value')]
    )
def plotting(btn,Hours,nn_model):
#     dfzz = pd.read_csv('dfz.csv')
    dfzz = pd.read_csv(nn_model)
    dfzza = dfzz.head(Hours)
    dfzzb = dfzz.loc[60:59+Hours]

    dfabc=pd.concat([dfzza,dfzzb])

    #import plotly.express as px
    #fig = px.line(dfabc, x='Hours', y='Megawatts', color='Label')
    #fig.show()
    x = dfzza['Hours']
    y= dfzza['Megawatts']
    yhat= dfzzb['Megawatts']
    x_rev = x[::-1]

    # Line - Actual Values
    y1 = y
    y1_upper = y1*1.001
    y1_lower = y1*0.999
    y1_lower = y1_lower[::-1]


    # Line - Predicted Value
    y2 = yhat
    y2_upper = y2*1.01
    y2_lower = y2*0.99
    y2_lower = y2_lower[::-1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x.tolist()+x_rev.tolist(),
        y=y1_upper.tolist()+y1_lower.tolist(),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Actual',
    ))

    fig.add_trace(go.Scatter(
        x=x.tolist()+x_rev.tolist(),
        y=y2_upper.tolist()+y2_lower.tolist(),
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Actual',
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y1,
        line_color='rgb(0,100,80)',
        name='Actual',
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y2,
        line_color='rgb(231,107,243)',
        name='Predicted',
    ))

    fig.update_traces(mode='lines')
    return fig


application = app.server
if __name__ == '__main__':
    app.run_server()
# PowerPH : Predicting the Power Demand for the Philippines
## Abstract
Predicting energy demand is crucial for power supply planning and policymaking. In this study, we used Neural Network Models to perform energy demand prediction to supplement traditional forecasting models, such as ARIMA and SARIMA, which have been used for decades. Using 6.5 years of hourly demand data from the National Grid Corporation of the Philippines (NGCP), we explored two different neural network models: Long Short-Term Memory (LSTM) and RNN-Gated Recurrent Unit (RNN-GRU). Each model is tested on three different lookbacks: 3 days, 3 weeks, and 3 months. Results show that LSTM with a lookback of 3 weeks achieves the highest prediction performance with a mean absolute error (MAE) of 0.0062. This means that given a mean hourly demand of 8,911.76 MW, the model can give a forecast that has an average accuracy that is within 55.34 MW.

## Recommendations
The study focused on applying neural network models to predict power demand using hourly demand data of the Philippines. The two neural networks explored, combined with three different lookback values, outperformed the baseline neural network models. LSTM with a 3-week lookback showed the highest forecast performance in terms of mean absolute error (MAE).  

<br>

These models can be trained on more granular data (e.g. per franchise area or station) and is expected to perform at a commensurate accuracy level.  

<br>

High forecast accuracy on power demand translates to huge business value for utility and power generation companies alike. These forecasts can augment in decision support tools for purchasing electricity and power supply planning, both in the short and the long term. These can also guide decisions on urban planning and policymaking in the energy and adjacent sectors.  

<br>

The choice of hyperparameters run were limited by the amount of computing resources and corresponding running time. Extensions of this study can explore more lookback values and other hyperparameters to find the optimal architecture. While LSTM is currently seen as the best forecasting model for time series data, other neural network models can still be experimented.  




### To create the environment and run the application, start an anaconda prompt in the cloned directory and run the following:

`conda env create --file environment.yml`  
`conda activate power-ph`  
`python application.py`  


#### Open this link in your browser: http://127.0.0.1:8082/

<br>

#### A deployed application is also available in the link below until January 31, 2020:  
#### http://deployment-dev.ap-northeast-1.elasticbeanstalk.com/

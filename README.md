# project_indianbanks_price_forecast
This project demonstrates that we programmatically download the bank stocks data and also forecast for a defined period of time and also display results on a streamlit UI

Steps to produce results:
1. run data_getter.ipynb and wait till it downloads the data for various banks. Feel free to modify list of banks & from date and to date accordingly. Be mindful of the date when the bank stock went live. From date earlier than the stock live date could result in errors.
2. Run forecast trainer script to train mlforecast models. Feel free to change the list of banks and from date as you did in data getter notebook. Also playaround the various parameters and hyper parameters available.
3. Documentation for MLForecast framework is available here: https://nixtlaverse.nixtla.io/mlforecast/index.html
4. Finally run the forecasting ui script using the command "streamlit run forecasting_ui.py". A localhost webserver will display results on your browser.

As next steps, try replicating this process with different forecasting models like prophet, LSTM, pandarima forecasting models etc

![forecasting_1](https://github.com/sadiqgpasha/project_indianbanks_price_forecast/blob/main/screenshot1.png)
![forecasting_2](https://github.com/sadiqgpasha/project_indianbanks_price_forecast/blob/main/screenshot2.png)

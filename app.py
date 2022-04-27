
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pylab as matpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import warnings
warnings.simplefilter("ignore")


matpy.rcParams['figure.figsize'] = (11, 8)


st.title("Future Framework Forcasting App")


# Quick way to do addfuller test #
def quick_adfuller(data):
    # as a default we say the time series is stationary
    stationarity_value = True
    dftest = adfuller(data, autolag='AIC')

    st.write("1. ADF : ", dftest[0])
    st.write("2. P-Value : ", dftest[1])
    st.write("3. Num Of Lags : ", dftest[2])
    st.write(
        "4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    st.write("5. Critical Values :")
    for key, val in dftest[4].items():
        st.write("\t", key, ": ", val)

    if dftest[1] <= 0.05:
        st.subheader("**Time series is stationary**")
        stationarity_value = True
    else:
        st.subheader("Time series is not stationary")
        stationarity_value = False

    return stationarity_value


def acf_plot(data):
    # acf used for MA ie acf=q
    fig = plot_acf(data.values)
    plt.grid(True)
    plt.xticks(np.arange(17))
    st.pyplot(fig)


def pacf_plot(data):
    # pacf used for AR ie pacf=p
    fig = plot_pacf(data.values)
    plt.grid(True)
    plt.xticks(np.arange(17))
    st.pyplot(fig)

# function not needed #


def arima_plot(data):
    #fig1 = plt.plot(data)
    fitted_df = data.fittedvalues
    #fitted_df.columns = ["fittedvalues"]
    fig2 = plt.plot(fitted_df, color='red')
    st.pyplot(fig2)

    # st.write(fitted_df)

    #fig2 = plt.plot(data.fittedvalues, color='red')
    # st.pyplot(fig1)
    # st.pyplot(fig2)


def plot_arima_focast(train, test, forecast_series):
    fig = plt.figure(figsize=(12, 5), dpi=100)

    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(forecast_series, label='forecast')
    # plt.fill_between(lower_series.index, lower_series, upper_series,
    #                  color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid()
    st.pyplot(fig)


def fbProphet_plot(model_obj, forecast_df):
    fig = model_obj.plot(forecast_df)
    st.pyplot(fig)

# shows tred and seasonality #


def fbProphet_plot_components(model_obj, forecast_df):
    fig = model_obj.plot_components(forecast_df)
    st.pyplot(fig)

# plot auto arima residuals #


def auto_ARIMA_risiduals(model_fitted):
    fig = model_fitted.plot_diagnostics(figsize=(12, 8))
    st.pyplot(fig)

# Plot residual errors ARIMA #


def plot_ARIMA_residuals(res):
    fig, ax = plt.subplots(1, 2, figsize=(11, 1.5))
    res.plot(title="Residuals", ax=ax[0])
    res.plot(kind="kde", title="Density", ax=ax[1])
    st.pyplot(fig)


# Loading Data #
df = pd.read_csv("FrameworkData.csv", index_col=["month"])
df.index = pd.to_datetime(df.index, yearfirst=True)


# select framework #
add_selectbox_framework = st.sidebar.selectbox(
    'Please select an Framework',
    ('nltk', 'stanford-nlp', 'python', 'r', 'numpy', 'scipy', 'pandas',
     'pytorch', 'keras', 'nlp', 'hadoop', 'python-3.x', 'tensorflow', 'lstm',
     'seaborn', 'plotly', 'scikit-learn', 'BeautifulSoup')
)

add_selectbox_evaluation_method = st.sidebar.selectbox(
    "Please select an algorithm",
    ("ARIMA", "Auto ARIMA", "FbProphet")
)

# initial stationarity #
stationarity = True
if add_selectbox_evaluation_method == "ARIMA":

    with st.expander("Show Adfuller Result"):

        st.subheader("Framework Data Frame")
        st.dataframe(df[add_selectbox_framework])

        st.subheader("AdFuller test")
        stationarity = quick_adfuller(df[add_selectbox_framework])

    if stationarity:
        # select apropriate p and q values #
        p_value = st.sidebar.slider('Select the best p value(pacf)', 0, 16)
        q_value = st.sidebar.slider('Select the best q value(acf)', 0, 16)

        with st.expander("Plot acf"):
            acf_plot(df[add_selectbox_framework])

        with st.expander("Plot pacf"):
            pacf_plot(df[add_selectbox_framework])

        # not nessesary just to make code more readable #
        df_train = df[add_selectbox_framework]

        arima_model = ARIMA(df_train, order=(p_value, 0, q_value))
        arima_fit = arima_model.fit()

        # put the fitted values into a DataFrame to be displayed #
        arima_df = pd.DataFrame(arima_fit.fittedvalues,
                                index=df.index, columns=["fitted"])

        # add the original values to the arima_fit_df dataframe #
        arima_df["Original figures"] = df[add_selectbox_framework]

        # button to display dataframe #
        with st.expander("ARIMA DataFrame"):
            st.subheader("Fitted and Original DataFrame")
            st.dataframe(arima_df)

        # button to display fitted vals #
        with st.expander("Show Model Fit"):
            st.subheader("Fitted and Original Graph")
            st.line_chart(arima_df)

        # 1 YEAR PREDICTION (we have 36 data entries.12 per year) #
        train_val = df[add_selectbox_framework][:24]
        test_val = df[add_selectbox_framework][24:]

        with st.expander("Forecast"):
            # Build Model
            model = ARIMA(train_val, order=(p_value, 0, q_value))
            fitted = model.fit()

            # Residuals #
            residuals = pd.DataFrame(fitted.resid)

            # Forecast
            fc, se, conf = fitted.forecast(3, alpha=0.05)  # 95% conf

            # # Make as pandas series
            fc_series = pd.Series(fc, index=test_val.index)

            # partitioning dataframes #
            col_fc, col_res = st.columns(2)

            with col_fc:
                # display forcast as a series #
                st.subheader("Forecast values")
                st.dataframe(fc_series)
            with col_res:
                # show residuals values #
                st.subheader("Residual values")
                st.write(residuals)

            # show residuals plot #
            plot_ARIMA_residuals(residuals)

            # show forecast plot #
            plot_arima_focast(train_val, test_val, fc_series)

    else:
        st.subheader("Select auto arima")
elif add_selectbox_evaluation_method == "Auto ARIMA":

    # auto arima getting params #
    auto_model = pm.auto_arima(df[add_selectbox_framework].values, start_p=1, start_q=1,
                               test='adf',       # use adftest to find optimal 'd'
                               max_p=4, max_q=4,  # maximum p and q
                               m=1,              # frequency of series
                               d=None,           # let model determine 'd'
                               seasonal=False,   # No Seasonality
                               start_P=0,
                               D=0,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)

    # display auto arima model summary #
    with st.expander("Best model"):
        st.subheader("Auto ARIMA summary")
        st.write(auto_model.summary())
    # train and test set #
    train_auto_arima = df[add_selectbox_framework][:33]
    test_auto_arima = df[add_selectbox_framework][33:]

    auto_model.fit(train_auto_arima)

    # show risiduals #
    with st.expander("Show model residuals"):
        st.subheader("Residuals")
        auto_ARIMA_risiduals(auto_model)

    # model forecast #
    with st.expander("forecast"):
        forecast_auto = auto_model.predict(len(df[add_selectbox_framework]))
        forecast_auto = pd.DataFrame(forecast_auto, index=df.index,
                                     columns=['Prediction'])

        # add the original data to the forecast data #
        forecast_auto["Original"] = df[add_selectbox_framework]

        col3, col4 = st.columns([2, 1])

        col3.subheader("Forecast Graph")
        col3.line_chart(forecast_auto)

        col4.subheader("Forecast DataFrame")
        col4.write(forecast_auto)

elif add_selectbox_evaluation_method == "FbProphet":

    # all this is so that the code can run in prophet #
    df_prophet = df[add_selectbox_framework].copy()
    df_prophet = df_prophet.reset_index()
    #df_prophet["ds"] = df_prophet.loc[0]
    df_prophet["ds"] = df_prophet["month"]
    df_prophet["y"] = df_prophet[add_selectbox_framework]

    df_prophet = df_prophet.drop(
        labels=["month", add_selectbox_framework], axis=1)

    # definning model #
    m = Prophet()
    model = m.fit(df_prophet)

    # make predicitons for the next 13 months #
    # 13 because our data starts at the begining of the month while fbProphet starts at the end of the month #
    future = m.make_future_dataframe(periods=13, freq='M')
    forecast = m.predict(future)

    # button to show forecast data #
    with st.expander("Show forecast"):
        st.subheader("Forecast DataFrame")
        st.write(forecast)

    # button to plot forecast #
    with st.expander("Plot forecast"):
        # plot forecast #
        st.subheader("Year Forecast")
        fbProphet_plot(m, forecast)

    # button to plot Trend and Seasonality #
    with st.expander("Trend and Seasonality"):
        # show trend and seasonality #
        st.subheader("Model Components")
        fbProphet_plot_components(m, forecast)

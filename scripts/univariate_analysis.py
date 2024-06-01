import numpy as np
import pandas as pd
import statsmodels
import seaborn as sns

import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots

# plotly
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

""""""""""""""""""""""""""""""""""""""""""""""""" DESCRIPTIVE STATISTICS """""""""""""""""""""""""""""""""""""""""""""""""
def generate_descriptive_statistics(df):
    """
    Generate descriptive statistics of a dataframe. All the values are rounded by 3 decimals. Generate a dataframe and transform it into a plotly table
    
    Args
        df (dataframe): dataframe input

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # generate table to save
    list_percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    statistics = df.describe(percentiles = list_percentiles)
    
    # round 3 decimals
    statistics = statistics.round(3)

    # reset index
    statistics.reset_index(inplace = True)

    # transform dataframe into plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(statistics.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[statistics[col] for col in statistics.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    # update title
    fig.update_layout(
      title_text = 'Descriptive Statistics',
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )

    return fig

"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""





""""""""""""""""""""""""""""""""""""""""""""""""" HISTOGRAMS """""""""""""""""""""""""""""""""""""""""""""""""
def plot_individual_hist_segment(df, var_segment, feature_hist):
    """
    Plot individual hist
    Args
        df (dataframe): input dataframe
        varg_segment (string): name of the column in the input dataframe that indicate the differents segments in the data
        feature_hist (string): name of the feature in the input dataframe that will plot its histogram

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    fig = px.histogram(df, x = feature_hist, color = var_segment, barmode='overlay', opacity=0.4)

    # update title
    fig.update_layout(
      title_text = f'Histogram: {feature_hist}',
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )
    return fig


def plot_histogram(df, feature_hist, type_hist = 'automatically_plotly', nbins = 10):
    """
    Plot histogram of a individual feature

    Args
        df (dataframe): dataframe input
        feature (string): feature to calculate the hist
        type_hist (string): selection type of histograms. Choices: ['automatically_ploty', 'indicate_number_bins', 'automatically_np']
            automatically_ploty: automatilly histogram created by plotly
            indicate_number_bins: indicate the number of bins in plotly histogram
            automatically_np: calcuate bins and count using hist of numpy and plot this in a bar graph
        nbins (number): number of bins, used when select the choice "indicate_number_bins"

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    choices_hist_type = ['automatically_ploty', 'indicate_number_bins', 'automatically_np']
    
    # automatically hist create
    if type_hist == 'automatically_plotly':
        fig = px.histogram(df, x = feature_hist)
    
    # indicate number of bins
    elif type_hist == 'indicate_number_bins':
        fig = px.histogram(df, x = feature_hist, nbins = nbins)

    # calculate hist and bins
    elif type_hist == 'automatically_np':
        counts, bins = np.histogram(df[feature_hist])
        bins = 0.5 * (bins[:-1] + bins[1:])
        fig = px.bar(x=bins, y=counts, labels={'x':feature_hist, 'y':'count'}, text_auto=True) # TODO: hover shows the interval instead the mean

    else:
        print('bad input type_hist -- plot default hist')
        fig = px.histogram(df, x = feature_hist)


    # update title
    fig.update_layout(
      title_text = f'Histogram: {feature_hist}',
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )
    
    return fig

def plot_multiple_hist(df, number_columns = 2):
    """
    Plot multiple hist for each feature in the dataframe
    
    Args
        df (datafame)

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # get list features
    list_features = df.columns.tolist()


    # get number of rows (number row = number of data / number of columns)
    # (considering fixed the number of columns) 
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)


    ############################## 
    # Create los subplots
    fig = make_subplots(rows = number_rows, cols = number_columns, shared_xaxes=False, subplot_titles=list_features)

    # add each histogram
    for index_feature, feature in enumerate(list_features):

        # obtener índices en el subplot (en plotly los índices comienzan en 1, por lo que debe sumarse un 1 a los resultados obtenidos)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1
        
        # add histogram
        fig.add_trace(go.Histogram(x = df[feature], name = feature), row = row, col = column)
        

    # update layout
    fig.update_layout(height=len(list_features)*250, 
                      width=1600, 
                      title_text = "Histograms",
                      title_x = 0.5, # centrar titulo
                    title_font = dict(size = 28)
                     )
    #fig.update_layout( title_text = "Histograms")

    
    return fig



def plot_kde_hist(df, number_columns = 2):
    """
    Plot the histogram and the KDE.
    Using seaborn

    Args
        df (dataframe): data. The index should be the timestamp

    Return
        fig (figure matplotlib): fig of matplotlib with the plot generated
    """

    ############################################################################
    # get list of features
    list_features = df.columns.tolist()
    
    
    # define number of rows with a number of columns fixed pass as parameter
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)

    
    # create subplots
    fig, axes = plt.subplots(nrows = number_rows, 
                             ncols = number_columns,
                             #figsize = (subplot_width * number_columns, subplot_height * number_rows),
                             figsize=(7*number_columns, 4*number_rows + 0),
                             tight_layout = True
                            )
    sns.set(style = "darkgrid", palette="gray")
    
    
    # add title
    #fig.suptitle("Histogram with kde", fontsize=28)  # sometimes the tittle is overlaped in the plots
    
    # add subplot for each of the features -> feature
    for index_feature, feature in enumerate(list_features):
        row = (index_feature // number_columns) #+ 1 # in matplotlib index starts in 0, in plolty starts in 1
        column = (index_feature % number_columns) #+ 1
    
        # subplot each feature
        sns.histplot(df, x = feature, kde=True, color='gray', element='step', fill=True, ax=axes[row, column])
        axes[row, column].set_title(f'Histogram and KDE of "{feature}"')
    
    # adjust design
    plt.subplots_adjust(top=0.95) # sup title above the subplots
    
    ############################## 

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""""""""""""" TENDENCY """""""""""""""""""""""""""""""""""""""""""""""""
def plot_tendency(df, feature_plot):
    """
    Plot the individual tendency of a feature present in a dataframe

    Args
        df (dataframe): data. The index should be the timestamp
        feature_plot (string): name of the feature that will be ploted

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    fig = px.line(df, x = df.index, y = feature_plot, title = f'plot_{feature_plot}')

    # update title
    fig.update_layout(
      title_text = feature_plot,
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )
    
    return fig

def plot_multiple_tendency(df, number_columns = 2):
    """
    Plot multiple tendency for each feature in the dataframe
    
    Args
        df (datafame): dataframe input
        number_columns (integer): number of columns, by default ONE column

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # get list of features
    list_features = df.columns.tolist()

    # get number of rows (number row = number of data / number of columns)
    # (considering fixed the number of columns) 
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)


    ############################## 
    # create subplots
    fig = make_subplots(rows = number_rows, 
                        cols = number_columns, 
                        subplot_titles = tuple(list_features)
                       )

    # add subplot for each of the features -> feature
    for index_feature, feature in enumerate(list_features):
    
        # obtener índices en el subplot (en plotly los índices comienzan en 1, por lo que debe sumarse un 1 a los resultados obtenidos)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

    # subplot each feature
        fig.add_trace(go.Scatter(
            x = df.index, 
            y = df[feature],
            name = feature
            ), 
        row = row, 
        col = column
        )
  
  # change shape subplot
    fig.update_layout(
            height = 550 * number_rows, # largo
            width = 1850 * number_columns, # ancho
      title_text="Plots of tendency",
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 28)
      )
    
    ############################## 

    return fig


def plot_all_trend_oneplot(df):
    """
    Plot the trend of all dataframes into one plot. All features with differents with its own scale

    Args
        df (dataframe): input dataframe

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    # create figure
    fig = go.Figure()
    
    # plot each trend
    for index, feature in enumerate(df.columns.tolist()):
        fig.add_trace(go.Scatter(
            x = df.index,
            y = df[feature],
            name = f"trend {feature}",
            yaxis = f"y{index+1}"
        ))
    
    
    # generate axis_configuracion dictionary
    axis_configurations = {
        "yaxis1": dict(title="yaxis1 title")
    }
    for index, feature in enumerate(df.columns.tolist()[1:]):  # since second feature because first feature yaxis already has defined
        axis_configurations[f"yaxis{index+2}"] = dict(title = f"yaxis {feature}", anchor="free", overlaying="y", autoshift=True, title_standoff=0)
    
    
    # update loyout according axis_configuration dictionary
    for axis_name, config in axis_configurations.items():
        fig.update_layout(**{axis_name: config})

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""




""""""""""""""""""""""""""""""""""""""""""""""""" TENDENCY - BOXPLOT FOR MONTHS AND YEAR """""""""""""""""""""""""""""""""""""""""""""""""
def plot_boxplot_months(df, feature_plot):
    """
    Plot boxplots of each month and each year. See the montly distribution of one feature

    Args
        df (dataframe): data. The index should be the timestamp
        feature_plot (string): name of the feature that will be ploted

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """


    fig = px.box(df, x=df.index.month, y = feature_plot, color = df.index.year)
    
    # Configure plot
    fig.update_layout(title = f'Boxplot of "{feature_plot}" for month and year',
                      xaxis_title='Month',
                      yaxis_title='Value',
                      legend_title='Year',
                      title_x = 0.5, # center
                      title_font = dict(size = 20),
                      showlegend=True
                     )
    return fig



def plot_multiple_boxplot_months(df, number_columns = 1):
    """
    Plot boxplots of each month and each year. See the montly distribution of ALL features

    Args
        df (datafame): dataframe input
        number_columns (integer): number of columns, by default ONE column

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    # get list of features
    list_features = df.columns.tolist()

    # get number of rows (number row = number of data / number of columns)
    # (considering fixed the number of columns) 
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)


    ############################## 
    # create subplots
    fig = make_subplots(rows = number_rows, 
                        cols = number_columns, 
                        subplot_titles = df.columns,
                        shared_xaxes=False
                        #vertical_spacing = 0.2 / len(number_rows) # with this parameter is possible reduce the vertical space, but the figure size need to be modified because the subplots become biggers
                       )

    # add subplot of boxplots for each month and year
    for index_feature, feature in enumerate(list_features):

        # obtener índices en el subplot (en plotly los índices comienzan en 1, por lo que debe sumarse un 1 a los resultados obtenidos)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1
        
        # boxplot
        box_fig = px.box(df, x=df.index.month, y=feature, color=df.index.year)
        for trace in box_fig.data:
            fig.add_trace(trace, row = row, col = column)


  # adjust plot
    fig.update_layout(title = 'Boxplots for Month and Year',
                      xaxis_title='Month',
                      yaxis_title='Value',
                      legend_title='Year',
                      title_x=0.5,  # center
                      title_font=dict(size=20),
                      height=1450 * number_rows,  # largo
                      width=1850 * number_columns, # ancho
                      showlegend=True,
                      boxmode='group',  # Group boxplots by month
                      boxgap=0.2)  # Adjust the gap between grouped boxplots
    ############################## 

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""""""""""""" SMOOTH DATA """""""""""""""""""""""""""""""""""""""""""""""""
def plot_compare_tendencias(df_original, df_smoothed, kind_smooth, number_columns=2):
    '''plot_compare_tendencias
    Plot all the features in two differents dataframes in only one plot. The idea is compare the tendency of two differents dataframes where
    one dataframe is the original and the second is the dataframe with smoothed values

    Each feature is ploted in one subplot
    
    Args
        df_original (dataframe): original dataframe
        df_smoothed (dataframe): smoothed dataframe
        kind_smooth (string): kind of smooth. In the plot is showed only in the title
        number_columns (int): number of columns in the subplot. by default 2 columns
    '''
    # get list of features of both dataframes
    list_features = list(set(df_original.columns.tolist() + df_smoothed.columns.tolist()))

    # calculate number of rows (considering the number of colums passed as args)
    if (len(list_features) % number_columns) != 0:
        number_rows = (len(list_features) // number_columns) + 1
    else:
        number_rows = (len(list_features) // number_columns)

    # create fig to plot
    fig = make_subplots(rows=number_rows, cols=number_columns, subplot_titles=tuple(list_features))

    ########## for each feature plot:
    for index_feature in range(len(list_features)):
        feature = list_features[index_feature]

        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

        # plot feature of df_original in gray
        if feature in df_original.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_original.index,
                    y=df_original[feature],
                    name='original - ' + feature,
                    line=dict(color='gray')
                ),
                row=row,
                col=column
            )

        # plot feature of df_smoothed in orange
        if feature in df_smoothed.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_smoothed.index,
                    y=df_smoothed[feature],
                    name='df_smoothed - ' + feature,
                    line=dict(color='orange')
                ),
                row=row,
                col=column
            )

    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text = f"Compare smooth data: {kind_smooth}",
        title_x=0.5,
        title_font = dict(size = 28)
    )

    return fig


def apply_moving_average(df, window_size):
    """
    Moving average
    
    Args
        df (dataframe)
        window_size (int)
    
    Return
        df_smoothed (dataframe)
    """
    df_smoothed = df.rolling(window = window_size).mean()
    df_smoothed = df_smoothed.dropna()

    return df_smoothed

def apply_weighted_moving_average(df, weights):
    '''
    Calcula el promedio móvil ponderado de una serie de datos. "rolling" junto con "dot" para realizar la multiplicación y la suma ponderada. 
    
    Args
        df (dataframe)
        weights (list) una lista o array de pesos correspondientes a cada lag
    
    Return
        df_smoothed (dataframe)
    '''
    window_size = len(weights)
    weights = np.array(weights)
    
    # Extraer los valores de la columna de datos
    values = df.iloc[:, 0].values
    
    # Calcular el promedio móvil ponderado utilizando rolling y dot
    rolling_weights = df.rolling(window_size).apply(lambda x: np.dot(x, weights))
    df_smoothed = rolling_weights / sum(weights)

    # dropna
    df_smoothed = df_smoothed.dropna()
    return df_smoothed

def apply_exponential_moving_average(df, alpha):
    """
    Moving average
    
    Args
        df (dataframe)
        alpha (int)
    
    Return
        df_smoothed (dataframe)
    """
    df_smoothed = df.ewm(adjust = False,alpha = alpha).mean()
    df_smoothed = df_smoothed.dropna()

    return df_smoothed

"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""""""""""""" ACF plotly """""""""""""""""""""""""""""""""""""""""""""""""
def plot_individual_acf(df, feature_plot, lags):
    """
    Plot the individual ACF of a feature of with x number of lags

    Args
        df (dataframe): data. The index should be the timestamp
        feature_plot (string): name of the feature that will be ploted
        lags (int): Number of lags in the ACF

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    # calculate autocorrelation function with for N lags
    df_autocorr = pd.DataFrame() 
    for lag in range(1, lags+1):
        aux_value_autocorr = df[feature_plot].autocorr(lag = lag)  # calculate autocorrelation lag N
        df_autocorr.loc[lag, feature_plot] = aux_value_autocorr
    df_autocorr.index.rename('lags', inplace = True)
    df_autocorr.reset_index(inplace = True)
    
    
    # Crea una figura con Plotly
    fig = go.Figure()
    
    # add bar chart autocorrelation
    fig.add_trace(go.Bar(x=df_autocorr['lags'], 
                             y = df_autocorr[feature_plot], 
                             width=0.3,
                             name='ACF'))
    
    # modify layout
    fig.update_layout(
        title_text='Autocorrelation',
        xaxis=dict(title = 'Lags'),
        yaxis=dict(title = f'Autocorrelation {feature_plot}'),
        title_x = 0.5, # centrar titulo
        title_font = dict(size = 20)
    )

    return fig

def plot_all_acf(df, lags, number_columns = 2):
    """
    Plot the individual ACF of ALL FEATURES of with x number of lags

    Args
        df (dataframe): data. The index should be the timestamp
        lags (int): Number of lags in the ACF

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ############################################################################
    """ calculate autocorrelation """
    # calculate autocorrelation function with for N lags
    df_autocorr = pd.DataFrame() 

    # calcular autocorrelación de cada feature
    for feature in df.columns.tolist():
        #print(f'calculating ACF {feature}...')
        for lag in range(1, lags+1):
            aux_value_autocorr = df[feature].autocorr(lag = lag) # calculate autocorrelation lag N
            df_autocorr.loc[lag, feature] = aux_value_autocorr
    df_autocorr.index.rename('lags', inplace = True)
    df_autocorr.reset_index(inplace = True)
    ############################################################################

    
    ############################################################################
    """ plot autocorrelations """
    # get list of features
    list_features = df.columns.tolist()

    # create subplots
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)
    
    fig = make_subplots(rows = number_rows, 
                        cols = number_columns, 
                        subplot_titles = tuple(list_features)
                       )

    # add subplot for each of the features -> feature
    for index_feature, feature in enumerate(list_features):
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

        # subplot each feature
        fig.add_trace(go.Bar(x = df_autocorr['lags'], 
                             y = df_autocorr[feature], 
                             width = 0.3,
                             name = f'ACF {feature}'),
                row = row, 
                col = column
                     )
  
  # change shape subplot
    fig.update_layout(
            height = 350 * number_rows, # largo
            width = 850 * number_columns, # ancho
      title_text="Plots of Autocorrelation",
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 28)
      )
    
    ############################## 

    return fig

"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""""""""""""" ACF statsmodels-seaborn """""""""""""""""""""""""""""""""""""""""""""""""
def plot_individual_acf_stats(df, feature_plot, lags):
    """
    Plot the individual ACF of a feature of with x number of lags. 
    ->ACF generated by statsmodels

    Args
        df (dataframe): data. The index should be the timestamp
        feature_plot (string): name of the feature that will be ploted
        lags (int): Number of lags in the ACF

    Return
        fig (figure matplotlib): fig of matplotlib with the acf generated by statsmodels
    """
    fig = tsaplots.plot_acf(df[feature_plot], 
                            lags = lags)

    return fig


def plot_all_acf_stats(df, lags, number_columns = 2):
    """
    Plot the individual ACF of ALL FEATURES of with x number of lags
    ->ACF generated by statsmodels

    Args
        df (dataframe): data. The index should be the timestamp
        lags (int): Number of lags in the ACF

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ############################################################################
    # get list of features
    list_features = df.columns.tolist()
    
    
    # define number of rows with a number of columns fixed pass as parameter
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)

    
    # create subplots
    fig, axes = plt.subplots(nrows = number_rows, 
                             ncols = number_columns,
                             #figsize = (subplot_width * number_columns, subplot_height * number_rows),
                             figsize=(7*number_columns, 4*number_rows + 0),
                             tight_layout = True
                            )
    
    # add title
    #fig.suptitle("Plots of Autocorrelation", fontsize=28)  # sometimes the tittle is overlaped in the plots
    
    # add subplot for each of the features -> feature
    for index_feature, feature in enumerate(list_features):
        row = (index_feature // number_columns) #+ 1 # in matplotlib index starts in 0, in plolty starts in 1
        column = (index_feature % number_columns) #+ 1
    
        # subplot each feature
        tsaplots.plot_acf(df[feature], lags=lags, ax=axes[row, column])
        axes[row, column].set_title(f'ACF of "{feature}"')
    
    # adjust design
    plt.subplots_adjust(top=0.95) # sup title above the subplots
    
    ############################## 

    return fig


"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""""""""""""" PACF """""""""""""""""""""""""""""""""""""""""""""""""
def plot_individual_pacf(df, feature_plot, lags):
    """
    Plot the individual PACF of a feature of with x number of lags

    Args
        df (dataframe): data. The index should be the timestamp
        feature_plot (string): name of the feature that will be ploted
        lags (int): Number of lags in the ACF

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    # calculate PACF function with for N lags
    aux_value_pacf, ci = statsmodels.tsa.stattools.pacf(df[feature_plot], nlags = lags, method = 'ywadjusted', alpha = 0.05)
    df_pacf = pd.DataFrame(aux_value_pacf, columns = [feature_plot])    
    df_pacf.index.rename('lags', inplace = True)
    df_pacf.reset_index(inplace = True)
    
    
    # Crea una figura con Plotly
    fig = go.Figure()
    
    # add bar chart autocorrelation
    fig.add_trace(go.Bar(x=df_pacf['lags'], 
                             y = df_pacf[feature_plot], 
                             width=0.3,
                             name='ACF'))
    
    # modify layout
    fig.update_layout(
        title_text='Autocorrelation',
        xaxis=dict(title = 'Lags'),
        yaxis=dict(title = f'Partial Autocorrelation {feature_plot}'),
        title_x = 0.5, # centrar titulo
        title_font = dict(size = 20)
    )

    return fig

def plot_all_pacf(df, lags, number_columns = 2):
    """
    Plot the individual PACF of ALL FEATURES of with x number of lags

    Args
        df (dataframe): data. The index should be the timestamp
        lags (int): Number of lags in the ACF

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ############################################################################
    """ calculate autocorrelation """
    # calculate autocorrelation function with for N lags
    df_pacf = pd.DataFrame()
    for feature in df.columns.tolist():
        #print(f'calculating PACF {feature}...')
        aux_value_pacf, ci = statsmodels.tsa.stattools.pacf(df[feature], nlags = lags, method = 'ywadjusted', alpha = 0.05)
        df_pacf[feature] = pd.DataFrame(aux_value_pacf, columns = [feature])   
    df_pacf.index.rename('lags', inplace = True)
    df_pacf.reset_index(inplace = True)
    ############################################################################

    
    ############################################################################
    """ plot autocorrelations """
    # get list of features
    list_features = df.columns.tolist()

    # create subplots
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)
    
    fig = make_subplots(rows = number_rows, 
                        cols = number_columns, 
                        subplot_titles = tuple(list_features)
                       )

    # add subplot for each of the features -> feature
    for index_feature, feature in enumerate(list_features):
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

        # subplot each feature
        fig.add_trace(go.Bar(x = df_pacf['lags'], 
                             y = df_pacf[feature], 
                             width = 0.3,
                             name = f'ACF {feature}'),
                row = row, 
                col = column
                     )
  
  # change shape subplot
    fig.update_layout(
            height = 350 * number_rows, # largo
            width = 850 * number_columns, # ancho
      title_text="Plots of Partial Autocorrelation",
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 28)
      )
    
    ############################## 

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""""""""""""" PACF - statsmodels - seaborn """""""""""""""""""""""""""""""""""""""""""""""""
def plot_individual_pacf_stats(df, feature_plot, lags):
    """
    Plot the individual PACF of a feature of with x number of lags. 
    ->PACF generated by statsmodels

    Args
        df (dataframe): data. The index should be the timestamp
        feature_plot (string): name of the feature that will be ploted
        lags (int): Number of lags in the PACF

    Return
        fig (figure matplotlib): fig of matplotlib with the pacf generated by statsmodels
    """
    fig = tsaplots.plot_pacf(df[feature_plot], 
                            lags = lags)

    return fig


def plot_all_pacf_stats(df, lags, number_columns = 2):
    """
    Plot the individual PACF of ALL FEATURES of with x number of lags
    ->PACF generated by statsmodels

    Args
        df (dataframe): data. The index should be the timestamp
        lags (int): Number of lags in the PACF

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ############################################################################
    # get list of features
    list_features = df.columns.tolist()
    
    
    # define number of rows with a number of columns fixed pass as parameter
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)

    
    # create subplots
    fig, axes = plt.subplots(nrows = number_rows, 
                             ncols = number_columns,
                             #figsize = (subplot_width * number_columns, subplot_height * number_rows),
                             figsize=(7*number_columns, 4*number_rows + 0),
                             tight_layout = True
                            )
    
    # add title
    #fig.suptitle("Plots of Autocorrelation", fontsize=28)  # sometimes the tittle is overlaped in the plots
    
    # add subplot for each of the features -> feature
    for index_feature, feature in enumerate(list_features):
        row = (index_feature // number_columns) #+ 1 # in matplotlib index starts in 0, in plolty starts in 1
        column = (index_feature % number_columns) #+ 1
    
        # subplot each feature
        tsaplots.plot_pacf(df[feature], lags=lags, ax=axes[row, column])
        axes[row, column].set_title(f'PACF of "{feature}"')
    
    # adjust design
    plt.subplots_adjust(top=0.95) # sup title above the subplots
    
    ############################## 

    return fig
""""""""""""""""""""""""""""""""""""""""""""""""" """""""""""""""""""""""""""""""""""""""""""""""""
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


""""""""""""""""""""""""""""""""""""""""""""""""" CORRELATIONS """""""""""""""""""""""""""""""""""""""""""""""""
# auxiliar function - filter correlation by threshold
def filter_correlations_by_threshold(df, threshold):
    """
    Given a dataframe and a threshold, transform all the values BELOW the threshold (in absolute value) into NaN
    Args:
        df (dataframe): dataframe with correlations (this dataframe can have null values)
        threshold (int): 

    Return
        df_threshold (dataframe): dataframe output correlations
    """   
    # if threshold is none, set it in 0
    if threshold == None:
        threshold = 0
    
    # transform values in absolute value below the theshold into nan
    mask = (df <= -threshold) | (df >= threshold)
    #df_threshold = df.mask(mask, np.nan)
    df_threshold = df.where(mask)
    return df_threshold


# auxiliar function - plot heatmap correlations
def plot_heatmap(df_corr):
    """
    Plot heatmap using the input dataframe
    It could be used to plot the correlations between differents variables

    Args
        df_corr (dataframe): dataframe with correlations to plot

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    # heatmap
    fig = px.imshow(df_corr, text_auto=True, aspect="auto")
    
    # change title
    fig.update_layout(
      title_text = "Correlations",
        title_x = 0.5,
    title_font = dict(size = 28)
      )
    
    return fig


# calculate correlations between each features
def calculate_correlations_triu(df):
    """
    Given a dataframe, calculate the correlations (pearson) between all the variables in the dataframe
    Args
        df (dataframe)

    Return
        df_corr (dataframe): dataframe with correlations
        df_corr_upper(dataframe): dataframe with correltions - upper triangular matrix - round by 2 decimals
  """

    # calculate correlations
    df_corr = df.corr(method='pearson')
    
    # upper triangular matrix
    df_corr_upper = df_corr.where(np.triu(np.ones(df_corr.shape)).astype('bool'))
    
    # round 2 decimals
    df_corr = np.round(df_corr, 2)
    df_corr_upper = np.round(df_corr_upper, 2)
    
    return df_corr, df_corr_upper


# calculate correlations between each feature against the target
def calculate_correlations_target(df, target):
    """
    Given a dataframe and a target (that will be present in the dataframe) calculate the correlations of all features agains the target

    Args
        df (dataframe): dataframe
        target (string): feature target - that will be present in the dataframe
    
    Return
        df_corr (dataframe): dataframe with the correlations
    """

    # calculate correlations select only with the target
    df_corr_target = df.corr(method='pearson')[[target]]
    
    # roudn 3 decimals
    df_corr_target = np.round(df_corr_target, 3)
    
    # transpose to see in a better way
    df_corr_target = df_corr_target.T
    
    return df_corr_target
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""""""""""""" SCATTER PLOT """""""""""""""""""""""""""""""""""""""""""""""""
def plot_individual_scatter_plot(df, feature_x, feature_y, marginal_hist = False):
    """
    Create an individual scatter plot between two variables
    
    Args
        df (dataframe): input dataframe with the feature to plot in the scatter plot
        feature_x (string): name of the feature in x-axis
        feature_y (string): name of the feature in y-axis
        marginal_hist (bool): plot a histogram as marginal (feature_x and feature_y). By default in false
    
    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    # plot scatter plot
    if marginal_hist == True:
        fig = px.scatter(df, x = feature_x, y = feature_y, marginal_x = "histogram", marginal_y="histogram", trendline="ols")
        tittle_plot = f'scatter plot: {feature_x} vs {feature_y}. Marginal distributions'
    else:
        fig = px.scatter(df, x = feature_x, y = feature_y, trendline="ols")
        tittle_plot = f'scatter plot: {feature_x} vs {feature_y}'

    # change color trendline to redgenerated
    fig.data[-1]['marker']['color'] = '#d62728' # change color to brick red

    # update title
    fig.update_layout(
      title_text = tittle_plot,
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )

    return fig

def plot_features_to_target_scatter_plot_low(df, target, number_columns=2):
    """
    Create multiples plots (subplots) of the scatter plot between a list of features againts the target.
    -> All scatter plot with the same color. low resources, the pc is freeze to me doing scatter plot with different color

    Args
        df (dataframe): input dataframe with features and target to plot in the scatter plot
        target (string): target to be ploted in each graph
        number_columns (int): number of columns in the subplot. by default 2 columns

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """
    # get list of features
    list_features = df.columns.tolist()
    
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


        # get fig individual
        fig_aux = px.scatter(df, x = feature, y = target, trendline = "ols")
        
        # add scatter to fig global
        fig.add_trace(fig_aux.data[0],
            row = row,
            col = column
        )
        # add trendile to fig global
        trendline_ux = fig_aux.data[1]
        trendline_ux['marker']['color'] = '#d62728' # change color to brick red
        fig.add_trace(trendline_ux,
                     row = row,
                     col = column)
    
    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text = "Compare scatters features againts a target",
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig


def plot_features_to_target_scatter_plot(df, target, number_columns=2):
    """
    Create multiples plots (subplots) of the scatter plot between a list of features againts the target

    Args
        df (dataframe): input dataframe with features and target to plot in the scatter plot
        target (string): target to be ploted in each graph
        number_columns (int): number of columns in the subplot. by default 2 columns

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """
    # get list of features
    list_features = df.columns.tolist()
    
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


        # get fig individual
        trace = go.Scatter(
            x = df[feature],
            y = df[target],
            mode = 'markers',
            name = f'plot - {feature} vs {target}'
        )
        
        # add to fig global
        fig.add_trace(trace,
            row=row,
            col=column
        )

    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text = "Compare scatters features againts a target",
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig


##### scatter plot between all features - scatter matrix - mine function and default function plotly
def list_map_features_features(df):
    """
    generate a list of tuples of each pair of features to generate a tuple(feature_x, feature_y) to generate a bivariate plot
    Args
        df (dataframe): dataframe with the features to plot
    Return
        list_pair_features: list with the tuples to plot
    """
    # create dataframe with each cell is a tuple formed by a pari (row,column)
    df_tuple_features = pd.DataFrame(columns = df.columns.tolist(), index = df.columns.tolist())
    for column in df_tuple_features.columns:
        for index in df_tuple_features.index:
            df_tuple_features.at[index, column] = (index, column)
    df_tuple_features = df_tuple_features.where(np.triu(np.ones(df_tuple_features.shape), k=1).astype('bool'))
    
    # get a list of tuple of each pair of features to do a scatter plot
    stacked_series = df_tuple_features.stack().dropna()
    list_pair_features = list(stacked_series)

    return list_pair_features



def plot_all_features_scatter_plot_mine(df, number_columns=2):
    """
    MY FUNCTION TO DO SCATTER MATRIX. https://plotly.com/python/splom/. TOO HEAVY WITH HIGH DIMENSIONALITY DATA
    
    Create multiples plots (subplots) of the scatter plot between all features againts all features
    -> All scatter plot with the same color. low resources, the pc is freeze to me doing scatter plot with different color

    Args
        df (dataframe): input dataframe with features and target to plot in the scatter plot
        target (string): target to be ploted in each graph
        number_columns (int): number of columns in the subplot. by default 2 columns

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """
    # get list of features
    list_features = df.columns.tolist()
    
    ################# generate a list of tuples of each pair of features to generate a scatter plot  #####################
    list_pair_features = list_map_features_features(df)


    ####################### plot #################################
    
    # calculate number of rows (considering the number of colums passed as args)
    if (len(list_pair_features) % number_columns) != 0:
        number_rows = (len(list_pair_features) // number_columns) + 1
    else:
        number_rows = (len(list_pair_features) // number_columns)

    # create fig to plot
    fig = make_subplots(rows=number_rows, cols=number_columns, 
                        subplot_titles = tuple([str(tupla) for tupla in list_pair_features]) ### title of each subplots
                       )

    ########## for each tuple of features to plot:
    for index_feature, (feature_x, feature_y) in enumerate(list_pair_features):

        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

        # get fig individual
        fig_aux = px.scatter(df, x = feature_x, y = feature_y, trendline = "ols")
        
        # add scatter to fig global
        fig.add_trace(fig_aux.data[0],
            row = row,
            col = column
        )
        # add trendile to fig global
        trendline_ux = fig_aux.data[1]
        trendline_ux['marker']['color'] = '#d62728' # change color to brick red
        fig.add_trace(trendline_ux,
                     row = row,
                     col = column)
    
    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text = "Compare scatters for each feature againts each feature",
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig


def plot_all_features_scatter_plot(df):
    """  
    Create multiples plots (subplots) of the scatter plot between all features againts all features

    Args
        df (dataframe): input dataframe with features to plot in the scatter matrix features vs features

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """
    fig = px.scatter_matrix(df)

    # update title
    fig.update_layout(
      title_text = "Compare scatters for each feature againts each feature",
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""""""""""""" CORRELATIONS FEATURES LAGGED VS TARGET """""""""""""""""""""""""""""""""""""""""""""""""
def calculate_corr_features_lag_target(df, target, lags):
    """
    Calculate correlation of each feature againt a target. Each correlation is calculated with the feature lagged differents instance of time
    Ex. corr target vs feature1 (lag) with lag [0, 1, 2, ..., n]
    
    Args
        df (dataframe): input dataframe
        target (string): name of the target. the target needs to be inside the dataframe df
        lags (integer): number of lags to apply to calculate the correlations
    
    Return
        df_corr_lags(dataframe): dataframe with the correlations of the features lagged against a target
    """
    # initialize dataframe with correlations with lags in the features. row: lags // columns: features
    df_corr_lags = pd.DataFrame()
    df_to_lag = df.copy() # in this code is generated a new column target to lag it. so it is necesary clone the data to not affect the oriignal df
    
    # create a column additional (target_shift) that is a clone of the target. see the correlation of the target with itself (autocorrelation)
    df_to_lag[target + '_shift'] = df_to_lag[target]
    
    # define list feature and list target. Only list_feature is lagged
    list_all = df.columns.tolist()
    list_features = list(set(df_to_lag.columns.tolist()) - set([target]))
    list_target = [target]

    # calculate correlation
    for lag in range(lags+1):
        # verbose
        df_original_data = df_to_lag.copy()
        if lag % 20 == 0:
            print(f'calculating corr with lag: {lag}')
        
        # lag the data
        df_shifted = df_original_data[list_features].shift(lag)
        df_shifted[list_target] = df_original_data[list_target]
        df_shifted = df_shifted[list_all]
        
        # calculate correlation feature lagged against the target
        aux_corr_lags = df_shifted.corr()[list_target].T.reset_index(drop = True)
        aux_corr_lags.index = [lag]
        
        # save results
        df_corr_lags = pd.concat([df_corr_lags, aux_corr_lags])

    # round correlation with 3 decimals
    df_corr_lags = np.round(df_corr_lags, 3)
    
    return df_corr_lags


def plot_corr_features_lag_target(df_corr_lags, number_columns=2):
    """
    Plot correlation of each feature againt a target. Each correlation is calculated with the feature lagged differents instance of time
    The plot is a line. Axis y: corr. Axis x: lag

    Args
        df_corr_lags(dataframe): dataframe with the correlations of the features lagged against a target
        number_columns (int): number columns in the plot

    Return
        fig (figure plotly): fig of plotly with the plot generated 

    TODO:
        - inverter the y axis when the correlation is negative
    """

    # get list of features
    list_features = df_corr_lags.columns.tolist()

    # get number of rows (number row = number of data / number of columns) (considering fixed the number of columns) 
    if (df_corr_lags.shape[1] % number_columns) != 0:
        number_rows = (df_corr_lags.shape[1] // number_columns) + 1 
    else:
        number_rows = (df_corr_lags.shape[1] // number_columns)


    ############################## 
    # create subplots
    fig = make_subplots(rows = number_rows, 
                        cols = number_columns, 
                        subplot_titles = tuple(list_features),
                        shared_xaxes = False
                       )

    # add subplot for each of the features -> feature
    for index_feature, feature in enumerate(list_features):
    
        # obtener índices en el subplot (en plotly los índices comienzan en 1, por lo que debe sumarse un 1 a los resultados obtenidos)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

        # subplot each feature
        fig.add_trace(go.Scatter(
            x = df_corr_lags.index, 
            y = df_corr_lags[feature],
            name = feature
            ), 
        row = row, 
        col = column
        )

        # add y label and x label each subplot
        fig.update_xaxes(title_text = "Lag", row = row, col = column)
        fig.update_yaxes(title_text = "Correlation", row  =row, col = column)
    
    # change shape subplot
    fig.update_layout(
            height = 350 * number_rows, # largo
            width = 850 * number_columns, # ancho
      title_text="Plots of correlations features lagged against the target",
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 28)
      )


    # Actualizar los títulos de los ejes x en cada subplot
    # for i in range(len(variables)):
    #     fig.update_xaxes(title_text="Lags", row=i+1, col=1, autorange="reversed")
    #     fig.update_yaxes(title_text="Autocorrelación", row=i+1, col=1)
    ############################## 
    
    return fig


""""""""""""""""""""""""""""""""""""""""""""""""" PARALLEL """""""""""""""""""""""""""""""""""""""""""""""""
def plot_parallel_continuous(df, list_features_target, target):
    """
    Plot a parallel with features continous variables an target continuous variable

    Args
        df (dataframe): dataframe with the data
        list_features_target (list): list with the features to plot in the parallel plot and also it has to have the target
        target (string): in addition it is necesary define a string with the name of the target

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # generate df to plot in parallel plot. in this kind of plot duplicated values are not soported and return an error
    df_parallel = df[list_features_target].drop_duplicates()
    
    fig = px.parallel_coordinates(df_parallel, 
                                  color = target,
                                  color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

    # change title
    fig.update_layout(
      title_text = "Parallel continuous features",
        title_x = 0.5,
    title_font = dict(size = 28)
      )

    return fig

""""""""""""""""""""""""""""""""""""""""""""""""" """""""""""""""""""""""""""""""""""""""""""""""""
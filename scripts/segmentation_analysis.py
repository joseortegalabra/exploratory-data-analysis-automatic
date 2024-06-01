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


""""""""""""""""""""""""""""""""""""""""""""""""" CORRELATIONS SEGMENTATION """""""""""""""""""""""""""""""""""""""""""""""""
def calculate_correlations_triu_segmentation(df, var_segment):
    """
    Given a dataframe and a variable that are segmented the data calculate the correlations (pearson) between all the variables
    Args
        df (dataframe): input dataframe
        var_segment (string): variable in the input dataframe that indicate the segments in the data

    Return
        dict_df_corr_segment(dict): dictionary of dataframes with correltions for each segment - upper triangular matrix - round by 3 decimals
  """

    # get name of each segment in a list
    unique_values_segments = df[var_segment].unique().tolist()
    unique_values_segments = list(filter(pd.notna, unique_values_segments)) # delete null values in segments
    unique_values_segments.sort()

    # generate a list of dataframes with each dataframe is the df_corr for each segment
    dict_df_corr_segment = {}
    for name_segment in unique_values_segments:
    
        # generate auxiliar df for each segment
        df_aux = df[df[var_segment] == name_segment]
        df_aux = df_aux.drop(columns = var_segment)
    
        # calculate corr triu with 3 decimals
        df_corr_segment_aux = df_aux.corr()
        df_corr_segment_aux_upper = df_corr_segment_aux.where(np.triu(np.ones(df_corr_segment_aux.shape), k=1).astype('bool'))
        df_corr_segment_aux_upper = np.round(df_corr_segment_aux_upper, 3)
    
        # append to list
        dict_df_corr_segment[name_segment] = df_corr_segment_aux_upper
    
    return dict_df_corr_segment


def calculate_correlations_target_segmentation(df, var_segment, target):
    """
    Given a dataframe and a variable that are segmented the data calculate the correlations (pearson) between all the features against the target
    Args
        df (dataframe): input dataframe
        var_segment (string): variable in the input dataframe that indicate the segments in the data
        target (string): target

    Return
        dict_df_corr_segment(dict): dictionary of dataframes with correltions for each segment - upper triangular matrix - round by 3 decimals
  """

    # get name of each segment in a list
    unique_values_segments = df[var_segment].unique().tolist()
    unique_values_segments = list(filter(pd.notna, unique_values_segments)) # delete null values in segments
    unique_values_segments.sort()

    # generate a list of dataframes with each dataframe is the df_corr for each segment
    dict_df_corr_segment = {}
    for name_segment in unique_values_segments:
    
        # generate auxiliar df for each segment
        df_aux = df[df[var_segment] == name_segment]
        df_aux = df_aux.drop(columns = var_segment)
    
        # calculate corr triu with 3 decimals
        df_corr_segment_aux = df_aux.corr()[[target]]
        df_corr_segment_aux = np.round(df_corr_segment_aux, 3)
        df_corr_segment_aux = df_corr_segment_aux.T
    
        # append to list
        dict_df_corr_segment[name_segment] = df_corr_segment_aux
    
    return dict_df_corr_segment

def filter_correlations_segment_by_threshold(dict_df_corr_segment, threshold):
    """
    Given a dictionary of dataframes with correlations by segments and a threshold, 
    transform all the values BELOW the threshold (in absolute value) into NaN
    
    Args:
        dict_df_corr_segment (dict): dictionary where each element is a dataframe with the correlations for each segment
        threshold (int): 

    Return
        dict_df_corr_segment (dataframe): dictionary with dataframes updated
    """   
    # get list of segments - keys in the dict
    list_segments = list(dict_df_corr_segment.keys())

    for segment in list_segments:
    
        # if threshold is none, set it in 0
        if threshold == None:
            threshold = 0
        
        # transform values in absolute value below the theshold into nan
        mask = (dict_df_corr_segment[segment] <= -threshold) | (dict_df_corr_segment[segment] >= threshold)
        df_threshold_update = dict_df_corr_segment[segment].where(mask)
        

        # replace new dataframe in dict - no optimal code
        dict_df_corr_segment[segment] = df_threshold_update
        
    return dict_df_corr_segment


def plot_corr_segmentation_vertical_barchart(dict_df_corr_segment):
    """
    Given a dictionary with the correlations for each segment, plot it into a format vertical barchat

    Args
        dict_df_corr_segment (dict): dictionary where each element is a dataframe with the correlations for each segment

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """

    ''' process data corr'''
    # get list of segments - keys in the dict
    list_segments = list(dict_df_corr_segment.keys())
    
    # get a unique dataframe of corr for each segment
    df_corr_segments = pd.DataFrame()
    for segment in list_segments:
        df_corr_aux = dict_df_corr_segment[segment] # get df corr
        df_corr_aux_stack = df_corr_aux.stack().dropna() # stack
        df_corr_segments[segment] = df_corr_aux_stack # merge into a dataframe
    
    # transform index tuple into a string
    index_tuple_string = [str(x) for x in df_corr_segments.index.tolist()]

    ''' plot '''
    # plot corr in barchat
    fig = go.Figure()
    for segment in list_segments:
        fig.add_trace(go.Bar(x= df_corr_segments[segment].tolist() , #x-axis -> each value of correlation
                             y = index_tuple_string,  # y-axis -> each pair of feature in the correlation
                             orientation='h',
                             name = segment  # name - color according the segment
                            )
                     )
    
    # update layout
    fig.update_layout(
        height = 150 * len(index_tuple_string),  # largo
        width = 1850 * 1,  # ancho
        title_text = "Correlations features for each segment",
        title_x = 0.5,
        title_font = dict(size = 20)
    )

    return fig


def plot_corr_segmentation_subplots_heatmap(dict_df_corr_segment, number_columns = 1):
    """
    Given a dictionary with the correlations for each segment, plot it into a format a subplots of heatmaps

    Args
        dict_df_corr_segment (dict): dictionary where each element is a dataframe with the correlations for each segment
        number_columns (int): for the dimensions of heatmaps set it always in 1

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """
    
    # get list of segments - keys in the dict
    list_segments = list(dict_df_corr_segment.keys())
    
    # calculate number of rows (considering the number of colums passed as args)
    if (len(list_segments) % number_columns) != 0:
        number_rows = (len(list_segments) // number_columns) + 1
    else:
        number_rows = (len(list_segments) // number_columns)

    # create fig to plot
    fig = make_subplots(rows=number_rows, cols=number_columns, subplot_titles=tuple(list_segments))

    ########## for each feature plot:
    for index_segment in range(len(list_segments)):
        segment = list_segments[index_segment]

        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_segment // number_columns) + 1
        column = (index_segment % number_columns) + 1


        # get fig individual
        fig_aux = px.imshow(dict_df_corr_segment[segment], text_auto=True, aspect="auto")
        
        # add scatter to fig global
        fig.add_trace(fig_aux.data[0],
            row = row,
            col = column
        )
    
    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text = "Correlations features for each segment",
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig




"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""""""""""""" SCATTER SEGMENTATION """""""""""""""""""""""""""""""""""""""""""""""""
def plot_individual_scatter_plot_segment(df, feature_x, feature_y, var_segment, marginal_hist = False):
    """
    Create an individual scatter plot between a feature_x , feauture_y and color the plot considering the feature segmentation

    Args
        df (dataframe): input dataframe
        feature_x (string): name of the feature in x-axis
        feature_y (string): name of the feature in y-axis
        var_segment (string): name of the variable that segment the data to color the scatter plot
        marginal_hist (bool): plot a histogram as marginal (feature_x and feature_y). By default in false

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """
    if marginal_hist == True:
        fig = px.scatter(df, x= feature_x, y = feature_y, color = var_segment, opacity = 0.1, marginal_x="histogram", marginal_y="histogram")
        tittle_plot = f'scatter plot: {feature_x} vs {feature_y}. color: segments. Marginal distributions'
    else:
        fig = px.scatter(df, x= feature_x, y = feature_y, color = var_segment, opacity = 0.1)
        tittle_plot = f'scatter plot: {feature_x} vs {feature_y}. color: segments'

    # update title
    fig.update_layout(
      title_text = tittle_plot,
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )
    
    return fig

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

def plot_all_features_scatter_plot_segment_mine(df, var_segment, number_columns=2):
    """
    Create multiples plots (subplots) of the scatter plot between all features againts all features
    -> All scatter plot with the same color. low resources, the pc is freeze to me doing scatter plot with different color

    Args
        df (dataframe): input dataframe with features and target to plot in the scatter plot
        target (string): target to be ploted in each graph
        number_columns (int): number of columns in the subplot. by default 2 columns

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """
    
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
        fig_aux = px.scatter(df, x = feature_x, y = feature_y, color = var_segment, opacity = 0.1)
        
        # add scatter to fig global
        for index in range(len(fig_aux.data)):
            fig.add_trace(fig_aux.data[index],
                row = row,
                col = column
            )
            
    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text = "Compare scatters for each feature againts each feature",
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig

def plot_all_features_scatter_plot_segment(df, var_segment):
    """  
    Create multiples plots (subplots) of the scatter plot between all features againts all features

    Args
        df (dataframe): input dataframe with features to plot in the scatter matrix features vs features

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """
    fig = px.scatter_matrix(df, color = var_segment, opacity = 0.1)

    # update title
    fig.update_layout(
      title_text = "Compare scatters for each feature againts each feature",
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )

    return fig

"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""




""""""""""""""""""""""""""""""""""""""""""""""""" DESCRIPTIVE STATISTICS SEGMENTATION """""""""""""""""""""""""""""""""""""""""""""""""
def calculate_descriptive_statistics_segment(df, var_segment):
    """
    Given a dataframe and a variable that are segmented the data calculate the descriptive statistics of each segment
    Args
        df (dataframe): input dataframe
        var_segment (string): variable in the input dataframe that indicate the segments in the data

    Return
        dict_df_statistics_segment(dict): dictionary of dataframes with the descriptive statistics
  """

    # get name of each segment in a list
    unique_values_segments = df[var_segment].unique().tolist()
    unique_values_segments = list(filter(pd.notna, unique_values_segments)) # delete null values in segments
    unique_values_segments.sort()

    # generate a list of dataframes with each dataframe is the df statistics for each segment
    dict_df_statistics_segment = {}
    for name_segment in unique_values_segments:
    
        # generate auxiliar df for each segment
        df_aux = df[df[var_segment] == name_segment]
        df_aux = df_aux.drop(columns = var_segment)
    
        # calculate descriptive statistics
        list_percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        statistics = df_aux.describe(percentiles = list_percentiles)
        statistics = statistics.round(3)
        statistics.reset_index(inplace = True)

        # append to list
        dict_df_statistics_segment[name_segment] = statistics
    
    return dict_df_statistics_segment


def merge_segmentation_statistics(dict_df_statistics_segment):
    """
    Given a dictionary of statistics for each segment given, merge it into a only dataframe with statistics
    """
    
    # Iterar sobre las claves y concatenate dataframes in columns
    result_df = pd.DataFrame()
    for key, df in dict_df_statistics_segment.items():
        df_aux = df.copy()
        df_aux.set_index('index', inplace = True)
        df_aux.columns = [x + '__' + key for x in df_aux.columns.tolist()]
        
        result_df = pd.concat([result_df, df_aux], axis = 1)
    result_df.reset_index(inplace = True)
    
    # list columns - list segments
    list_columns = list(dict_df_statistics_segment[list(dict_df_statistics_segment.keys())[0]].columns)
    list_columns = list_columns[1:] # delete index
    list_segments = list(dict_df_statistics_segment.keys())
    list_segments.sort()
    
    # sort result df
    list_columns_statisctics_sort = []
    for columns in list_columns:
        for segments in list_segments:
            list_columns_statisctics_sort.append(f'{columns}__{segments}')
    list_columns_statisctics_sort = ['index'] + list_columns_statisctics_sort
    result_df = result_df[list_columns_statisctics_sort]

    return result_df


def plot_descriptive_statistics_segment(df_merge_statistics_segments):
    """
    Generate descriptive statistics in a plotly table from a dataframe

    Args
        df_merge_statistics_segments (dataframe): dataframe merged with staticstics for each segment

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df_merge_statistics_segments.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df_merge_statistics_segments[col] for col in df_merge_statistics_segments.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    # update title
    fig.update_layout(
      title_text = 'Descriptive Statistics',
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20),
      height = 850 * 1,  # largo
      width = 150 * len(df_merge_statistics_segments.columns.tolist()),  # ancho
    )

    return fig

"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""""""""""""" HISTOGRAMS - BOXPLOTS - SEGMENTATION """""""""""""""""""""""""""""""""""""""""""""""""
def plot_histograms_segments_old(df, var_segment, number_columns = 2):
    """
    Plot multiple hist for each feature in the dataframe. Differents colors in the histogram according the segmentation in the data
    
    Args
        df (datafame): input dataframe
        varg_segment (string): name of the column in the input dataframe that indicate the differents segments in the data

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
        
        # get histogram individual
        fig_aux = px.histogram(df, x = feature, color = var_segment, barmode='overlay', opacity=0.4)
        
        # add histogram to fig global
        for index in range(len(fig_aux.data)):
            fig.add_trace(fig_aux.data[index],
                row = row,
                col = column
            )

    
    # update layout
    fig.update_layout(height=len(list_features)*250, 
                      width=1600, 
                      title_text = "Histograms Segmentations",
                      title_x = 0.5, # centrar titulo
                    title_font = dict(size = 28)
                     )    
    return fig


def plot_histograms_segments(df, var_segment, number_columns = 2):
    """
    Plot multiple hist for each feature in the dataframe. Differents colors in the histogram according the segmentation in the data
    
    Args
        df (datafame): input dataframe
        varg_segment (string): name of the column in the input dataframe that indicate the differents segments in the data

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # get list features
    list_features = df.columns.tolist()


    # get name of each segment in a list
    unique_values_segments = df[var_segment].unique().tolist()
    unique_values_segments = list(filter(pd.notna, unique_values_segments)) # delete null values in segments
    unique_values_segments.sort()

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
        

        # histogram
        for segment in unique_values_segments:
            df_segment = df[df[var_segment] == segment]
            fig.add_trace(go.Histogram(x = df_segment[feature], 
                                       name = segment,
                                       opacity=0.4), 
                          row = row, col = column)

    
    # update layout
    fig.update_layout(height=len(list_features)*250, 
                      width=1600, 
                      title_text = "Histograms Segmentations",
                      title_x = 0.5, # centrar titulo
                    title_font = dict(size = 28),
                      barmode = 'stack'
                     )    
    return fig


def plot_boxplots_segments(df, var_segment, number_columns = 2):
    """
    Plot multiple boxplots for each feature in the dataframe. Differents colors in the histogram according the segmentation in the data
    
    Args
        df (datafame): input dataframe
        varg_segment (string): name of the column in the input dataframe that indicate the differents segments in the data

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

    # add each boxplot
    for index_feature, feature in enumerate(list_features):

        # obtener índices en el subplot (en plotly los índices comienzan en 1, por lo que debe sumarse un 1 a los resultados obtenidos)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1
        
        # add trace boxplot
        #fig.add_trace(go.Box(x=df[var_segment], y=df[feature], name = f'Boxplot {feature} by segments {var_segment}'),
        fig.add_trace(go.Box(x=df[var_segment], y=df[feature]),
                row = row,
                col = column)
        
    # update layout
    fig.update_layout(height=len(list_features)*250, 
                      width=1600, 
                      title_text = "Boxplots Segmentations",
                      title_x = 0.5, # centrar titulo
                    title_font = dict(size = 28)
                     )
    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""""""""""""" TENDENCY - SEGMENTS """""""""""""""""""""""""""""""""""""""""""""""""
def plot_tendency_segment(df, feature_plot, var_segment):
    """
    Plot the individual tendency of a feature present in a dataframe. Color the tendency according the segment

    Args
        df (dataframe): data. The index should be the timestamp and the dataframe must be sorted by the index
        feature_plot (string): name of the feature that will be ploted
        var_segment (string): name of the variable that segment the data to color the scatter plot

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    # get name of each segment in a list
    unique_values_segments = df[var_segment].unique().tolist()
    unique_values_segments = list(filter(pd.notna, unique_values_segments)) # delete null values in segments
    unique_values_segments.sort()

    # Create figure
    fig = go.Figure()
    
    # generate plot scatter color by segment
    fig_scatter_segment = px.scatter(df, x=df.index, y=feature_plot, title=f'plot_{feature_plot}', color = var_segment, opacity = 0.7)
    
    # generate trend gray
    fig_trend = px.line(df, x = df.index, y = feature_plot, title = f'plot_{feature_plot}')
    fig_trend.data[0]['line']['color'] = '#D3D3D3'
    
    # add to fig
    fig.add_trace(fig_trend.data[0])
    for index_segment in range(len(unique_values_segments)): # add each scatter plot for each segment
        fig.add_trace(fig_scatter_segment.data[index_segment])
    
    
    # update layout
    fig.update_layout(
        title_text=feature_plot,
        title_x=0.5,  # centrar título
        title_font=dict(size=20)
    )
    
    return fig


def plot_multiple_tendency_segmentation(df, var_segment, number_columns = 2):
    """
    Plot multiple tendency for each feature in the dataframe
    
    Args
        df (dataframe): data. The index should be the timestamp and the dataframe must be sorted by the index
        number_columns (integer): number of columns, by default ONE column
        var_segment (string): name of the variable that segment the data to color the scatter plot

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # get list of features
    list_features = df.columns.tolist()

    # get name of each segment in a list
    unique_values_segments = df[var_segment].unique().tolist()
    unique_values_segments = list(filter(pd.notna, unique_values_segments)) # delete null values in segments
    unique_values_segments.sort()

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

        # get trend gray with scatter plot of differets segments
        fig_scatter_segment = px.scatter(df, x=df.index, y=feature, color = var_segment, opacity = 0.7)
        fig_trend = px.line(df, x = df.index, y = feature)
        fig_trend.data[0]['line']['color'] = '#D3D3D3'
        
        # add the individual plot scatter-trend into a subplot
        fig.add_trace(fig_trend.data[0],  row = row, col = column)
        for index_segment in range(len(unique_values_segments)): ### add scatter plot segments
            fig.add_trace(fig_scatter_segment.data[index_segment], row = row, col = column)
        

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



""""""""""""""""""""""""""""""""""""""""""""""""" FREQ SEGMENTATION """""""""""""""""""""""""""""""""""""""""""""""""
def plot_freq_segmentation(df, var_segment):
    """
    Given a segmentation in the data, plot the freq of each segment
    
    Args
        df (dataframe): input dataframe
        var_segment (string): variable in the input dataframe that indicate the segments in the data
    
    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ''' calculate dataframe with freq '''
    df_freq_segmentation = df[var_segment].value_counts()
    df_freq_segmentation = pd.DataFrame(df_freq_segmentation)
    df_freq_segmentation.reset_index(inplace = True)
    
    
    ''' plot barplot freq '''
    # create freq bar
    fig = px.histogram(df_freq_segmentation, x = var_segment, y = 'count', barmode='group')
    
    # add value each bar
    fig.update_traces(text = df_freq_segmentation['count'], textposition='outside')
    
    # update layout
    fig.update_layout(
        title_text=f'Freq of each segments for segmentation by {var_segment}',
        title_x=0.5,  # centrar título
        title_font=dict(size=20),
        yaxis=dict(title = 'freq')
    )

    return fig

"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""""""""""""" PARALLEL DISCRETE TARGET """""""""""""""""""""""""""""""""""""""""""""""""
def plot_parallel_continuous_discrete_target(df, list_features_target, var_segment_target_discrete):
    """
    Plot a parallel with features continous variables an target discrete target.
    
    Important the discrete target can be a string categorical (ex 'low', 'medium', 'high') and this function transform it into a integer categorical
    (ex. 1, 2, 3). This only works if the column categorical in pandas as internally defined the order in the string categories (ex: 'low' < 'medium' < 'high')
    (pandas dtype category)

    Args
        df (dataframe): dataframe with the data
        list_features_target (list): list with the features to plot in the parallel plot and also it has to have the target
        var_segment_target_discrete (string): define a string with the name of the target. name of the target as discrete variable

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # generate df to plot in parallel plot. in this kind of plot duplicated values are not soported and return an error
    df_parallel = df[list_features_target].drop_duplicates()

    # transform target_discrete string into integer. using internally definition of the variable in pandas
    df_parallel[var_segment_target_discrete] = df_parallel[var_segment_target_discrete].cat.codes

    # plot
    fig = px.parallel_coordinates(df_parallel, 
                                  color = var_segment_target_discrete,
                                  color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

    # change title
    fig.update_layout(
      title_text = "Parallel continuous features - discrete target",
        title_x = 0.5,
    title_font = dict(size = 28)
      )

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



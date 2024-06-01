import numpy as np
import pandas as pd
import statsmodels
import seaborn as sns
from itertools import combinations

import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots

# plotly
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


""""""""""""""""""""""""""""""""""""""""""""""""" AUXILIAR FUNCTION """""""""""""""""""""""""""""""""""""""""""""""""
# auxiliar function to map features in a pair (feature_x, feature_y)
def list_map_combinations_features(list_features, dim_combinations = 2):
    """
    Given a list of features of a dataframe, map all the combinations between each features. combinations without replace and (a,b) is the same (b,a)
    IN PREVIOUS CODES THERE ARE OTHER WAY TO MAP THE FEATURESS, ACUALLY THIS WAY IS BETTER

    Args:
        list_features (list): list of features that will generate the combinations
        dim_combinations (string): dimensions of combinations. default 2 -> generate a pair of features (feature_x, feature_y)

    Return
        list_tuple_combinations (list): list where each element is a tuple with the combination
    """
    # get all the possible combinations withtout repeteat
    todas_combinaciones = combinations(list_features, dim_combinations)
    
    # generate output
    list_tuple_combinations = []
    for comb in todas_combinaciones:
        list_tuple_combinations.append(comb)

    return list_tuple_combinations

# auxliar function to plot a dataframe as a plotly table
def plot_df_table_plotly(df_to_plotly):
    """
    Given a dataframe, transform into a plotly table
    Args
        df_to_plotly (dataframe): dataframe that will be transformed into plotly table

    Return
        table_fig (figure plotly): fig of plotly with the plot generated
    """
    table_fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_to_plotly.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df_to_plotly[col] for col in df_to_plotly.columns],
               fill_color='lavender',
               align='left'))
    ])
    return table_fig

"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""""""""""""" CROSSTAB FREQ FEATURES TARGET """""""""""""""""""""""""""""""""""""""""""""""""
def calculate_freq_data(df):
    """
    Given a categorical dataframe, calculate the freq of each category of each variable presents in the dataframe

    Args
        df (dataframe): input dataframe

    Return
        freq_df (dataframe): dataframe with the freq and percent
        freq_df_to_plotly (dataframe): dataframe with the freq and percent adapted to show in a plotly graph
    """
    variables_to_analyze = df.columns.tolist()

    # generate dataframe freq_df
    freq_df = pd.DataFrame()
    for i in variables_to_analyze:
        #obtener valores de cada una de las características
        A = df[i].value_counts()
        #crear base de datos con el total y porcentaje de cada caracteerística
        B = pd.DataFrame({ 'Freq (count)' : A, 'Freq (percent)': A.map(lambda x: x / A.sum())  })
        # crear multiindex
        B.index = pd.MultiIndex.from_product([[i], A.index.tolist()] )
        # unir a una sola base datos
        freq_df = pd.concat([freq_df, B], axis = 0)


    # round to 3 decimals
    freq_df['Freq (percent)'] = freq_df['Freq (percent)'].round(3)

    # order table by index - get the values ordered by percentile of each feauture
    freq_df = freq_df.sort_index()

    # transform df freq_df into a format to plotly
    freq_df_to_plotly = freq_df.reset_index()
    freq_df_to_plotly.loc[freq_df_to_plotly['level_0'].duplicated(), 'level_0'] = ''
    
    return freq_df, freq_df_to_plotly

def plot_df_table_plotly(df_to_plotly):
    """
    Given a dataframe, transform into a plotly table
    Args
        df_to_plotly (dataframe): dataframe that will be transformed into plotly table

    Return
        table_fig (figure plotly): fig of plotly with the plot generated
    """
    table_fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_to_plotly.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df_to_plotly[col] for col in df_to_plotly.columns],
               fill_color='lavender',
               align='left'))
    ])
    return table_fig

"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""




""""""""""""""""""""""""""""""""""""""""""""""""" FREQ between a pair of categorical variable CAT VS CAT (hist 2d) """""""""""""""""""""""""""""""""""""""""""""""""
##### inidividual plotS
def generate_crosstab_freq_between_2_features(df, feature_x, feature_y, ct_normalized = True):
    """
    Given a dataframe and 2 features x-axis and y-axis generate a cross table of frecuency between the intersection of this pair of categorical
    features

    Args:
        df (dataframe): input dataframe
        feature_x (string): feature to show in x-axis. this feature needs to be in the input dataframe
        feature_y (string): feature to show in y-axis. this feature needs to be in the input dataframe
        ct_normalized (bool): boolean that indicate if the freq of each feature are normalized or not
    """
    ### generte cross table ofr frecuency normalizaed between each feature and margins
    ct_margins = pd.crosstab(df[feature_x], df[feature_y], normalize = ct_normalized, margins = True)
    
    # round 3 decimals
    ct_margins = ct_margins.round(3)

    return ct_margins


def plot_heatmap_hist2d_individual_features(df_ct, name_table):
    """
    Plot heatmap using the input dataframe

    Args
        df_corr (dataframe): dataframe with crosstable to plot

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    # heatmap
    fig = px.imshow(df_ct, text_auto=True, aspect="auto")
    
    # change title
    fig.update_layout(
      title_text = name_table,
        title_x = 0.5,
    title_font = dict(size = 28)
      )
    
    return fig


#### subplots of each pair of features
def heatmap_hist2d_features_percentile(df, target, ct_normalized = True, number_columns = 1):
    """
    Given a dataframe with columns features + target. Genereate a heatmap/histogram 2d between each pair of features categorical (ej percentile)
    Detail: given a dataframe with features categorical, generate a crosstab of freq between 2 features and plot it in a heatmap
    
    Args
        df (dataframe): input dataframe with columns features and target
        target (string): target of the dataframe, column that will be delete to plot the relations between only features
        ct_normalized (bool): boolean that indicate if the freq of each feature are normalized or not
        number_columns (integer): number of columns. because heatmap could be bigger, plot it into 1 columns by default

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ################# generate a list of tuples of each pair of features to generate the cross table  #####################
    df_only_features = df.drop(columns = target) # delete target of the data
    list_pair_features = list_map_combinations_features(df_only_features) # generate list of pair features

    
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

        ## get cross table freq between feature_x and feature_y. with margins. It is possible to select between normalized values or not
        if ct_normalized:
            ct_freq_features = pd.crosstab(df[feature_x], df[feature_y], normalize=True, margins = True)
        else:
            ct_freq_features = pd.crosstab(df[feature_x], df[feature_y], normalize=False, margins = True)
            ct_freq_features = ct_freq_features.round(3)
        
        
        ## tranform cross table freq between pair of features into a heatmap
        fig_aux = px.imshow(ct_freq_features, text_auto=True, aspect="auto")
        
        
        # add heatmap to fig global
        fig.add_trace(fig_aux.data[0],
            row = row,
            col = column
        )

    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text =  f'[freq/cross table/hist 2d] betweeen pair of features',
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""




""""""""""""""""""""""""""""""""""""""""""""""""" UNIVARIATE ANALYSIS FEATURE(X) CATEGORICAL VS TARGET(Y) CONTINUOUS """""""""""""""""""""""""""""""""""""""""""""""""
# table statistics of target for each category in each feature
def descriptive_statistics_target_for_each_feature(df, target):
    """
    Calculate descriptive statistics of target for each feature categorical (and for each category in each feature)

    Args:
        df (dataframe): input dataframe
        target (string): string to target that will be calcuated its statistics

    Return:
        df_statistics_target (dataframe): dataframe with the statistics of the target
        df_statistics_target_to_plotly (dataframe): dataframe with the statistics of the target adapted to show in a plotly graph
    """
    
    ### list_features
    list_features = list(set(df.columns.tolist()) - set([target]))
    
    ###### generate descriptive statistics of the target for each percentil of each feature
    df_statistics_target = pd.DataFrame()
    for feature in list_features:
        #print(feature)
        
        # calculate statistic descriptive of target for a categories of a feature
        aux_statistics_target = df.groupby(feature)[target].describe()
        
        # set multiindex (feature, percentile_feature)
        aux_statistics_target.index = pd.MultiIndex.from_product([
            [feature], 
            aux_statistics_target.index.tolist()
        ] )
        
        # join in a unique dataframe
        df_statistics_target = pd.concat([df_statistics_target, aux_statistics_target], axis = 0)
    
    
    ##### round to 3 decimals
    df_statistics_target.round(2)
    
    
    #### tranform output dataframe into format to plotly
    df_statistics_target_to_plotly = df_statistics_target.reset_index()
    df_statistics_target_to_plotly.loc[df_statistics_target_to_plotly['level_0'].duplicated(), 'level_0'] = '' # delete duplicated rows
    
    return df_statistics_target, df_statistics_target_to_plotly



#### INDIVIDUAL PLOTS - HISTOGRAM AND BOXPLOT
def plot_individual_hist_target_categorical_features(df, var_categorical, var_continuous_hist):
    """
    Plot individual hist of a variable (target in this example) colored by different categories of a feature (categorical variable)
    Args
        df (dataframe): input dataframe
        var_categorical (string): name of the column in the input dataframe that indicate the differents categories in the data
        var_continuous_hist (string): name of the feature in the input dataframe that will plot its histogram

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # sort data before plot - to show de categories of categorical variable in order
    df_sorted = df.sort_values(by = [var_categorical], ascending = True)
    
    fig = px.histogram(df_sorted, x = var_continuous_hist, color = var_categorical, barmode='overlay', opacity=0.4)

    # update title
    fig.update_layout(
      title_text = f'Histogram: {var_continuous_hist} by percentil of {var_categorical}',
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )
    return fig

def plot_individual_boxplot_target_categorical_features(df, var_categorical, var_continuous_hist):
    """
    Plot individual boxplot of a variable (target in this example) colored by different categories of a feature (categorical variable)
    Args
        df (dataframe): input dataframe
        var_categorical (string): name of the column in the input dataframe that indicate the differents categories in the data
        var_continuous_hist (string): name of the feature in the input dataframe that will plot its histogram

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # sort data before plot - to show de categories of categorical variable in order
    df_sorted = df.sort_values(by = [var_categorical], ascending = True)
    
    fig = go.Figure()
    fig.add_trace(go.Box(x=df_sorted[var_categorical], y=df_sorted[var_continuous_hist]))

    # update title
    fig.update_layout(
      title_text = f'Histogram: {var_continuous_hist} by percentil of {var_categorical}',
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )
    return fig


###### GRUPAL PLOTS - SUBPLOTS - BOXPLOT
def plot_boxplots_target_categorical_features(df, var_continuous_hist, number_columns = 2):
    """
    Plot multiple boxplots of the target (continous variable) colored individually for each feature in the dataframe (each feature is categorical)
    Differents colors in the histogram according the segmentation in the data
    
    Args
        df (datafame): input dataframe
        var_continuous_hist (string): name of the feature in the input dataframe that will plot its histogram

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # get list features
    list_features_target = df.columns.tolist()
    list_features = list(set(list_features_target) - set([var_continuous_hist]))


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

        # sort data by variable to plot. show categories of categorical varialble in order
        df_sorted = df.sort_values(by = [feature], ascending = True)
        
        # add trace boxplot
        fig.add_trace(go.Box(x=df_sorted[feature], y=df_sorted[var_continuous_hist]),
                row = row,
                col = column)
        fig.update_yaxes(title_text = var_continuous_hist, row=row, col=column)
        
    # update layout
    fig.update_layout(height = number_rows * 250,
                      width = number_columns * 800,
                      title_text = "Boxplots of a target variable vs features categorized in percentil",
                      title_x = 0.5, # centrar titulo
                    title_font = dict(size = 28)
                     ) 
                     
    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""""""""""""" UNIVARIATE ANALYSIS FEATURE(X) CATEGORICAL VS TARGET(Y) CATEGORICAL """""""""""""""""""""""""""""""""""""""""""""""""
# table statistics of target for each category in each feature
def calculate_freq_target_each_features(df, target, ct_normalized = True):
    """
    Given a dataframe with discrete features and target discrete. Calculate the freq of the target in each features (each category of each feature)

    Args
        df (dataframe): input dataframe
        target (string): variable target. it needs to be present in the df
        ct_normalized (boolean): way to show the freq of the target. Percent or Value. if the value is true, show the value in percent normalized

    Return
        resume (dataframe): dataframe resume with the freq of each category of target in each feature
        resume_to_plotly (dataframe): the same output dataframe but with transformations to show into a plotly table
    """
    # get list of features
    list_features_target = df.columns.tolist() # calculate all variables in dataframe features+target
    list_features = list(set(list_features_target) - set([target]))

    # define kind to aggregation - count/percent
    if ct_normalized == True:
        aux = np.array(df[target].value_counts().tolist()).reshape(1,len(df[target].cat.categories.tolist()))
        resume = pd.DataFrame(aux / aux.sum() , columns = df[target].value_counts().index.tolist())
        resume.index =  pd.MultiIndex.from_product([ [target]  , ['Mean'] ] )
    
    else:
        aux = np.array(df[target].value_counts().tolist()).reshape(1,len(df[target].cat.categories.tolist()))
        resume = pd.DataFrame(aux , columns = df[target].value_counts().index.tolist())
        resume.index =  pd.MultiIndex.from_product([ [target]  , ['Mean'] ] )
    
    # create cross table
    for var in list_features:
        if ct_normalized == True:
            data_aux = pd.crosstab(df[var], df[target])
            data_aux = data_aux.div(data_aux.sum(1).astype(float), axis = 0)
            data_aux.index = pd.MultiIndex.from_product([ [var]  , df[var].cat.categories.tolist() ] )
            resume = pd.concat((resume,data_aux) , axis = 0, sort = False)
        
        else:
            data_aux = pd.crosstab(df[var], df[target])
            data_aux.index = pd.MultiIndex.from_product([ [var]  , df[var].cat.categories.tolist() ] )
            resume = pd.concat((resume,data_aux) , axis = 0, sort = False)

    # transform output into 2 decimals
    resume = resume.round(2)

    # order table by index - get the values ordered by percentile of each feauture
    resume = resume.sort_index()
    
    # transfrom table into a format to show in plotly
    resume_to_plotly = resume.reset_index()
    resume_to_plotly.loc[resume_to_plotly['level_0'].duplicated(), 'level_0'] = ''
    
    return resume, resume_to_plotly


# INDIVIDUAL PLOT - BARPLOT - freq categorical target vs categorical features
def crosstab_freq_target_1_feature(df, feature, target):
    """
    Calculate a cross tab of frecuency of target (categorical) given one categorical feature.
    The output are 2 dataframes, the first is the output of pd.crosstab() and the second one is the previous output transformed to plot in plotly

    Args:
        df (dataframe): input dataframe with feature and target categorical variables
        feauture (string): name categorical variable to compare target
        target (string): name categorical target

    Return
        ct_freq_target (dataframe): cross tab of frecuency of target given according a categorical feature
        ct_freq_target_reset_index (dataframe): previous dataframe with transformations to plot in plotly barplot
    """

    ##### calculate cross tab
    # calculate cross table freq
    ct_freq_target = pd.crosstab(index = df[feature], columns = df[target])

    
    ##### transform into cross table accepted to plotly
    # reset index  to plot
    ct_freq_target_reset_index = ct_freq_target.reset_index()
    
    # convert table into a format to plotly express
    ct_freq_target_reset_index = pd.melt(ct_freq_target_reset_index, id_vars = feature, value_name='freq_target')

    return ct_freq_target, ct_freq_target_reset_index


def plot_barplot_crosstab_individual(df_ct_freq_plotly):
    """
    Plot barplot using the input dataframe with the cross table to plot

    Args
        df_ct_freq_plotly (dataframe): dataframe with crosstable of freq of target given an categorical feature in format to show in plotly

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # given a input crosstable get feature and target
    feature_ = df_ct_freq_plotly.columns.tolist()[0]
    target_ = df_ct_freq_plotly.columns.tolist()[1]
    
    # barplot
    fig = px.bar(df_ct_freq_plotly, 
                 x = feature_, 
                 y='freq_target',
                 color = target_,
                 title='Crosstab Plot',
                 barmode='group' # Especifica que no quieres barras apiladas
                )
    
    # change title
    fig.update_layout(
      title_text = f'freq of categorical target:{feature_} given a categorical feature:{target_}',
        title_x = 0.5,
    title_font = dict(size = 28)
      )
    
    return fig


# MULTIPLE PLOTS - SUBPLTOS - BARPLOT - freq categorical target vs categorical features
def barplot_crosstab_freq_target_1_features(df, target, number_columns = 1):
    """
    Given a dataframe with columns features + target. Genereate a barplot of relations between each features and the freq of the target
    Detail: 
        Given a dataframe with features categorical, generate a crosstab of freq of target between feature and plot it in a barplot
        Calling a function to generate a cross table and then plot it with plotly
    
    Args
        df (dataframe): input dataframe with columns features and target
        target (string): target of the dataframe, column that will be delete to plot the relations between only features
        number_columns (integer): number of columns. because heatmap could be bigger, plot it into 1 columns by default

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ################# generate a list of tuples of each pair of features to generate the cross table  #####################
    list_features = list(set(df.columns.tolist()) - set([target]))

    
    ####################### plot #################################
    
    # calculate number of rows (considering the number of colums passed as args)
    if (len(list_features) % number_columns) != 0:
        number_rows = (len(list_features) // number_columns) + 1
    else:
        number_rows = (len(list_features) // number_columns)

    # create fig to plot
    fig = make_subplots(rows=number_rows, cols=number_columns, 
                        subplot_titles = tuple([str(tupla) for tupla in list_features]) ### title of each subplots
                       )

    ########## for each tuple of features to plot:
    for index_feature, feature in enumerate(list_features):
        
        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

        
        ## get cross table freq of target vs 1 categorical features - call the INDIVIDUAL FUNCTION TO GENERATE CROSS TABLE
        # the output are 2 dataframes, the first is the output of pd.crosstab() and the second one is the previous output transformed to plot in plotly
        _, ct_freq_target_plotly = crosstab_freq_target_1_feature(df = df, 
                                                         feature = feature, 
                                                         target = target)
        
        ## tranform cross table freq target vs one categorical feature into a barplot
        fig_barplot_aux = px.bar(ct_freq_target_plotly, 
                     x = feature, 
                     y='freq_target',
                     color = target,
                     barmode='group'
                    )
        
        # add barplot to fig global
        for index_plot in range(len(fig_barplot_aux.data)):
            fig.add_trace(fig_barplot_aux.data[index_plot],
                row = row,
                col = column
            )

    # adjust the shape
    fig.update_layout(
        height = 550 * number_rows,  # largo
        width = 1850 * number_columns,  # ancho
        title_text =  f'freq of target:{target} vs each categorical feature individual',
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""




""""""""""""""""""""""""""""""""""""""""""""""""" BIVARIATE ANALYSIS FEATURE(X) CATEGORICAL VS TARGET(Y) CONTINOUS """""""""""""""""""""""""""""""""""""""""""""""""
# heatmap feature1 & feature 2 vs target -- INDIVIDUAL PLOT
def crosstab_agg_target_2_features(df, feature_index, feature_column, target, agg_target = 'mean'):
    """
    generate crosstable of aggregation of the target between 2 features (feature index and feature column)

    Args
        ------
    
    Return
        df_ct (dataframe): dataframe with the cross table
    """
    # calculate cross_table
    df_ct= pd.crosstab(index = df[feature_index], 
                          columns = df[feature_column], 
                          values = df[target], 
                              aggfunc = agg_target) # agg fuction without list
    
    # round 3 decimals
    df_ct = df_ct.round(3)
    
    return df_ct

def plot_heatmap_crosstable_individual(df_ct, name_table):
    """
    Plot heatmap using the input dataframe with the cross table to plot

    Args
        df_corr (dataframe): dataframe with crosstable to plot

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    # heatmap
    fig = px.imshow(df_ct, text_auto=True, aspect="auto")
    
    # change title
    fig.update_layout(
      title_text = name_table,
        title_x = 0.5,
    title_font = dict(size = 28)
      )
    
    return fig


# # heatmap feature1 & feature 2 vs target - MULTIPLE PLOTS - SUBPLOTS
def heatmap_crosstab_aggregation_target_2_features(df, target, agg_target = 'mean', number_columns = 1):
    """
    Given a dataframe with columns features + target. Genereate a heatmap of relations between 2 features and one aggregation function of the target
    Detail: 
        Given a dataframe with features categorical, generate a crosstab of aggregation of target between 2 features and plot it in a heatmap
        Calling a individual function to generate a cross table
    
    Args
        df (dataframe): input dataframe with columns features and target
        target (string): target of the dataframe, column that will be delete to plot the relations between only features
        agg_target (string): aggregation function of the target
        number_columns (integer): number of columns. because heatmap could be bigger, plot it into 1 columns by default

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ################# generate a list of tuples of each pair of features to generate the cross table  #####################
    df_only_features = df.drop(columns = target) # delete target of the data
    list_pair_features = list_map_combinations_features(df_only_features.columns.tolist(), 2)

    
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

        
        ## get cross table agg function of target, between 2 features - call the INDIVIDUAL FUNCTION TO GENERATE CROSS TABLE 
        ct_agg_target = crosstab_agg_target_2_features(df = df, 
                                                       feature_index = feature_x, 
                                                       feature_column = feature_y, 
                                                       target = target, 
                                                       agg_target = agg_target)
        
        ## tranform cross table freq between pair of features into a heatmap
        fig_aux = px.imshow(ct_agg_target, text_auto=True, aspect="auto")
        
        # add heatmap to fig global
        fig.add_trace(fig_aux.data[0],
            row = row,
            col = column
        )

    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text =  f'Cross table betweeen "pair of features" with "{agg_target} of the {target}"',
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig


#-------

# heatmap feature1 & feature 2 & feature 3 vs target - INDIVIDUAL PLOT
def crosstab_agg_target_3_features(df, feature_index1, feature_index2, feature_column, target, agg_target = 'mean'):
    """
    generate crosstable of aggregation of the target between 3 features (2 features indexes and 1 feature column)

    Args
        ------
    
    Return
        ct_2index (dataframe): dataframe with the cross table with multiindex of 2 features
        ct_2index_reset_index (dataframe): dataframe previous with reset of multiindex
    """
    
    ###### calculate cross table 2 index
    ct_2index = pd.crosstab(index = [df[feature_index1], df[feature_index2]], 
                          columns = df[feature_column], 
                          values = df[target], 
                            aggfunc = agg_target)

    
    ##### transform into plotly heatmap format - only one index
    # generate a dataframe with multiindex into only one index (join intro string the multi index into one index)
    
    # 0. reset index
    ct_2index_reset_index = ct_2index.reset_index()
    
    # 1. transform each categorical column into string.
    ct_2index_reset_index[feature_index1] = ct_2index_reset_index[feature_index1].astype(str)
    ct_2index_reset_index[feature_index2] = ct_2index_reset_index[feature_index2].astype(str)
    
    # 2. add name of the column (beacase actually only show q1, q2, etc)
    ct_2index_reset_index[feature_index1] = feature_index1 + '|' + ct_2index_reset_index[feature_index1]
    ct_2index_reset_index[feature_index2] = feature_index2 + '|' + ct_2index_reset_index[feature_index2]
    
    # 3. combine content of 2 columnas and delete old ones
    ct_2index_reset_index['index'] = ct_2index_reset_index[feature_index1] + '__&&__' + ct_2index_reset_index[feature_index2]
    ct_2index_reset_index.drop(columns = [feature_index1, feature_index2], inplace = True)
    ct_2index_reset_index.set_index('index', inplace = True)

    ##### round 3 decimals
    ct_2index = ct_2index.round(3)
    ct_2index_reset_index = ct_2index_reset_index.round(3)
    
    return ct_2index, ct_2index_reset_index


# heatmap feature1 & feature 2 & feature 3 vs target - MULTIPLE PLOT
def heatmap_crosstab_aggregation_target_3_features(df, target, agg_target = 'mean', number_columns = 1):
    """
    Given a dataframe with columns features + target. Genereate a heatmap of relations between 3 features and one aggregation function of the target
    Detail: 
        Given a dataframe with features categorical, generate a crosstab of aggregation of target between 3 features and plot it in a heatmap
        Calling a individual function to generate a cross table
    
    Args
        df (dataframe): input dataframe with columns features and target
        target (string): target of the dataframe, column that will be delete to plot the relations between only features
        agg_target (string): aggregation function of the target
        number_columns (integer): number of columns. because heatmap could be bigger, plot it into 1 columns by default

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ################# generate a list of tuples of each pair of features to generate the cross table  #####################
    df_only_features = df.drop(columns = target) # delete target of the data
    list_triple_features = list_map_combinations_features(df_only_features.columns.tolist(), 3)

    
    ####################### plot #################################
    
    # calculate number of rows (considering the number of colums passed as args)
    if (len(list_triple_features) % number_columns) != 0:
        number_rows = (len(list_triple_features) // number_columns) + 1
    else:
        number_rows = (len(list_triple_features) // number_columns)

    # create fig to plot
    fig = make_subplots(rows=number_rows, cols=number_columns, 
                        subplot_titles = tuple([str(tupla) for tupla in list_triple_features]) ### title of each subplots
                       )

    ########## for each tuple of features to plot:
    for index_feature, (feature_index1_ct, feature_index2_ct, feature_column_ct) in enumerate(list_triple_features):
        
        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

        
        ## get cross table agg function of target, between 3 features - call the INDIVIDUAL FUNCTION TO GENERATE CROSS TABLE 
        _, ct_agg_target_plotly = crosstab_agg_target_3_features(df = df, 
                               feature_index1= feature_index1_ct, 
                               feature_index2 = feature_index2_ct, 
                               feature_column = feature_column_ct, 
                               target = target, 
                               agg_target = agg_target
                              )
        
        ## tranform cross table freq between pair of features into a heatmap
        fig_aux = px.imshow(ct_agg_target_plotly, text_auto=True, aspect="auto")
        
        # add heatmap to fig global
        fig.add_trace(fig_aux.data[0],
            row = row,
            col = column
        )

    # adjust the shape
    fig.update_layout(
        height = 450 * number_rows,  # largo
        width = 1850 * number_columns,  # ancho
        title_text =  f'Cross table betweeen "pair of features" with "{agg_target} of the {target}"',
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""




""""""""""""""""""""""""""""""""""""""""""""""""" BIVARIATE ANALYSIS FEATURE(X) CATEGORICAL VS TARGET(Y) CATEGORICAL """""""""""""""""""""""""""""""""""""""""""""""""
#### barplot feature1 & feature2 vs target - INDIVIDUAL PLOT
def crosstab_freq_target_2_feature(df, feature_index1, feature_index2, target):
    """
    Calculate a cross tab of frecuency of target (categorical) given two categorical feature
    The output are 2 dataframes, the first is the output of pd.crosstab() and the second one is the previous output transformed to plot in plotly

    Args:
        df (dataframe): input dataframe with feature and target categorical variables
        feauture_index1 (string): name categorical feature 1 to compare target
        feature_index2 (string): name categorical feature 2 to compare target
        target (string): name categorical target

    Return
        ct_freq_target_2features (dataframe): cross tab of frecuency of target given according a categorical feature
        ct_freq_target_2features_plotly (dataframe): previous dataframe with transformations to plot in plotly barplot
    """

    ######################## calculate cross tab ########################
    # calculate cross table freq
    ct_freq_target_2features = pd.crosstab(index = [df[feature_index1], df[feature_index2]], 
                                           columns = df[target]
                                          )
    
    
    ######################## transform into plotly barplot format - only one index ########################
    # generate a dataframe with multiindex into only one index (join intro string the multi index into one index)
    # 0. reset index
    ct_freq_target_2features_reset_index = ct_freq_target_2features.reset_index()
    
    # 1. transform each categorical column into string.
    ct_freq_target_2features_reset_index[feature_index1] = ct_freq_target_2features_reset_index[feature_index1].astype(str)
    ct_freq_target_2features_reset_index[feature_index2] = ct_freq_target_2features_reset_index[feature_index2].astype(str)
    
    # 2. add name of the column (beacase actually only show q1, q2, etc)
    ct_freq_target_2features_reset_index[feature_index1] = feature_index1 + '|' + ct_freq_target_2features_reset_index[feature_index1]
    ct_freq_target_2features_reset_index[feature_index2] = feature_index2 + '|' + ct_freq_target_2features_reset_index[feature_index2]
    
    # 3. combine content of 2 columnas and delete old ones
    ct_freq_target_2features_reset_index['index'] = ct_freq_target_2features_reset_index[feature_index1] + '__&&__' + ct_freq_target_2features_reset_index[feature_index2]
    ct_freq_target_2features_reset_index.drop(columns = [feature_index1, feature_index2], inplace = True)
    ct_freq_target_2features_reset_index.set_index('index', inplace = True)


    #### finally transformation to plot plotly
    # 0. reset index  to plot
    ct_freq_target_2features_plotly = ct_freq_target_2features_reset_index.reset_index()
    
    # 1. convert table into a format to plotly express
    ct_freq_target_2features_plotly = pd.melt(ct_freq_target_2features_plotly, id_vars='index', value_name='freq_target')

    return ct_freq_target_2features, ct_freq_target_2features_plotly

def plot_barplot_crosstab_individual(df_ct_freq_plotly):
    """
    Plot barplot using the input dataframe with the cross table to plot

    Args
        df_ct_freq_plotly (dataframe): dataframe with crosstable of freq of target given an categorical feature in format to show in plotly

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # given a input crosstable get feature and target
    feature_ = df_ct_freq_plotly.columns.tolist()[0]
    target_ = df_ct_freq_plotly.columns.tolist()[1]
    
    # barplot
    fig = px.bar(df_ct_freq_plotly, 
                 x = feature_, 
                 y='freq_target',
                 color = target_,
                 title='Crosstab Plot',
                 barmode='group' # Especifica que no quieres barras apiladas
                )
    
    # change title
    fig.update_layout(
      title_text = f'freq of categorical target:{feature_} given a categorical feature:{target_}',
        title_x = 0.5,
    title_font = dict(size = 28)
      )
    
    return fig


#### barplot feature1 & feature2 vs target - MULTIPLE PLOT - SUBPLOTS
def barplot_crosstab_freq_target_2_features(df, target, number_columns = 1):
    """
    Given a dataframe with columns features + target. Genereate a barplot of relations between each a pair of features (feature_x, feature_y) 
    and the freq of the target
    
    Detail: 
        Given a dataframe with all variables categorical, generate a crosstab of freq of target between the pair of features and plot it in a barplot
        Calling a function to generate a cross table and then plot it with plotly
    
    Args
        df (dataframe): input dataframe with columns features and target
        target (string): target of the dataframe, column that will be delete to plot the relations between only features
        number_columns (integer): number of columns. because heatmap could be bigger, plot it into 1 columns by default

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ################# generate a list of tuples of each pair of features to generate the cross table  #####################
    df_only_features = df.drop(columns = target) # delete target of the data
    list_pair_features = list_map_combinations_features(df_only_features.columns.tolist(), 2)
    #print(len(list_pair_features))

    
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
    for index_feature, (feature_x1, feature_x2) in enumerate(list_pair_features):
        
        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

        
        ## get cross table freq of target vs 2 categorical features - call the INDIVIDUAL FUNCTION TO GENERATE CROSS TABLE
        # the output are 2 dataframes, the first is the output of pd.crosstab() and the second one is the previous output transformed to plot in plotly
        #print(f'debugging - running plot of pair of features: {feature_x1} vs {feature_x2}')
        _, ct_freq_target_plotly = crosstab_freq_target_2_feature(df = df, 
                                                                  feature_index1 = feature_x1, 
                                                                  feature_index2 = feature_x2,
                                                                  target = target)
        
        # given a input crosstable in plotly format get the inputs to plot in ploty
        feature_plot_plotly = ct_freq_target_plotly.columns.tolist()[0]
        target_plot_plotly = ct_freq_target_plotly.columns.tolist()[1]

        
        ## tranform cross table freq target vs one categorical feature into a barplot
        fig_barplot_aux = px.bar(ct_freq_target_plotly, 
                     x = feature_plot_plotly, 
                     y='freq_target',
                     color = target_plot_plotly,
                     barmode='group'
                    )
        
        # add barplot to fig global
        for index_plot in range(len(fig_barplot_aux.data)):
            fig.add_trace(fig_barplot_aux.data[index_plot],
                row = row,
                col = column
            )

    # adjust the shape
    fig.update_layout(
        height = 550 * number_rows,  # largo
        width = 1850 * number_columns,  # ancho
        title_text =  f'freq of target:{target} vs each categorical feature individual',
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""




""""""""""""""""""""""""""""""""""""""""""""""""" PARALLEL """""""""""""""""""""""""""""""""""""""""""""""""
def plot_parallel_discrete_variables(df_percentile, list_features_target_to_plot, target):
    """
    Plot a parallel with features discretes variables an target discrete
    
    Important the discrete variables can be a string categorical (ex 'low', 'medium', 'high').
    
    But in the parallel plot, it needs to be colored according the target and it needs to be a numerical category. This function transform it into 
    a integer categorical (ex. 1, 2, 3). This only works if the column categorical in pandas as internally defined the order in the string categories 
    (ex: 'low' < 'medium' < 'high') (pandas dtype category)

    Args
        df (dataframe): dataframe with the data
        list_features_target_to_plot (list): list with the features to plot in the parallel plot and also it has to have the target
        target_discrete (string): in addition it is necesary define a string with the name of the target

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    
    # generate df to plot in parallel plot. in this kind of plot duplicated values are not soported and return an error
    df_percentile_parallel = df_percentile[list_features_target_to_plot].drop_duplicates()

    # transform target_discrete string into integer. using internally definition of the variable in pandas.
    # this is neccesary to color the parallel according the values of the target
    df_percentile_parallel[target] = df_percentile_parallel[target].cat.codes

    # plot
    fig = px.parallel_categories(df_percentile_parallel, 
                                 color = target, 
                                 color_continuous_scale=px.colors.sequential.Inferno)

    # change title
    fig.update_layout(
      title_text = "Parallel discrete variables",
        title_x = 0.5,
    title_font = dict(size = 28)
      )

    return fig
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""""""""""""" WOE / IV """""""""""""""""""""""""""""""""""""""""""""""""
def calculate_woe_iv(df, feature_woe, target):
    """
    Calculate WOE and IV for a categories of a feature

    Args:
        df (dataframe): input dataframe
        feature_woe (string): feature to calculate woe of its categories. feature_woe needs to be in the input dataframe
        target (string): target. target needs to be in the input dataframe

    Return
        table_woe_iv (dataframe): dataframe with the values of woe and iv of each categorie of the feature
        value_iv (float): value of IV for the feature
    """
    # get unique values of the target and get the positive categorie and negative categorie of the target. 
    # First cat -> positive / second cat -> negative
    list_cat_target = df[target].cat.categories.tolist()
    target_cat_positive = list_cat_target[0]
    target_cat_negative = list_cat_target[1]


    ################################# WOE #################################
    # Crear una tabla de resumen con el conteo de eventos positivos y negativos por categoría
    table_woe_iv = pd.DataFrame({feature_woe: df.loc[:, feature_woe], 'Target': df.loc[:, target]})
    table_woe_iv = table_woe_iv.groupby(feature_woe)['Target'].value_counts().unstack().reset_index()
    table_woe_iv.rename(columns={0: target_cat_negative, 1: target_cat_positive}, inplace=True)

    # Calcular la proporción de eventos positivos y negativos
    table_woe_iv[f'{target_cat_positive}_Rate'] = table_woe_iv[target_cat_positive] / table_woe_iv[target_cat_positive].sum()
    table_woe_iv[f'{target_cat_negative}_Rate'] = table_woe_iv[target_cat_negative] / table_woe_iv[target_cat_negative].sum()
    
    # Calcular el WOE (Weight of Evidence)
    table_woe_iv['WOE'] = np.log(table_woe_iv[f'{target_cat_positive}_Rate'] / table_woe_iv[f'{target_cat_negative}_Rate'])
    

    ################################# IV #################################
    # Calcular la contribución del WOE al IV
    table_woe_iv['IV_category'] = (table_woe_iv[f'{target_cat_positive}_Rate'] - table_woe_iv[f'{target_cat_negative}_Rate']) * table_woe_iv['WOE']


    # Calcular el IV (Information Value)
    value_iv = table_woe_iv['IV_category'].sum()

    return table_woe_iv, value_iv


def plot_hist_woe_iv(table_woe_iv, value_iv):
    """
    Plot a hist/freq of each categories of the feature and also plot the woe of each categorie.
    Also, show in the title of the plot the IV of the feature
  
    Args:
        table_woe_iv (dataframe): dataframe with the values of woe and iv of each categorie of the feature
        value_iv (float): value of IV for the feature

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    """ GENERAR DATAFRAME CON LOS VALORES A PLOTEAR """
    # given the structure of the table table_woe_iv, get the column name of categories positives and negatives of the target
    list_cat_feature = table_woe_iv[table_woe_iv.columns.tolist()[0]].cat.categories.tolist()
    positive_cat_target = table_woe_iv.columns.tolist()[1]
    negative_cat_target = table_woe_iv.columns.tolist()[2]
    
    
    #calcular frecuencia de cada categoría 
    df_to_plot = pd.DataFrame(columns = ['freq', 'WOE'])
    df_to_plot['freq'] = table_woe_iv[negative_cat_target] + table_woe_iv[positive_cat_target]
    df_to_plot['WOE'] = table_woe_iv['WOE']
    df_to_plot.index = list_cat_feature

    """ GRAFICAR """
    # Crear la figura
    fig = go.Figure()

    # Agregar el gráfico de barras
    fig.add_trace(go.Bar(
      x=df_to_plot.index,
      y=df_to_plot['freq'],
      name='Freq',
      marker_color='steelblue'
    ))

    # Agregar el gráfico de línea en el eje derecho
    fig.add_trace(go.Scatter(
      x=df_to_plot.index,
      y=df_to_plot['WOE'],
      name='WOE',
      yaxis='y2',
      line=dict(color='firebrick', width=2)
    ))
    
    # Configurar el diseño del gráfico
    fig.update_layout(
      title={
          'text': f'IV VALUE: {round(value_iv, 3)}',
          'x': 0.5,
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 24}
      },
      xaxis=dict(title='Category'),
      yaxis=dict(title='Freq'),
      yaxis2=dict(title='WOE', overlaying='y', side='right'),
      legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0.5)')
    )
    
    return fig

def plot_hist_woe_iv(table_woe_iv, value_iv):
    """
    Plot a hist/freq of each categories of the feature and also plot the woe of each categorie.
    Also, show in the title of the plot the IV of the feature
  
    Args:
        table_woe_iv (dataframe): dataframe with the values of woe and iv of each categorie of the feature
        value_iv (float): value of IV for the feature

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    """ GENERAR DATAFRAME CON LOS VALORES A PLOTEAR """
    # given the structure of the table table_woe_iv, get the column name of categories positives and negatives of the target
    list_cat_feature = table_woe_iv[table_woe_iv.columns.tolist()[0]].cat.categories.tolist()
    positive_cat_target = table_woe_iv.columns.tolist()[1]
    negative_cat_target = table_woe_iv.columns.tolist()[2]
    
    
    #calcular frecuencia de cada categoría 
    df_to_plot = pd.DataFrame(columns = ['freq', 'WOE'])
    df_to_plot['freq'] = table_woe_iv[negative_cat_target] + table_woe_iv[positive_cat_target]
    df_to_plot['WOE'] = table_woe_iv['WOE']
    df_to_plot.index = list_cat_feature

    """ GRAFICAR """
    # Crear la figura
    fig = go.Figure()

    # Agregar el gráfico de barras
    fig.add_trace(go.Bar(
      x=df_to_plot.index,
      y=df_to_plot['freq'],
      name='Freq',
      marker_color='steelblue'
    ))

    # Agregar el gráfico de línea en el eje derecho
    fig.add_trace(go.Scatter(
      x=df_to_plot.index,
      y=df_to_plot['WOE'],
      name='WOE',
      yaxis='y2',
      line=dict(color='firebrick', width=2)
    ))
    
    # Configurar el diseño del gráfico
    fig.update_layout(
      title={
          'text': f'IV VALUE: {round(value_iv, 3)}',
          'x': 0.5,
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 24}
      },
      xaxis=dict(title='Category'),
      yaxis=dict(title='Freq'),
      yaxis2=dict(title='WOE', overlaying='y', side='right'),
      legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0.5)')
    )
    
    return fig
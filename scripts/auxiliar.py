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


""""""""""""""""""""""""""""""""""""""""""""""""" SEGMENTATION DATA CUSTOM """""""""""""""""""""""""""""""""""""""""""""""""
def custom_segmentation(df, var_segment, intervals_segments, labels_segments):
    """
    Given a dataframe, generate a new column with a categorical values that divide the data in differents segments. 
    Segment the data by a certain variable with a custom segmentation
    
    Args
        df (dataframe): dataframe input
        var_segment (string): variable feature/target used to segment the data
        intervals_segments (list of numbers): list with the thresholds used to segment the data
        labels_segments (list of strings): list with the names of the differents segments generated. Shape: len(intervals_segments) - 1

    Return
        df(dataframe): the input dataframe with a new column with the segment
    """

    # apply pd.cut to generate intervals
    df[f'{var_segment}_segments'] = pd.cut(df[var_segment], 
                                           bins = intervals_segments, 
                                           labels = labels_segments, 
                                           include_lowest = True
                                          )

    # order data by the custom segmentation - to generate plots it is neccesary to sort the data
    # if the plot show a temporal relation like trends plots, it is necessary sort the data by index
    df = df.sort_values(by = [var_segment])
    
    return df
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""""""""""""" SEGMENTATION DATA PERCENTILES """""""""""""""""""""""""""""""""""""""""""""""""
def generate_labels_percentile_segmentation(df, var_segment, list_percentile, list_labels_percentile_base):
    """
    Given a dataframe and a feature to segment in percentiles, calculate the labels of the segmentation
    
    Choices of labels:
        labels_percentile: ['q1', 'q2', 'q3', 'q4']
        labels_values: ['(0.15-1.2)', '(1.2-1.8)', '(1.8-2.65)', '(2.65-5.0)']
        labels_percentile_values: ['q1 - (0.15-1.2)', 'q2 - (1.2-1.8)', 'q3 - (1.8-2.65)', 'q4 - (2.65-5.0)']
        
    Args
        df (dataframe): dataframe input
        var_segment (string): variable feature/target used to segment the data
        list_percentile (list): list of floats with the percentiles to divide the data
        list_labels_percentile_base (list): list of strings with the base labels of percentiles to divide the data 

    Return
        list_labels_percentile_base, list_labels_values_range, list_labels_percentile_values_range (lists). list of the 3 types of labels generated
    """

    # get values of each percentile
    list_percentile_values = [df[var_segment].quantile(x).round(2) for x in list_percentile]
    
    # generate a list of string with the start value and end value of each interval
    list_percentile_start_end = [] 
    for index in range(len(list_percentile_values)-1): 
        start_value = list_percentile_values[index]
        end_value = list_percentile_values[index+1]
        string_start_end = f'{start_value}-{end_value}'
        list_percentile_start_end.append(string_start_end)
    
    # output final v0 - base
    #list_labels_percentile_base
    
    # output final v1 - only values start end
    list_labels_values_range = []
    for index in range(len(list_labels_percentile_base)):
        string_output = f'({list_percentile_start_end[index]})'
        list_labels_values_range.append(string_output)
    
    # output final v2 - percentile and values start end
    list_labels_percentile_values_range = []
    for index in range(len(list_labels_percentile_base)):
        string_output = f'{list_labels_percentile_base[index]} - ({list_percentile_start_end[index]})'
        list_labels_percentile_values_range.append(string_output)
    
    return list_labels_percentile_base, list_labels_values_range, list_labels_percentile_values_range



def percentile_segmentation(df, var_segment, type_percentile):
    """
    Given a dataframe, generate a new column with a categorical values that divide the data in differents segments. 
    Segment the data by a certain variable with a percentile segmentation. the segmentation could be by quartiles, quintiles, deciles
    
    Args
        df (dataframe): dataframe input that will be modified
        var_segment (string): variable feature/target used to segment the data
        type_percentile(string): type of percentile segmentation
    
    Return
        df(dataframe): the input dataframe with a new column with the segment

    TODO: THE LABELS GERATED AND USED ARE ONLY ['q1 - (0.15-1.2)', 'q2 - (1.2-1.8)', 'q3 - (1.8-2.65)', 'q4 - (2.65-5.0)']
    ADD A ARGS TO SELECT THE KIND OF LABELS
    """

    # validate input - TODO: create a decent unit test
    choices_segmentation = ['quartile', 'quintile', 'decile']
    if type_percentile not in choices_segmentation:
        print('error in choices of segmentation')
        print(f'Possibles choices: {choices_segmentation}')
        return 0

    # quartile
    if type_percentile == 'quartile':
        quartile = [0, 0.25, 0.5, 0.75, 1]
        labels_quartile_base = ['q1', 'q2', 'q3', 'q4']
        _, _,  labels_quartile = generate_labels_percentile_segmentation(df, var_segment, quartile, labels_quartile_base)
        df[f'quartile_{var_segment}'] = pd.qcut(df[var_segment], q = quartile, labels = labels_quartile)
    
    # quintile
    if type_percentile == 'quintile':
        quintile = [0, 0.2, 0.4, 0.6, 0.8, 1]
        labels_quintile_base = ['q1', 'q2', 'q3', 'q4', 'q5']
        _, _,  labels_quintile = generate_labels_percentile_segmentation(df, var_segment, quintile, labels_quintile_base)
        df[f'quintile_{var_segment}'] = pd.qcut(df[var_segment], q = quintile, labels = labels_quintile)


    # decile
    if type_percentile == 'decile':
        decile = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        labels_decile_base = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10']
        _, _,  labels_decile = generate_labels_percentile_segmentation(df, var_segment, decile, labels_decile_base)
        df[f'decile_{var_segment}'] = pd.qcut(df[var_segment], q = decile, labels = labels_decile)

    return df
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



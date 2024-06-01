import datetime
import pandas as pd
import os
from plotly.subplots import make_subplots
import plotly.io as pio
import json

### import scripts with codes to do eda
from scripts import auxiliar as aux
from scripts import ydata_profiling as dp
from scripts import univariate_analysis as uv
from scripts import bivariate_analysis as bv
from scripts import segmentation_analysis as se
from scripts import categorical_analysis as ca

# auxiliar function for dataset example
def transform_strings_to_save(var_string):
    """ Replace characters that can be saved in windows """
    var_string= var_string.replace('/', '_') # replace element bad name windows
    var_string= var_string.replace('**', '_') # replace element bad name windows
    return var_string


def do_reports_ydata_profiling():
    # read params
    param_minimal = config['ydata_profiling']['minimal']

    # generate report
    print(f'ydata-profiling... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    dp.generate_report_ydata_profiling(df = data, 
                                       minimal = param_minimal, 
                                       id_report = id_report)
    



#if __name__ == "__main__":


""" read json config """
path_json = 'config.json'
with open(path_json, 'r') as archivo_json:
    config = json.load(archivo_json)


""" read data """
name_data_pkl = config['config_report']['name_data_pkl']
path_data_pkl = 'data/' + name_data_pkl
data = pd.read_pickle(path_data_pkl)

""" read global params and do global actions """
# define id report
name_report = config['config_report']['name_report']
datetime_report = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
id_report = name_report + '-' + datetime_report

# define number of columns in the plots
param_number_columns = config['config_report']['number_columns']

# read params feature target
param_target = config['config_report']['target']

# read params list feautures
param_list_features = config['config_report']['list_features']


""" Create folders neccesary to save results """
# create folders to save each kind of reports
os.makedirs('output_eda/' + id_report)
os.makedirs('output_eda/' + id_report + '/ydata_profiling')
os.makedirs('output_eda/' + id_report + '/univariate_analysis')
os.makedirs('output_eda/' + id_report + '/bivariate_analysis')
os.makedirs('output_eda/' + id_report + '/segmentation_analysis')
os.makedirs('output_eda/' + id_report + '/categorical_analysis')

# create folder to save univariate analysis - trends
os.makedirs('output_eda/' + id_report + '/univariate_analysis/trend') 
os.makedirs('output_eda/' + id_report + '/univariate_analysis/trend_zoom')


# create folder to save bivariate_analysis - scatter plots
os.makedirs('output_eda/' + id_report + '/bivariate_analysis/scatter-features-target')
os.makedirs('output_eda/' + id_report + '/bivariate_analysis/scatter-features-features')


# create folder to save segmentation_analysis - scatter plots
os.makedirs('output_eda/' + id_report + '/segmentation_analysis/scatter-features-target')
os.makedirs('output_eda/' + id_report + '/segmentation_analysis/scatter-features-features')



""" Define reports to show """
### define reports to show
show_ydata_profiling = config['reports_to_show']['ydata_profiling']
show_univariate_analysis = config['reports_to_show']['univariate_analysis']
show_bivariate_analysis = config['reports_to_show']['bivariate_analysis']
show_segmentation_analysis = config['reports_to_show']['segmentation_analysis']
show_categorical_analysis = config['reports_to_show']['categotical_analysis']


print('--- repots to show ---')
print('show_ydata_profiling: ', show_ydata_profiling)
print('show_univariate_analysis: ', show_univariate_analysis)
print('show_bivariate_analysis: ', show_bivariate_analysis)
print('show_segmentation_analysis: ', show_segmentation_analysis)
print('show_categorical_analysis:', show_categorical_analysis)
print('--- --- --- --- --- ---')



if show_ydata_profiling:
    print("\n--------------- YDATA PROFILING ---------------")
    # read params
    param_minimal = config['ydata_profiling']['minimal']

    # generate report
    print(f'ydata-profiling... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    dp.generate_report_ydata_profiling(df = data, 
                                       minimal = param_minimal, 
                                       id_report = id_report)



if show_univariate_analysis:
    print("\n--------------- UNIVARIATE ANALYSIS ---------------")
    
    """ PARAMS """
    # read params zoom tendency
    param_zoom_start_date = config['univariate_analysis']['zoom_tendency']['start_date']
    param_zoom_end_date = config['univariate_analysis']['zoom_tendency']['end_date']
    
    # read params smooth data
    param_smooth_ma_window = config['univariate_analysis']['smooth_data']['moving_average']['window']
    param_smooth_wma_weights = config['univariate_analysis']['smooth_data']['weighted_moving_average']['weights']
    param_smooth_ema_aplha = config['univariate_analysis']['smooth_data']['exponential_moving_average']['alpha']
    
    # read params acf/pacf
    param_lags = config['univariate_analysis']['acf_pacf']['lags']


    """ PLOTS """
    ################### fig histogram all features ###################
    print(f'statistics... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_statistics = uv.generate_descriptive_statistics(df = data)
    fig_statistics.write_html(f"output_eda/{id_report}/univariate_analysis/statistics.html")

    
    ################### fig histogram all features ###################
    print(f'histogram... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_hist_all = uv.plot_multiple_hist(df = data, number_columns = param_number_columns)
    fig_hist_all.write_html(f"output_eda/{id_report}/univariate_analysis/histograms.html")

    fig_hist_kde_all = uv.plot_kde_hist(df = data, number_columns = param_number_columns)
    fig_hist_kde_all.savefig(f"output_eda/{id_report}/univariate_analysis/histograms_kde.png", dpi = 300)


    ################### zoom data to tendency plots (trend & moving averavge) - zoom to reduce cost to plot ###################
    data_zoom = data.loc[param_zoom_start_date:param_zoom_end_date]

    
    # ################### data zoom - trend ###################
    print(f'trend data zoom... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')  
    # fig tendency all features in subplots
    fig_tendency_all_subplots = uv.plot_multiple_tendency(df = data_zoom, number_columns = 1)
    fig_tendency_all_subplots.write_html(f"output_eda/{id_report}/univariate_analysis/trend_zoom/subplots_zoomtendency.html")

    # fig tendency all features in individual plots
    for feature_ in param_list_features:
        fig_tendency_all_individual = uv.plot_tendency(df = data_zoom, feature_plot = feature_)
        feature_ = transform_strings_to_save(feature_) # replace bad characters to save name
        fig_tendency_all_individual.write_html(f"output_eda/{id_report}/univariate_analysis/trend_zoom/tendency_{feature_}.html")
    
    # fig tendency all features in oneplot
    fig_tendency_all_oneplot = uv.plot_all_trend_oneplot(df = data_zoom)
    fig_tendency_all_oneplot.write_html(f"output_eda/{id_report}/univariate_analysis/trend_zoom/oneplot_zoom_tendency.html")

    
    # ################### full data - trend ###################
    print(f'trend full data... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    for feature_ in param_list_features:
        fig_tendency_all_individual = uv.plot_tendency(df = data_zoom, feature_plot = feature_)
        feature_ = transform_strings_to_save(feature_) # replace bad characters to save name
        fig_tendency_all_individual.write_html(f"output_eda/{id_report}/univariate_analysis/trend/tendency_{feature_}.html")    

    
    # ################### fig boxplot for each month and year ###################  
    print(f'boxplots... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')    
    fig_boxplot_all = uv.plot_multiple_boxplot_months(df = data, number_columns = 1)  # always 1 boxplot for column beacuse there are 12 months
    fig_boxplot_all.write_html(f"output_eda/{id_report}/univariate_analysis/boxplots.html")
    
    # ################### fig smooth data ###################    
    ## moving average
    print(f'moving average... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    data_moving_average = uv.apply_moving_average(df = data_zoom.copy(), window_size = param_smooth_ma_window)
    fig_moving_average = uv.plot_compare_tendencias(df_original = data_zoom, 
                                                    df_smoothed = data_moving_average,
                                                    number_columns = param_number_columns,
                                                    kind_smooth = f'moving average - window: {param_smooth_ma_window}'
                                                )
    fig_moving_average.write_html(f"output_eda/{id_report}/univariate_analysis/moving_average.html")
    
    ## weighted moving average
    print(f'weighted moving average... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    data_weighted_moving_average = uv.apply_weighted_moving_average(df = data_zoom.copy(), weights = param_smooth_wma_weights)
    fig_weighted_moving_average = uv.plot_compare_tendencias(df_original = data_zoom,
                                                             df_smoothed = data_weighted_moving_average,
                                                             number_columns = param_number_columns,
                                                             kind_smooth = f'weighted moving average - weights: [{param_smooth_wma_weights}]'
                                                            )
    fig_weighted_moving_average.write_html(f"output_eda/{id_report}/univariate_analysis/weighted_moving_average.html")
    
    ## exponential moving average
    print(f'exponential moving average... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    data_exponential_moving_average = uv.apply_exponential_moving_average(df = data_zoom.copy(), alpha = param_smooth_ema_aplha)
    fig_exponential_moving_average = uv.plot_compare_tendencias(df_original = data_zoom,
                                                                df_smoothed = data_exponential_moving_average,
                                                                number_columns = param_number_columns,
                                                                kind_smooth = f'exponential moving average - alpha: {param_smooth_ema_aplha}'
                                                               )
    fig_exponential_moving_average.write_html(f"output_eda/{id_report}/univariate_analysis/exponential_moving_average.html")
    
    
    # ################### fig acf ###################
    print(f'acf... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_acf = uv.plot_all_acf(df = data, lags = param_lags, number_columns = param_number_columns)
    fig_acf.write_html(f"output_eda/{id_report}/univariate_analysis/acf.html")

    print(f'acf stats models... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_acf_stats = uv.plot_all_acf_stats(df = data, lags = param_lags, number_columns = param_number_columns) # v2 statsmodels
    fig_acf_stats.savefig(f"output_eda/{id_report}/univariate_analysis/acf_stats.png", dpi = 300)
    
    
    # ################### fig pacf ###################
    print(f'pacf... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_pacf = uv.plot_all_pacf(df = data, lags = param_lags, number_columns = param_number_columns)
    fig_pacf.write_html(f"output_eda/{id_report}/univariate_analysis/pacf.html")

    print(f'pacf stats models... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_acf_stats = uv.plot_all_pacf_stats(df = data, lags = param_lags, number_columns = param_number_columns) # v2 statsmodels
    fig_acf_stats.savefig(f"output_eda/{id_report}/univariate_analysis/pacf_stats.png", dpi = 300)


    # ################### end ###################
    print(f'end... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')



if show_bivariate_analysis:
    print("\n--------------- BIVARIATE ANALYSIS ---------------")
    
    """ PARAMS """
    # read param correlations
    param_theshold_corr_all_features = config['bivariate_analysis']['correlations']['threshold_corr_all_features']  # threshold in correlations between each feature 
    param_theshold_corr_target = config['bivariate_analysis']['correlations']['threshold_corr_target']  # threshold in correlations between a target
    
    # read param scatter plot individual
    param_individual_scatter_marginal = config['bivariate_analysis']['scatter_plot']['individual_scatter']['marginal']
        
    # read param corr features lagged vs target
    param_lag_features = config['bivariate_analysis']['correlations_features_lagged_target']['lags']

    # read params parallel
    param_features_parallel = config['bivariate_analysis']['parallel']['list_features']
    param_features_target_parallel = param_features_parallel + [param_target]
    
    
    """ PLOTS """
    ################### fig correlations ###################
    print(f'correlations... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # correlations all features
    _, df_corr_upper = bv.calculate_correlations_triu(data)
    df_corr_upper_filtered = bv.filter_correlations_by_threshold(df_corr_upper, param_theshold_corr_all_features)
    fig_corr_all = bv.plot_heatmap(df_corr = df_corr_upper_filtered)
    fig_corr_all.write_html(f"output_eda/{id_report}/bivariate_analysis/corr_all.html")
    
    # correlations against the target
    corr_target = bv.calculate_correlations_target(data, param_target)
    corr_target_filtered = bv.filter_correlations_by_threshold(corr_target, param_theshold_corr_target)
    fig_corr_target = bv.plot_heatmap(df_corr = corr_target_filtered)
    fig_corr_target.write_html(f"output_eda/{id_report}/bivariate_analysis/corr_target.html")
    
    
    ################### fig scatter plots ###################
    print(f'scatters plots... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # --- plots features-target - subplots. save in scatter-features-target
    fig_scatter_features_target = bv.plot_features_to_target_scatter_plot_low(df = data, target = param_target, number_columns = param_number_columns)
    fig_scatter_features_target.write_html(f"output_eda/{id_report}/bivariate_analysis/scatter-features-target/scatter_features_target.html")
    pio.write_image(fig_scatter_features_target, f"output_eda/{id_report}/bivariate_analysis/scatter-features-target/scatter_features_target.png")  # -----> save as png because a lot of plots could be generated and freeze the pc in a html file

    # --- plots features-target - individual. save in scatter-features-target
    for feature_ in param_list_features:
        fig_individual_scatter_features_target = bv.plot_individual_scatter_plot(df = data, feature_x = feature_, feature_y = param_target,
                                                                                   marginal_hist = param_individual_scatter_marginal)
        feature_ = transform_strings_to_save(feature_) # replace bad characters to save name
        fig_individual_scatter_features_target.write_html(f"output_eda/{id_report}/bivariate_analysis/scatter-features-target/scatter-{feature_}.html")
        
    # --- plots features-features - scatter matrix. save in scatter-features-features
    #fig_scatter_all_features = bv.plot_all_features_scatter_plot_mine(df = data, number_columns = param_number_columns) ## mine old
    fig_scatter_features_features = bv.plot_all_features_scatter_plot(df = data[param_list_features])
    fig_scatter_features_features.write_html(f"output_eda/{id_report}/bivariate_analysis/scatter-features-features/scatter_matrix_features_features.html")
    
    # --- plots features-features - individual. save in scatter-features-features
    list_features_features = bv.list_map_features_features(df = data[param_list_features])
    for feature_x, feature_y in list_features_features:
        fig_individual_scatter_features_features = bv.plot_individual_scatter_plot(df = data, feature_x = feature_x, feature_y = feature_y,
                                                                                   marginal_hist = param_individual_scatter_marginal)
        feature_x = transform_strings_to_save(feature_x) # replace bad characters to save name
        feature_y = transform_strings_to_save(feature_y) # replace bad characters to save name
        fig_individual_scatter_features_features.write_html(f"output_eda/{id_report}/bivariate_analysis/scatter-features-features/scatter-{feature_x}-{feature_y}.html")

    
    
    ################### fig correlations features lagged vs target ###################
    print(f'correlations features lagged vs target... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    df_corr_features_lag_target = bv.calculate_corr_features_lag_target(df = data, target = param_target, lags = param_lag_features)
    fig_corr_lag = bv.plot_corr_features_lag_target(df_corr_lags = df_corr_features_lag_target)
    fig_corr_lag.write_html(f"output_eda/{id_report}/bivariate_analysis/plot_corr_features_lag_target.html")



    ################### fig parallel all continuous variables ###################
    print(f'parallel continuous ... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_parallel_continuous = bv.plot_parallel_continuous(df = data, 
                                                         list_features_target = param_features_target_parallel, 
                                                         target = param_target)
    fig_parallel_continuous.write_html(f"output_eda/{id_report}/bivariate_analysis/parallel_continous_variables.html")
    
    
    # ################### end ###################
    print(f'end... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')



if show_segmentation_analysis:
    print("\n--------------- SEGMENTATION ANALYSIS ---------------")
    
    """ PARAMS """
    # list of independient segmentations of the data
    list_segments_data = config['segmentation_analysis']['segments']
    config_segments_data = list_segments_data[0] # TODO: modify codes to multiple independient segmentations
    
    param_segments_data_var = config_segments_data['var_segment']
    param_segments_data_intervals = config_segments_data['interval_segment']
    param_segments_data_labels = config_segments_data['labels_segment']
    
    # params correlations
    param_segmentation_corr_all_threshold = config['segmentation_analysis']['correlations']['threshold_corr_all_features']
    param_segmentation_corr_target_threshold = config['segmentation_analysis']['correlations']['threshold_corr_target']

    # read param scatter plot individual
    param_segmentation_individual_scatter_marginal = config['segmentation_analysis']['scatter_plot']['individual_scatter']['marginal']

    # read param parallel discrete target
    param_parallel_discrete_target_show = config['segmentation_analysis']['parallel_target_discrete']['show']
    param_features_parallel_discrete_target = config['segmentation_analysis']['parallel_target_discrete']['list_features']
    param_features_target_parallel_discrete_target = param_features_parallel_discrete_target + [param_segments_data_var + '_segments'] # list features and target with the suffix "_segment" beacuase the target is segmented


    """ GENERATE DATA SEGMENTED """ # segment and sort data by variable segment to do all the plots in order incremental of the segmentation
    data_segmented = aux.custom_segmentation(df = data.copy(), 
                                            var_segment = param_segments_data_var, 
                                            intervals_segments = param_segments_data_intervals, 
                                            labels_segments = param_segments_data_labels
                                           )

    """ PLOTS """
    ################### freq each segment ###################
    print(f'freq segment... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_freq_segmentation = se.plot_freq_segmentation(df = data_segmented, var_segment = param_segments_data_var)
    fig_freq_segmentation.write_html(f"output_eda/{id_report}/segmentation_analysis/freq_segmentation.html")
    
    
    ################### descriptive statistics ###################
    print(f'descriptive staticstics segmented segmented data... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    dict_df_statistics_segment = se.calculate_descriptive_statistics_segment(df = data_segmented, var_segment = param_segments_data_var + '_segments')
    df_statistics_segments = se.merge_segmentation_statistics(dict_df_statistics_segment)
    fig_statistics_segmentation = se.plot_descriptive_statistics_segment(df_statistics_segments)
    fig_statistics_segmentation.write_html(f"output_eda/{id_report}/segmentation_analysis/statistics_segmentation.html")
    
    
    ################### histograms - boxplots ###################
    print(f'histograms - boxplots segmented data... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # hist
    fig_hist_segment = se.plot_histograms_segments(df = data_segmented, var_segment = param_segments_data_var + '_segments', 
                                                 number_columns = param_number_columns)
    fig_hist_segment.write_html(f"output_eda/{id_report}/segmentation_analysis/histograms_segmentation.html")
    
    # boxplot
    fig_boxplots_segment = se.plot_boxplots_segments(df = data_segmented, var_segment = param_segments_data_var + '_segments', 
                                                 number_columns = param_number_columns)
    fig_boxplots_segment.write_html(f"output_eda/{id_report}/segmentation_analysis/boxplots_segmentation.html")
    
    
    ################### trend - scatter segmentation ###################
    print(f'trend - scatter segmentation... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    data_segmented_sort_index = data_segmented.sort_index() # sort data by index
    fig_trend_segment = se.plot_multiple_tendency_segmentation(df = data_segmented_sort_index, var_segment = param_segments_data_var + '_segments', 
                                                               number_columns = 1)
    fig_trend_segment.write_html(f"output_eda/{id_report}/segmentation_analysis/trend_segmentation.html")
    
    ################### correlations ###################
    print(f'correlations segmented data... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # corr all features
    dict_df_corr_segment = se.calculate_correlations_triu_segmentation(df =  data_segmented, 
                                                                       var_segment = param_segments_data_var + '_segments')
    dict_df_corr_segment = se.filter_correlations_segment_by_threshold(dict_df_corr_segment, threshold = param_segmentation_corr_all_threshold)
    fig_corr_segmentation_heatmap = se.plot_corr_segmentation_subplots_heatmap(dict_df_corr_segment)
    fig_corr_segmentation_heatmap.write_html(f"output_eda/{id_report}/segmentation_analysis/corr_segmentation_heatmap.html")
    
    # corr target
    dict_df_corr_segment_target = se.calculate_correlations_target_segmentation(df =  data_segmented, 
                                                                                var_segment = param_segments_data_var + '_segments', 
                                                                                target = param_target)
    dict_df_corr_segment_target = se.filter_correlations_segment_by_threshold(dict_df_corr_segment_target, threshold = param_segmentation_corr_target_threshold)
    fig_corr_segmentation_target_barchat = se.plot_corr_segmentation_vertical_barchart(dict_df_corr_segment_target)
    fig_corr_segmentation_target_barchat.write_html(f"output_eda/{id_report}/segmentation_analysis/corr_segmentation_target_barchat.html")
    fig_corr_segmentation_target_barchat.write_image(f"output_eda/{id_report}/segmentation_analysis/corr_segmentation_target_barchat.png")
    
    
    ################### fig scatter plots ###################
    # # individual scatter
    # if param_segmentation_individual_scatter_show == True:
    #     fig_segmentation_individual_scatter = se.plot_individual_scatter_plot_x_y_segment(df = data_segmented, 
    #                                                              feature_x = param_segmentation_individual_scatter_feature_x, 
    #                                                              feature_y = param_segmentation_individual_scatter_feature_y, 
    #                                                              var_segment = param_segments_data_var + '_segments')
    #     fig_segmentation_individual_scatter.write_html(f"output_eda/{id_report}/segmentation_analysis/segmentation_individual_scatter.html")
        
    # # scatter all features vs all features
    # if param_segmentation_features_scatter_show == True:
    #     #fig_segmentation_scatter_all_features = se.plot_all_features_scatter_plot_segment_mine(df = data_segmented, var_segment = param_segments_data_var + '_segments', number_columns = param_number_columns)
    #     fig_segmentation_scatter_all_features = se.plot_all_features_scatter_plot_segment(df = data_segmented, var_segment = param_segments_data_var + '_segments')
    #     fig_segmentation_scatter_all_features.write_html(f"output_eda/{id_report}/segmentation_analysis/segmentation_scatter_matrix_all_features.html")
    #     #pio.write_image(fig_segmentation_scatter_all_features, f"output_eda/{id_report}/segmentation_analysis/segmentation_scatter_matrix_all_features.png") 
    


    ################### fig scatter plots ###################
    print(f'scatters plots segmented... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


    # --- plots features-target - subplots. save in scatter-features-target
    # TODO
    list_features_features_segmented = param_list_features + [param_segments_data_var + '_segments']
    
    # --- plots features-target - individual. save in scatter-features-target
    for feature_ in param_list_features:
        fig_individual_scatter_features_target = se.plot_individual_scatter_plot_segment(df = data_segmented, feature_x = feature_, feature_y = param_target,
                                                                                         var_segment = param_segments_data_var + '_segments',
                                                                                   marginal_hist = param_segmentation_individual_scatter_marginal)
        feature_ = transform_strings_to_save(feature_) # replace bad characters to save name
        fig_individual_scatter_features_target.write_html(f"output_eda/{id_report}/segmentation_analysis/scatter-features-target/scatter-{feature_}.html")

    
    # --- plots features-features - scatter matrix. save in scatter-features-features
    #fig_scatter_all_features = se.plot_all_features_scatter_plot_mine(df = data_segmented, var_segment = param_segments_data_var + '_segments', number_columns = param_number_columns) ## mine old
    fig_scatter_features_features = se.plot_all_features_scatter_plot_segment(df = data_segmented[list_features_features_segmented], 
                                                                      var_segment = param_segments_data_var + '_segments')
    fig_scatter_features_features.write_html(f"output_eda/{id_report}/segmentation_analysis/scatter-features-features/scatter_matrix_features_features.html")

    
    # --- plots features-features - individual. save in scatter-features-features
    list_features_features = se.list_map_features_features(df = data_segmented[param_list_features])
    for feature_x, feature_y in list_features_features:
        fig_individual_scatter_features_features = se.plot_individual_scatter_plot_segment(df = data_segmented, feature_x = feature_x, feature_y = feature_y,
                                                                                           var_segment = param_segments_data_var + '_segments',
                                                                                   marginal_hist = param_segmentation_individual_scatter_marginal)
        feature_x = transform_strings_to_save(feature_x) # replace bad characters to save name
        feature_y = transform_strings_to_save(feature_y) # replace bad characters to save name
        fig_individual_scatter_features_features.write_html(f"output_eda/{id_report}/segmentation_analysis/scatter-features-features/scatter-{feature_x}--{feature_y}.html")


    ################### fig parallel plots ###################
    if param_parallel_discrete_target_show == True:
        print(f'parallel discrete target... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        fig_parallel_target_discrete = se.plot_parallel_continuous_discrete_target(df = data_segmented, 
                                                                        list_features_target = param_features_target_parallel_discrete_target, 
                                                                        var_segment_target_discrete = param_segments_data_var + '_segments')
        fig_parallel_target_discrete.write_html(f"output_eda/{id_report}/segmentation_analysis/parallel_target_discrete.html")


    # ################### end ###################
    print(f'end... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')



if show_categorical_analysis:
    print("\n--------------- CATEGORICAL ANALYSIS ---------------")
    
    """ PARAMS """ # categorical analysis: ca
    # params with features and kind of percentile transformation
    # list features
    param_list_features_ca = config['categorical_analysis']['percentile_transform']['categories_features']['features']
    param_list_cat_features_percentile_ca = config['categorical_analysis']['percentile_transform']['categories_features']['percentile']
    
    # list feature+target
    param_list_target_ca = config['categorical_analysis']['percentile_transform']['categories_target']['target']
    param_list_target_percentile_ca = config['categorical_analysis']['percentile_transform']['categories_target']['percentile']
    param_list_features_target_ca = param_list_features_ca + param_list_target_ca
    param_list_cat_features_target_percentile_ca = param_list_cat_features_percentile_ca + param_list_target_percentile_ca


    ## GENERATE LIST OF NAMES OF FEATURES AND TARGET when the data is transformed into categorical change its name adding a suffix the percentile
    param_target_name_percentile = param_list_target_percentile_ca[0] + '_' + param_list_target_ca[0]

    
    # params to calculate table/heatmap/hist2d of frequency between eaach pair of features in the data
    param_ct_normalized_freq_pair_feature = config['categorical_analysis']["crosstab_freq_pair_features"]["freq_normalized"]
    
    # params to calculate table/heatmap of frequency between each target categorical vs feature categorical
    param_ct_normalized_freq_target_feature = config['categorical_analysis']["crosstab_freq_target_feature"]["freq_normalized"]
    
    # param functions of aggregation target in heatmap feature1 & feature 2 vs target
    param_list_agg_target_multiple_features = config['categorical_analysis']["heatmap_multiple_features_vs_target_continuous"]["aggregation_target"]
    
    # param list of features and target to plot into a parellel plot - read features originals name and transform name according percetile transformtation
    param_list_features_to_parallel_original = config['categorical_analysis']["parallel"]["list_features"]
    list_indexes_to_parallel_plot = []
    for index in range(len(param_list_features_ca)):
        if param_list_features_ca[index] in param_list_features_to_parallel_original:
            list_indexes_to_parallel_plot.append(index)
            
    list_percentile_to_parallel_plot = [param_list_cat_features_percentile_ca[element_index] for element_index in list_indexes_to_parallel_plot]
    param_list_features_to_parallel = [list_percentile_to_parallel_plot[index] + '_' + param_list_features_to_parallel_original[index] \
                                       for index in range(len(list_percentile_to_parallel_plot))]

    param_list_features_target_to_parallel = param_list_features_to_parallel + [param_target_name_percentile]


    """ GENERATE DATA CATEGORICAL """
    # categorize only features and conserve continuos target
    data_percentile_feature = data.copy()
    for index, variable in enumerate(param_list_features_ca):
        data_percentile_feature = aux.percentile_segmentation(df = data_percentile_feature, 
                                                              var_segment = variable, 
                                                              type_percentile = param_list_cat_features_percentile_ca[index]
                                                         )
        data_percentile_feature.drop(columns = variable, inplace = True)
    
    
    # categorize features+target and delete features continous variables
    data_percentile_feature_target = data.copy()
    for index, variable in enumerate(param_list_features_target_ca):
        data_percentile_feature_target = aux.percentile_segmentation(df = data_percentile_feature_target, 
                                                                 var_segment = variable, 
                                                                 type_percentile = param_list_cat_features_target_percentile_ca[index]
                                                                )
        data_percentile_feature_target.drop(columns = variable, inplace = True)


    
    """ PLOTS """
    # --- table frequency for each categorie for each feature
    print(f'table freq each catergory each feature.. time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    df_freq_categorical_variables, df_freq_categorical_variables_plotly = ca.calculate_freq_data(df = data_percentile_feature_target)
    fig_table_freq_categorical_variables = ca.plot_df_table_plotly(df_to_plotly = df_freq_categorical_variables_plotly)
    fig_table_freq_categorical_variables.write_html(f"output_eda/{id_report}/categorical_analysis/table_freq_categorical_variables.html")
    df_freq_categorical_variables.to_excel(f"output_eda/{id_report}/categorical_analysis/df_freq_categorical_variables.xlsx")
    
    
    
    # --- plots/tables/heatmap-> hist2d frecuency between each pair of features
    print(f'hist2d freq each feature categortical... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_freq_heatmap_all_features_percentile = ca.heatmap_hist2d_features_percentile(df = data_percentile_feature, 
                                                                                  target = param_target, 
                                                                                  ct_normalized = param_ct_normalized_freq_pair_feature)
    fig_freq_heatmap_all_features_percentile.write_html(f"output_eda/{id_report}/categorical_analysis/freq_heatmap_hist2d_all_features_percentile.html")
    
    
    
    
    # --- plots Individual analysis between "feature x categorical" and "target y Continuous" 
    
    # table statistics of target for each category in each feature
    print(f'statistics target each category each feature... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    df_statistics_target, df_statistics_target_to_plotly = ca.descriptive_statistics_target_for_each_feature(df = data_percentile_feature,
                                                                                                     target = param_target)
    fig_table_statistics_target_to_plotly = ca.plot_df_table_plotly(df_statistics_target_to_plotly)
    fig_table_statistics_target_to_plotly.write_html(f"output_eda/{id_report}/categorical_analysis/table_statistics_target.html")
    df_statistics_target.to_excel(f"output_eda/{id_report}/categorical_analysis/statistics_target.xlsx")
    
    
    # boxplot of target for each category in each feature
    print(f'boxplot target each category each feature... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_boxplot_target_colored_cartegorical_features = ca.plot_boxplots_target_categorical_features(df = data_percentile_feature, 
                                                                                                 var_continuous_hist = param_target, # target continous
                                                                                                 number_columns = param_number_columns)
    fig_boxplot_target_colored_cartegorical_features.write_html(f"output_eda/{id_report}/categorical_analysis/boxplot_target_colored_cartegorical_features.html")
    
    
    
    # --- plots Individual analysis between "feature x categorical" and "target y categorical"
    
    # table freq of target categorical for each category in each feature
    print(f'statistics target categorical each feature categorical... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    df_freq_target_each_feature, df_freq_target_each_feature_plotly = ca.calculate_freq_target_each_features(df = data_percentile_feature_target, 
                                                                                     target = param_target_name_percentile,
                                                                                     ct_normalized = param_ct_normalized_freq_target_feature)
    fig_freq_target_each_feature = ca.plot_df_table_plotly(df_freq_target_each_feature_plotly)
    fig_freq_target_each_feature.write_html(f"output_eda/{id_report}/categorical_analysis/freq_target_each_feature.html")
    df_freq_target_each_feature.to_excel(f"output_eda/{id_report}/categorical_analysis/statistics_target.xlsx")
    
    
    # barplot freq of target categorical for each category in each feature
    print(f'barplot freq target each category each feature... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    barplot_freq_target_1_all_features = ca.barplot_crosstab_freq_target_1_features(df = data_percentile_feature_target,
                                                                             target = param_target_name_percentile,
                                                                             number_columns = 1)
    barplot_freq_target_1_all_features.write_html(f"output_eda/{id_report}/categorical_analysis/barplot_freq_target_all_features_invidually.html")
    
    
    # --- plots multiple analysis between "feature x categorical" and "target y Continuous" - in the heatmap of relation between feature_x, feature_y and target, the target must be aggregate as mean, std, etc
    
    # heatmap feature1 & feature2 vs target
    print(f'heatmap continous target vs 2 categorical features... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    for param_agg_target in param_list_agg_target_multiple_features:
        fig_crosstab_agg_target_2_features = ca.heatmap_crosstab_aggregation_target_2_features(df = data_percentile_feature, 
                                                                                               target = param_target, 
                                                                                               agg_target = param_agg_target, 
                                                                                               number_columns = 1)
        fig_crosstab_agg_target_2_features.write_html(f"output_eda/{id_report}/categorical_analysis/crosstab_{param_agg_target}_target_2_features.html")
    
    
    # heatmap feature1 & feature2 & feature3 vs target
    print(f'heatmap continous target vs 3 categorical features... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    for param_agg_target in param_list_agg_target_multiple_features:
        fig_crosstab_mean_target_3_features = ca.heatmap_crosstab_aggregation_target_3_features(df = data_percentile_feature, 
                                                                                                target = param_target,
                                                                                                agg_target = param_agg_target, 
                                                                                                number_columns = 1)
        fig_crosstab_mean_target_3_features.write_html(f"output_eda/{id_report}/categorical_analysis/crosstab_{param_agg_target}_target_3_features.html")
    
    
    # --- plots multiple analysis between "feature x categorical" and "target y Continuous"
    print(f'barplot categorical target vs 2 categorical features... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    barplot_freq_target_2_all_features = ca.barplot_crosstab_freq_target_2_features(df = data_percentile_feature_target,
                                                                                    target = param_target_name_percentile,
                                                                                    number_columns = 1)
    barplot_freq_target_2_all_features.write_html(f"output_eda/{id_report}/categorical_analysis/barplot_freq_target_2_all_features.html")
    
    
    # --- plots parallel features categorical vs target categorical
    print(f'parellel categorical features and categorical target... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    fig_parallel_discrete = ca.plot_parallel_discrete_variables(df_percentile = data_percentile_feature_target, 
                                                                list_features_target_to_plot = param_list_features_target_to_parallel, 
                                                                target = param_target_name_percentile)
    fig_parallel_discrete.write_html(f"output_eda/{id_report}/categorical_analysis/parallel_discrete.html")
    
    
    
    # ################### end ###################
    print(f'end... time:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
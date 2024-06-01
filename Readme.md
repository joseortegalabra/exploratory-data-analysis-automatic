# Automatic EDA
Write a configuration json file "config.json" and run a python script "main.py" to generate automatic EDA for forecast time series


## Observations:
- Most of the analysis can be applied for other kind of data. Only trends plots and acf/pacf are applied only to time series
- In the folder data there are a jupyter notebook to generate 3 differents datasets used in this notebook. But the pkl file with the data are 
not pushed
- There is a folder output_eda where are saved the plots output of the eda. This output are not pushed but can be obtained running the codes
- There codes to do subplots that are invoked in the script main.py. But, in addition, there are codes to generate individual plots that there are not called in the script main.py (for example: there a function to plot subplots of histograms that are called in the script main.py but also there a function to plot the histogram of only one feature). **For ALL PLOTS there the version of individual plot and the version of subplot**
- The repo with jupyter notebooks used to generate this scripts can be find in the following link: https://github.com/joseortegalabra/exploratory-data-analysis


## Templates codes:
- There are a lot of functions that recibe a pandas dataframe and other parameters that each function needs and return a plotly figure
- Then with the plotly figure you can see in a jupyter notebok with the method fig.show() or you can saved it in a folder with the methods
fig.write_html to get a html interactive figure or fig.write_image to get a static image similar to other packages as matplotlib or seaborn
- In the script main.py the order is read the config file, then read the parameters since de config.json that you will use and finally call the function
to generate the plots to get a plotly figure and finally decide what to do (show, save html, save png, save pdf, etc)


## Run codes:
- It is very simple
- Open a console, for example anaconda prompt
- activate env that you are using. conda env list. conda activate -name
- navigate into folder where are located this repo. cd .. cd automatic-exploratory-data-analysis
- run script main.py.  -> python main.py


## Explications config.json
Explications of config.json to complete it. Important, this configuration is only for the plots that need it. In the codes a lot of more plots is generated, but this ones doens't need parameters

---

### Initial parametes

**global parameters**

"name_report": "indicate the name of the report"

"name_data_pkl": "indicate name of the file that have the data"

"target": "indicate name of target"

"list_features": "indicate list of features"

"number_columns": "indicate the numbers of columns to plot that accepted multiple columns"

**"reports_to_show"**
Indicate true/false which reports to do the plots and which reports skip


---
### Univariate analysis

**"ydata_profiling"**

"minimal": do a minimal report of ydata-profiling. always true when the dataset is huge


**"zoom_tendency:(start_date, end_date)"**: indicate the dates to plot trends. When the data is huge plot all the data could be too much

**"smooth_data"**: indicate the parameters of differents ways of smooth the data. Such as, moving average, weighted moving average and exponential moving average

**acf_pacf:lags** indicate the max number of lags to plot the autocorrelation function and partial autocorrelation function



---
### Bivariate analysis

**correlations:(threshold)**: indicate if add a threshold to show the correlations. for example, only show the correlations with value over 0.1 

**scatter_plot:(marginal)**: plot a scatter plot and a marginal histograms of each feature in the scatter plot

**correlations_features_lagged_target:(lags)**: indicate the number of lags in the features used to analyze the correlations of the features lagged againts the target

**parallel:(list_features)**:indicate the list of features to plot into a parallel plot vs the target as final step 


---
### Segmentation analysis

**segmentation_analysis**

"type": indicate type of segmentation. custom or by percentile

"var_segment": indicate feature or target to segment the data

"interval_segment": if custom segmentation, indicate the intervals of the values to generate the differents segments

"labels_segment": if custom segmentation, indicate the name of the differents segments



---
### Categorical analysis

**categorical_analysis**

"features": list of features that will transformend into a categorical variable. It is neccesary transform all the features of the dataframe

"percentile_features": list of kind of percentile to categorize each feature. choices: quartile, quintile, decile

"target":lit of the target that will transformed into a categorical variable

 "percentile_target":list of kind of percentile to categorize the target. choices: quartile, quintile, decile

"crosstab_freq_pair_features (freq_normalized)": normalize the table of frecuency between each pair of features

"crosstab_freq_target_feature (freq_normalized)": normalize the table of frecuency between each feature vs target

"heatmap_multiple_features_vs_target_continuous (aggregation_target)": aggregation of the target in a comparative table between each categorie in pair of feature and the aggregation of the target, for example mean, std, min, max, etc

**parallel:(list_features)**:indicate the list of features to plot into a parallel plot vs the target as final step 


---


### Kind of plots presents in this repo - notebooks with codes used to develop this plots


**ydata-profiling**

[1 link-data-profiling](https://github.com/joseortegalabra/exploratory-data-analysis/tree/main/1_ydata_profiling)



**Univariate Analysis**

[1 descriptive statistics table](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/2_plots_univariate_analysis/1_table_statistics.ipynb)

[2 histograms](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/2_plots_univariate_analysis/2_histogram.ipynb)

[3 kernel density + histograms](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/2_plots_univariate_analysis/3_kernel_density.ipynb)

[4 original data trend](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/2_plots_univariate_analysis/4_tendency.ipynb)

[5 boxplots with monthly aggregation](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/2_plots_univariate_analysis/5_boxplot_months.ipynb)

[6 original vs smoothed data trend](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/2_plots_univariate_analysis/6_smooth_data.ipynb)

[7 Autocorrelation and partial autocorrelation functions-v1](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/2_plots_univariate_analysis/7_autocorrelations.ipynb)

[7 Autocorrelation and partial autocorrelation functions-v2](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/2_plots_univariate_analysis/7_autocorrelations_v2.ipynb)

[8 Other analyzes for time series-v1](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/2_plots_univariate_analysis/8_partial_autocorrelacions.ipynb)

[8 Other analyzes for time series-v2](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/2_plots_univariate_analysis/8_partial_autocorrelations-v2.ipynb)


**Bivariate Analysis**

[1 correlations](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/3_plots_bivariate_analysis/1_correlations_pearson.ipynb)

[2 scatter plots](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/3_plots_bivariate_analysis/2_scatter_dispersion.ipynb)

[3 correlations features with lag](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/3_plots_bivariate_analysis/3_correlations_features_lag_target.ipynb)

[4 Multivariate parallel plot](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/3_plots_bivariate_analysis/4_parallel.ipynb)


**Segmentation Analysis**

[0 generate data segmentation](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/4_plots_segmentation_analysis/0_intro_get_data.ipynb)

[1 segmented data distribution](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/4_plots_segmentation_analysis/1_distribution_segmentation.ipynb)

[2 descriptive statistical table segmented data](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/4_plots_segmentation_analysis/2_table_statistics_segmentation.ipynb)

[3 histograms and boxplots segmented data](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/4_plots_segmentation_analysis/3_histogram_boxplots_segmentation.ipynb)

[4 trend segmented data](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/4_plots_segmentation_analysis/4_tendency_segmentation.ipynb)

[5 segmented data correlations](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/4_plots_segmentation_analysis/5_correlations_segmentation.ipynb)

[6 scatter plots segmented data](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/4_plots_segmentation_analysis/6_scatter_segmentation.ipynb)

[7 parallel plot only when target is segmented](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/4_plots_segmentation_analysis/7_parallel_target_segmentation.ipynb)


**Categorical Analysis**

[0 generate categorical data](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/5_plots_categorical_analysis/0_intro_get_data.ipynb)

[2 crosstab frequency features and target](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/5_plots_categorical_analysis/2_crosstab_freq_features_target.ipynb)

[3 frequency between 2 categorical features](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/5_plots_categorical_analysis/3_freq_x_cat_x_cat.ipynb)

[4 univariate analysis categorical features vs continuous target](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/5_plots_categorical_analysis/4_univariate_analysis_x_cat_y_cont.ipynb)

[5 univariate analysis categorical features vs categorical target](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/5_plots_categorical_analysis/5_univariate_analysis_x_cat_y_cat.ipynb)

[6 bivariate analysis categorical features vs continuous target](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/5_plots_categorical_analysis/6_bivariate_analysis_x_cat_y_cont.ipynb)

[7 bivariate analysis categorical features vs categorical target](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/5_plots_categorical_analysis/7_bivariate_analysis_x_cat_y_cat.ipynb)

[8 parallel plot features categorical vs categorical target](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/5_plots_categorical_analysis/8_parallel_x_cat_y_cat.ipynb)

[9 woe iv - categorical features vs binary target](https://github.com/joseortegalabra/exploratory-data-analysis/blob/main/5_plots_categorical_analysis/9_woe_iv_x_cat_y_cat.ipynb)

from ydata_profiling import ProfileReport



def generate_report_ydata_profiling(df, minimal = True, id_report = "0000"):
    """
    Given a dataframe, generate a basic html report using y data profiling
    
    Args
        df (dataframe): data to generate report
        minimal (boolean): True/False: generate a minimal report because some analysis are expensive. https://ydata-profiling.ydata.ai/docs/master/pages/use_cases/big_data.html#
        id_report (string): id_report when the code are running. Used to save differents reports and don't overwrite them

    Return
        html file saved with the report
    """

    # generar un report estandar de profiling
    profile = ProfileReport(df, title = "Basic profiling Report", minimal = minimal)

    # guardar html con el reporte
    profile.to_file(f"output_eda/{id_report}/ydata_profiling/profiling.html")
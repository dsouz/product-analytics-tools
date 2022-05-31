
def get_variant_binomial_stats(df, variant_col, metric_col, control_variant): # TODO: Need to fix
    temp_df = df[[variant_col, metric_col]].copy()
    sample_c = 1
    return 1
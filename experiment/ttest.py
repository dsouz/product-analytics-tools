import math
import scipy.stats as st
import logging
import pandas as pd
from statsmodels.stats.multitest import multipletests


def run_ttest(
    sample_c, conv_c, sample_t, conv_t, conf=0.95
):  # TODO: Add conf intervals
    p_c = conv_c / sample_c
    p_t = conv_t / sample_t
    if p_c <= 0 or p_t <= 0 or p_c > 1 or p_t > 1:
        return False, 1
    se_t = math.sqrt(p_t * (1 - p_t) / sample_t)
    se_c = math.sqrt(p_c * (1 - p_c) / sample_c)
    se_diff = math.sqrt(math.pow(se_c, 2) + math.pow(se_t, 2))
    z_val = (p_t - p_c) / se_diff
    pval = 1 - st.norm.cdf(z_val)
    logging.debug(
        f"Inputs {sample_c}:{conv_c}:{sample_t}:{conv_t}, p-val is {pval}, Z is {z_val }"
    )
    if pval < 1 - conf:
        return True, pval
    else:
        return False, pval
    return False, 1


def get_dim_level_pvals(exp_df, dim_list, metric_col, group_col="group"):
    all_pvals = []
    for dim in dim_list:
        group_l = [dim, group_col]
        grouped = exp_df.groupby(group_l).agg({metric_col: ["sum", "size"]})
        grouped.columns = ["_".join(a) for a in grouped.columns.to_flat_index()]
        grouped = grouped.reset_index()
        control_df = grouped.loc[grouped[group_col] == "control"]
        control_df = control_df.rename(
            columns=lambda x: "control_" + x if x not in group_l else x
        )
        control_df = control_df.drop(columns=[group_col])
        control_df["join_key"], grouped["join_key"] = 1, 1
        final_df = grouped.merge(control_df, how="left", on=["join_key"] + [dim])
        final_df[["result", "pval"]] = final_df.loc[
            final_df[group_col] != "control"
            ].apply(
            lambda x: run_ttest(
                x["control_convs_size"],
                x["control_convs_sum"],
                x["convs_size"],
                x["convs_sum"],
            ),
            axis=1,
            result_type="expand",
        )
        final_df['dim'] = dim
        final_df = final_df.rename(columns={dim: "dim_value"})
        sig_segments = final_df.loc[(final_df[group_col] != "control"), ["dim", "dim_value", "pval"]].to_dict(
            "records")
        all_pvals = all_pvals + sig_segments
    pval_df = pd.DataFrame(all_pvals)
    return pval_df


def run_bonferroni(
        exp_df, dim_list, metric_col, group_col="group", conf=0.95
):  # TODO: Support for custom control
    pval_df = get_dim_level_pvals(exp_df, dim_list,metric_col, group_col)
    mtest_result = multipletests(pval_df['pval'], alpha=(1-conf), method='bonferroni')
    pval_df['test_result'] = mtest_result[0]
    pval_df['adjusted_pvals'] = mtest_result[1]
    return pval_df.loc[pval_df['test_result'] == True].to_dict('records')


def run_holm_bonferroni(
        exp_df, dim_list, metric_col, group_col="group", conf=0.95
):  # TODO: Support for custom control
    pval_df = get_dim_level_pvals(exp_df, dim_list, metric_col, group_col)
    mtest_result = multipletests(pval_df['pval'], alpha=(1 - conf), method='holm')
    pval_df['test_result'] = mtest_result[0]
    pval_df['adjusted_pvals'] = mtest_result[1]
    return pval_df.loc[pval_df['test_result'] == True].to_dict('records')


def run_benjamini_hochberg(exp_df, dim_list, metric_col, group_col="group", conf=0.95):
    pval_df = get_dim_level_pvals(exp_df, dim_list, metric_col, group_col)
    mtest_result = multipletests(pval_df['pval'], alpha=(1 - conf), method='fdr_bh')
    pval_df['test_result'] = mtest_result[0]
    pval_df['adjusted_pvals'] = mtest_result[1]
    return pval_df.loc[pval_df['test_result'] == True].to_dict('records')


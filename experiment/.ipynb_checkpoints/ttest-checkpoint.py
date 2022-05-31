import math
import scipy.stats as st
import logging


class Ttest(object):

    @staticmethod
    def run_ttest(sample_c, conv_c, sample_t, conv_t, conf=0.95): # TODO: Add conf intervals
        p_c = conv_c/sample_c
        p_t = conv_t/sample_t
        if p_c <= 0 or p_t <= 0 or p_c > 1 or p_t > 1:
            return False
        se_t = math.sqrt(p_t * (1-p_t)/sample_t)
        se_c = math.sqrt(p_c * (1 - p_c) / sample_c)
        se_diff = math.sqrt(math.pow(se_c, 2)+math.pow(se_t, 2))
        z_val = (p_t - p_c)/se_diff
        pval = 1 - st.norm.cdf(z_val)
        logging.debug(f"Inputs {sample_c}:{conv_c}:{sample_t}:{conv_t}, p-val is {pval}, Z is {z_val }")
        if pval < 1-conf:
            return True, pval
        else:
            return False, pval
        return False, 0

    @staticmethod
    def run_bonferroni(exp_df, dim_col, metric_col, group_col='group', conf=0.95): # TODO: Support for custom control
        # Need to take in a list and then iterate
        # Create helper for reshaping
        dim_list = [dim_col, group_col]
        grouped = (exp_df.groupby(dim_list)
                   .agg({metric_col: ['sum', 'size']})
                   )
        grouped.columns = ["_".join(a) for a in grouped.columns.to_flat_index()]
        grouped = grouped.reset_index()
        control_df = grouped.loc[grouped[group_col] == 'control']
        control_df = control_df.rename(columns=lambda x: "control_" + x if x not in dim_list else x)
        control_df["join_key"], grouped["join_key"] = 1, 1
        final_df = grouped.merge(control_df, how="left", on=["join_key"] + dim_list)
        print(final_df)
        return 1

    def run_holm_bonferroni():
        return 1

    def run_benjamini_hochberg():
        return 1


# test_val = t_test.run_ttest(1000, 100, 1000, 124)
# print(test_val)


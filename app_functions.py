import numpy as np
import pandas as pd


######## WEB APP RECOMMENDATION & SUMMARY
def expected_portfolio_return_evenly_weighted(table_all_current, avail_funds):
    evenly_weighted_expected_returns = []
    for idx in table_all_current.index:
        evenly_weighted_expected_returns.append(
            (table_all_current['return_preds'][idx]) * (avail_funds/len(table_all_current)))
    expected_portfolio_return = sum(evenly_weighted_expected_returns)/avail_funds
    return expected_portfolio_return

def rank_table_by_shrop_ratio_RAR(table_all_current, avail_funds):
    table_all_current['shrop_ratio'] = np.nan
    for idx in table_all_current.index:
        table_all_current['shrop_ratio'][idx] = (
            (expected_portfolio_return_evenly_weighted(table_all_current, avail_funds) - 0.0188) / 
                                                        (table_all_current['prob_default'][idx]))
    table_all_current_ranked = table_all_current.sort_values(by='shrop_ratio',axis=0, ascending=False)
    return table_all_current_ranked

def recommended_loans_ranked_by_shrop_RAR(table_all_current, max_prob_default, min_desired_return, avail_funds):
    rec_table = table_all_current[(table_all_current['prob_default'] <= max_prob_default) & 
                             (table_all_current['return_preds'] >= min_desired_return)]
    rec_table_ranked = rank_table_by_shrop_ratio_RAR(rec_table, avail_funds)
    return rec_table_ranked

def portfolio_prob_default_evenly_weighted(table_all_current, avail_funds):
    evenly_weighted_prob_default = []
    for idx in table_all_current.index:
        evenly_weighted_prob_default.append(
            (table_all_current['prob_default'][idx]) * (avail_funds/len(table_all_current)))
    portfolio_prob_default = sum(evenly_weighted_prob_default)/avail_funds
    return portfolio_prob_default

def portfolio_shrop_ratio_evenly_weighted(table_all_current, avail_funds):
    table_all_current_ranked = rank_table_by_shrop_ratio_RAR(table_all_current, avail_funds)
    evenly_weighted_shrop_ratio = []
    for idx in table_all_current_ranked.index:
        evenly_weighted_shrop_ratio.append(
            (table_all_current['shrop_ratio'][idx]) * (avail_funds/len(table_all_current)))
    portfolio_shrop_ratio = sum(evenly_weighted_shrop_ratio)/avail_funds
    return portfolio_shrop_ratio

def summarize_recommendation(table_all_current, max_prob_default, min_desired_return, avail_funds):
    rec_table_ranked = recommended_loans_ranked_by_shrop_RAR(table_all_current, max_prob_default, min_desired_return, avail_funds)
    port_prob_def = portfolio_prob_default_evenly_weighted(rec_table_ranked,avail_funds)
    port_exp_return = expected_portfolio_return_evenly_weighted(rec_table_ranked, avail_funds)
    port_shrop_ratio = portfolio_shrop_ratio_evenly_weighted(rec_table_ranked,avail_funds)
    max_investable = rec_table_ranked['loan_amnt'].sum() - rec_table_ranked['funded_amnt'].sum()
    return (rec_table_ranked,port_prob_def,port_exp_return,port_shrop_ratio,max_investable)
    

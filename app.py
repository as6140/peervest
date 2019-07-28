from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from app_functions import (expected_portfolio_return_evenly_weighted, rank_table_by_shrop_ratio_RAR, 
                                recommended_loans_ranked_by_shrop_RAR,portfolio_prob_default_evenly_weighted, 
                                portfolio_shrop_ratio_evenly_weighted, summarize_recommendation)
import uuid

app = Flask(__name__, static_url_path="")
table_all_current = pd.read_pickle('table_all_current.pkl')


@app.route("/")
def index():
    """Return the main page."""
    return render_template("index.html")


@app.route("/output", methods=["GET", "POST"])
def output():
    """Return text from user input"""
    data = request.get_json(force=True)
    # every time the user_input identifier
    print(data)
    print(table_all_current.shape)
    print(table_all_current.head())
    min_return_as_float, max_prob_default_as_float = store_user_inputs(data)

    (rec_table_ranked,port_prob_def,port_exp_return,
    port_shrop_ratio,max_investable) = summarize_recommendation(table_all_current, max_prob_default_as_float, 
                                                                min_return_as_float, float(data["avail_funds"]))
    #choose CSV download columns
    full_table = rec_table_ranked[['shrop_ratio','prob_default','return_preds',
    'loan_amnt','funded_amnt','int_rate','fico_range_low','fico_range_high']]

    #round decimals for printed table
    rec_table_ranked['shrop_ratio'] = round(rec_table_ranked['shrop_ratio'],2)
    rec_table_ranked['prob_default'] = round(rec_table_ranked['prob_default'],3)
    rec_table_ranked['return_preds'] = round(rec_table_ranked['return_preds'],3)
    rec_table_ranked['loan_amnt'] = '${:,.2f}'.format(rec_table_ranked['loan_amnt'])
    rec_table_ranked['funded_amnt'] = '${:,.2f}'.format(round(rec_table_ranked['funded_amnt'],0))

    #choose which columns are returned in printed table
    tabl = rec_table_ranked[['shrop_ratio','prob_default','return_preds',
    'loan_amnt','funded_amnt','int_rate','fico_range_low','fico_range_high']].iloc[:10,:]

    html = f'<div># of Investable Loans: {len(table_all_current)}</div>'
    html += f'<div># of Loans That Fit Your Preferences: {len(rec_table_ranked)}</div>'
    html += f'<div>Portfolio Expected Return: {round(port_exp_return,2)*100}%</div>'
    html += f'<div>Portfolio Weighted Average Probability of Default: {round(port_prob_def,2)*100}%</div>'
    html += f'<div>Portfolio Weighted Average Shrop Ratio: {round(port_shrop_ratio,2)}</div>'
    html += f'<div>Maximum Investable in Recommended Loans: ${max_investable}</div>'
    html += tabl.to_html(index=False)
    filename = f'portfolio_{uuid.uuid4()}.csv'
    full_table.to_csv(f'static/port_downloads/{filename}')
    return jsonify({'html': html, 'filename': filename})

def store_user_inputs(data):
    scores = data.values()
    scores_as_float = [float(score) / 100 for score in scores]
    min_return_as_float = scores_as_float[0]
    max_prob_default_as_float = scores_as_float[1]
    return min_return_as_float, max_prob_default_as_float

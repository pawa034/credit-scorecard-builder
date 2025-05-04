import pandas as pd
import numpy as np
from datetime import datetime
import os
import statsmodels.api as sm

from scorecard.modeling import ks_calculator, gen_ks_calculator_groups, modeling2, predict_method, p_val_calc


def re_tune_models(result=None, oot=None, data=None, varlist=None, target=None, path=None):
    """Perform iterative model tuning by removing variables and evaluating performance
    
    Parameters:
    -----------
    result : statsmodels model
        Initial model result
    oot : pd.DataFrame
        Out-of-time dataset
    data : pd.DataFrame
        Development dataset
    varlist : list
        List of variables
    target : str
        Target column name
    path : str
        Path to save results
        
    Returns:
    --------
    pd.DataFrame
        Summary of model iterations
    """
    if data is None or target is None:
        raise ValueError("data and Target are required inputs.")
        
    # Extract model statistics
    cov = result.cov_params()
    sum1 = pd.DataFrame({
        "Z_SCORE": result.params / np.sqrt(np.diag(cov)),
        'pvals': result.pvalues,
        'p_values': (result.pvalues < 0.05)
    }).reset_index()
    
    sum1['z_abs'] = abs(sum1['Z_SCORE'])
    sum2 = pd.DataFrame(sum1.sort_values(by='pvals', ascending=True).reset_index().drop('level_0', axis=1))
    sum2 = sum2.rename(columns={'index': 'Vars'})
    
    # Display variable statistics
    with pd.option_context('display.max_columns', None):
        print(sum2.to_string())
    
    # Determine number of iterations
    total_vars = len(sum2['Vars'].unique())
    min_iter = int(total_vars - 5)
    original_vars = list(sum2['Vars'])
    vars1 = original_vars
    
    # Initialize tracking lists
    model_vars = []
    iteration = []
    dev_max_ks = []
    dev_ks_row = []
    flag_dev_T = []
    oot_max_ks = []
    oot_ks_row = []
    flag_oot_T = []
    best_decile_dev = []
    best_decile_oot = []
    models = []
    
    # Initialize data
    data['prob'] = ''
    oot['prob'] = ''
    
    # Print iteration information
    print(f'Total Variables Selected for Modelling: **{total_vars}**')
    print('\n')
    print(f'Total No of Iterations Modeller will run: **{min_iter}**')
    print('\n')
    print(f'Model summary is stored in result_summary_all_iterations.txt for iteration 0 to **{min_iter}**')
    
    # Iterate through variable selection
    for i in range(min_iter + 1):
        # Fit model with current variables
        result = modeling2(n=total_vars, varlist=vars1, data=data, target=target)
        
        # Get significant variables
        var_selected = p_val_calc(result)
        model_vars.append(var_selected)
        
        # Refit model with significant variables
        result = modeling2(n=total_vars, varlist=var_selected, data=data, target=target)
        models.append(result)
        
        # Generate predictions
        data = predict_method(data, var_selected, result)
        oot = predict_method(oot, var_selected, result)
        
        # Calculate KS statistics
        max_ks_dev, ks_row_dev, flag_dev, ks_table_dev = ks_calculator(data=data, target=target, prob='prob')
        max_ks_oot, ks_row_oot, flag_oot, ks_table_oot = ks_calculator(data=oot, target=target, prob='prob')
        
        # Find optimal grouping if monotonicity is not achieved
        decile_data = 0
        decile_oot = 0
        grps = 0
        grps1 = 0
        
        if flag_dev == False:
            for k in range(10, 5, -1):
                ks_groups = gen_ks_calculator_groups(data=data, target=target, prob='prob', groups=k)
                if (grps == 0) and ks_groups['badrate'].is_monotonic_decreasing:
                    grps = k
                    decile_data = int(grps)
                    # Store KS table
                    ks_df = ks_groups[['min_prob', 'max_prob', 'events', 'count', 'event_rate', 
                                      'cum_eventrate', 'KS']]
        
        if flag_oot == False:
            for j in range(10, 5, -1):
                ks_groups = gen_ks_calculator_groups(data=oot, target=target, prob='prob', groups=j)
                if (grps1 == 0) and ks_groups['badrate'].is_monotonic_decreasing:
                    grps1 = j
                    decile_oot = int(grps1)
                    # Store KS table
                    ks_df = ks_groups[['min_prob', 'max_prob', 'events', 'count', 'event_rate', 
                                      'cum_eventrate', 'KS']]
        
        # Store results for this iteration
        best_decile_dev.append(decile_data)
        best_decile_oot.append(decile_oot)
        grps = 0
        grps1 = 0
        dev_max_ks.append(max_ks_dev)
        dev_ks_row.append(ks_row_dev)
        flag_dev_T.append(flag_dev)
        oot_max_ks.append(max_ks_oot)
        oot_ks_row.append(ks_row_oot)
        flag_oot_T.append(flag_oot)
        
        # Clean up for next iteration
        del data['prob']
        del oot['prob']
        
        # Format KS table if available
        if max_ks_dev != 'FAILED':
            ks_table_dev = ks_table_dev.reset_index()
            ks_table_dev = ks_table_dev[['Decile', 'badrate', 'events', 'cum_eventrate', 'count', 'KS']]
        
        # Track iteration
        iteration.append(i)
        
        # Remove the least significant variable for next iteration
        if len(vars1) > 0:
            vars1.pop(-1)
    
    # Write model summaries to file
    print('\n\n')
    if path:
        output_file = os.path.join(path, 'result_summary_all_iterations.txt')
        with open(output_file, 'w') as file:
            for i, model in enumerate(models):
                file.write(f"\n\nSummary for Model {i + 1}:\n")
                file.write(model.summary().as_text())
        print('Models are Successfully Stored'.center(100, '-'))
    
    # Create output summary DataFrame
    print(f'**{min_iter}** Models Built, Below is Summary for the Models')
    output = pd.DataFrame({
        'iter': iteration,
        'dev_max_ks': dev_max_ks,
        'dev_ks_row': dev_ks_row,
        'flag_dev': flag_dev_T,
        'oot_max_ks': oot_max_ks, 
        'oot_ks_row': oot_ks_row,
        'flag_oot': flag_oot_T,
        'best_decile_dev': best_decile_dev,
        'best_decile_oot': best_decile_oot,
        'models': models, 
        "Vars Used ": model_vars
    })
    
    print('\n')
    
    return output


def best_model_selector(output, data, oot, target):
    """Select best model based on performance metrics
    
    Parameters:
    -----------
    output : pd.DataFrame
        Model evaluation results
    data : pd.DataFrame
        Development dataset
    oot : pd.DataFrame
        Out-of-time dataset
    target : str
        Target column name
        
    Returns:
    --------
    tuple
        Selected model, development data with predictions, OOT data with predictions,
        development KS table, OOT KS table
    """
    # Initialize
    select_model = ''
    
    # Filter out failed models
    output = output[output['dev_max_ks'] != 'FAILED']
    
    # Calculate metrics to help select best model
    output['decileSum'] = output['best_decile_dev'] + output['best_decile_oot']
    output['decileSum'] = output['decileSum'].astype(int)
    output['dif'] = abs(output['dev_max_ks'] - output['oot_max_ks'])
    
    # Display results
    ex_columns = ['iter', 'models', 'Vars Used ', 'decileSum', 'dif']
    output_excluded = output.drop(columns=ex_columns)
    with pd.option_context('display.max_columns', None):
        print(output_excluded.to_string())
    
    # First try to find a model with good performance on both dev and OOT
    flag_alpha = 0
    true_sets = output[(output['flag_dev'] == True) & (output['flag_oot'] == True)]
    semi_true_sets = output[output['flag_dev'] == True]
    
    if not true_sets.empty:
        flag_alpha = 1
        count = true_sets.shape[0]
        
        if count > 1:
            # Find model with minimum difference between dev and OOT KS
            min_diff = true_sets['dif'].min()
            best_model = true_sets[true_sets['dif'] == min_diff]
            
            if best_model.shape[0] > 1:
                # If multiple models tie, choose the one with highest dev KS
                m1 = best_model['dev_max_ks'].max()
                best_model = best_model[best_model['dev_max_ks'] == m1]
                
                if best_model.shape[0] > 1:
                    select_model = best_model.tail(1).reset_index()['models'][0]
                    try:
                        print(best_model['models'][0].summary)
                    except KeyError:
                        print("Error: Model summary not available for the selected model.")
                else:
                    select_model = best_model.reset_index()['models'][0]
                    print(best_model.reset_index()['models'][0].summary())
            else:
                print(best_model.reset_index()['models'][0].summary())
                select_model = best_model.reset_index()['models'][0]
        else:
            select_model = true_sets.reset_index()['models'][0]
            print(select_model.summary())
    else:
        # If no model performs well on both dev and OOT, choose based on decile sum
        max_sum = output['decileSum'].max()
        best_model = output[output['decileSum'] == max_sum]
        
        if not best_model.empty:
            if len(best_model) > 1:
                # If multiple models tie, choose the most recent one
                max_iter = best_model['iter'].max()
                best_model = best_model[best_model['iter'] == max_iter]
                select_model = best_model.reset_index()['models'][0]
                print(best_model.reset_index()['models'][0].summary())
            else:
                select_model = best_model.reset_index()['models'][0]
                print(best_model.reset_index()['models'][0].summary())
        else:
            print("No suitable model found")
            return None, data, oot, None, None
    
    # Get model variables (excluding constant)
    mod_vars = [col for col in list(select_model.cov_params().index) if 'const' not in col]
    
    # Generate predictions
    data = predict_method(data, mod_vars, select_model)
    oot = predict_method(oot, mod_vars, select_model)
    
    # Set default number of groups for KS tables
    if flag_alpha == 0:
        if not best_model.empty:
            dev_grp = best_model.reset_index()['best_decile_dev'][0]
            oot_grp = best_model.reset_index()['best_decile_oot'][0]
        else:
            dev_grp = 10
            oot_grp = 10
    else:
        dev_grp = 10
        oot_grp = 10
    
    # Generate KS tables
    print('\n')
    print('KS TABLE FOR BEST MODEL DEV SET IS AS BELOW '.center(100, '-'))
    print('\n')
    dev_ks = gen_ks_calculator_groups(data=data, target=target, prob='prob', groups=dev_grp)
    print(dev_ks[['min_prob', 'max_prob', 'events', 'count', 'event_rate', 'cum_eventrate', 'KS']])
    
    print('\n')
    print('KS TABLE FOR BEST MODEL TEST SET IS AS BELOW '.center(100, '-'))
    print('\n')
    test_ks = gen_ks_calculator_groups(data=oot, target=target, prob='prob', groups=oot_grp)
    print(test_ks[['min_prob', 'max_prob', 'events', 'count', 'event_rate', 'cum_eventrate', 'KS']])
    
    return (
        select_model, 
        data, 
        oot, 
        gen_ks_calculator_groups(data=data, target=target, prob='prob', groups=dev_grp),
        gen_ks_calculator_groups(data=oot, target=target, prob='prob', groups=oot_grp)
    )


def score_analyzer(test1, scorebins, id_col=None, target=None, bad=None):
    """Analyze score distribution across specified bins
    
    Parameters:
    -----------
    test1 : pd.DataFrame
        DataFrame with scores
    scorebins : list
        Score cutoff points for bins
    id_col : str
        ID column name
    target : str
        Target column name
    bad : str
        Additional bad event column name
        
    Returns:
    --------
    pd.DataFrame
        Summary of score distribution
    """
    derf = test1.copy()
    rang1 = []
    final_bins = []
    no_grps = len(scorebins) + 1
    
    # Create bin labels
    fst = '<' + str(scorebins[0])
    last = '>' + str(scorebins[no_grps - 2])
    
    for i in range(no_grps - 1):
        if i != 0 and i != no_grps - 1:
            rang1.append(str(scorebins[i - 1]) + '_' + str(scorebins[i]))
    
    final_bins.append(fst)
    final_bins.extend(rang1)
    final_bins.append(last)
    
    # Create score buckets
    conditions = []
    choices = []
    
    # First bucket: scores less than first cutoff
    conditions.append(derf['score'] < scorebins[0])
    choices.append(final_bins[0])
    
    # Middle buckets
    for i in range(len(scorebins) - 1):
        conditions.append(
            (derf['score'] >= scorebins[i]) & (derf['score'] < scorebins[i + 1])
        )
        choices.append(final_bins[i + 1])
    
    # Last bucket: scores greater than or equal to last cutoff
    conditions.append(derf['score'] >= scorebins[-1])
    choices.append(final_bins[-1])
    
    derf['scoreBucket'] = np.select(conditions, choices)
    
    print(derf['scoreBucket'].value_counts())
    
    # Aggregate statistics by score bucket
    try:
        if bad:
            summary = derf[[id_col, target, bad, 'scoreBucket']].groupby(['scoreBucket']).agg({
                id_col: 'count',
                target: 'sum',
                bad: 'sum'
            }).reset_index()
            
            summary = summary.rename(columns={id_col: '#CNT_LAN', target: 'BadCnt', bad: 'FEMICNT'})
            summary['BadRates'] = summary['BadCnt'] / summary['#CNT_LAN']
            summary['FEMiRates'] = summary['FEMICNT'] / summary['#CNT_LAN']
            summary['PopPercentage'] = summary['#CNT_LAN'] / np.sum(summary['#CNT_LAN'])
            final_sum = summary.sort_values(by='FEMiRates', ascending=False)
        else:
            summary = derf[[id_col, target, 'scoreBucket']].groupby(['scoreBucket']).agg({
                id_col: 'count',
                target: 'sum'
            }).reset_index()
            
            summary = summary.rename(columns={id_col: '#CNT_LAN', target: 'BadCnt'})
            summary['BadRates'] = summary['BadCnt'] / summary['#CNT_LAN']
            summary['PopPercentage'] = summary['#CNT_LAN'] / np.sum(summary['#CNT_LAN'])
            final_sum = summary.sort_values(by='BadRates', ascending=False)
            
        return final_sum
        
    except Exception as e:
        print(f"Error in score_analyzer: {str(e)}")
        return pd.DataFrame() 
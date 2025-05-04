import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

def scorecard_builder(sc, data, columns_to_remove=None):
    """
    Build a scorecard model from data
    
    Parameters:
    -----------
    sc : ScoreCardRisk
        Instance of ScoreCardRisk class
    data : pd.DataFrame
        Input DataFrame
    columns_to_remove : list, optional
        List of column names to remove
        
    Returns:
    --------
    tuple
        (best_model, path, cols_to_use, model_vars, data)
    """
    # Suppress pandas warnings
    import warnings
    import pandas as pd
    pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning
    warnings.filterwarnings('ignore', category=Warning)  # Suppress all warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Standardize column names to uppercase
    data.columns = data.columns.str.upper()
    
    # Process columns to remove
    upper_list = []
    if columns_to_remove:
        for var in columns_to_remove:
            upper_list.append(var.upper())
    
    # Create output directory
    path = sc.volumeAndDir()
    
    # Remove specified columns
    data = data.drop(columns=upper_list, errors='ignore')
    
    # Get target, ID, and feature columns
    Target, Id, num_cols, cat_cols = sc.DataSelector(data)
    
    # Print header information
    imac_text = '**IMAC APPLICATION IN ACTION**'
    print(f"{imac_text}".center(100, '-'))
    print('\n')
    print(f"All IMAC meta files will be dumped in directory : {path}")
    print('\n')
    
    # Calculate bad rate
    bdrate = data[data[Target] == 1].shape[0] / data.shape[0]
    t1 = datetime.now()
    
    # Determine bin cutoff based on bad rate
    mydict = pd.DataFrame({
        'BadRate_range': ['0-5', '5-10', '10-20', '20-30', '30-40', '40+'],
        'max_rate': [0.05, 0.1, 0.2, 0.3, 0.4, 0.75],
        'bin_cut_off': [0.001, 0.005, 0.01, 0.015, 0.016, 0.018]
    })
    mydict = mydict[mydict['max_rate'] >= bdrate]
    bin_cut_off = float(mydict['bin_cut_off'].head(1))
    
    # Print data statistics
    imac_text = '**DATA STATS**'
    print(f"{imac_text}".center(100, '-'))
    print('\n')
    print(f'Data Size  :  {data.shape[0]}')
    print(f'Response Variable : {sc.Target}')
    print(f'Unique Id : {sc.Id}')
    print('BadRate is: {:.4f}%'.format(bdrate * 100))
    
    print('\n')
    print('**Audit Stats**'.center(100, '-'))
    print('\n')
    
    # Generate data quality reports
    summary = sc.data_report(data, num_cols)
    report = sc.report_builder(summary)
    print(f'Variables Included : **{len(set(num_cols + cat_cols))}**')
    
    t2 = datetime.now()
    VarReportFinal = pd.DataFrame({'VariableList': list(num_cols + cat_cols)})
    
    # Filter variables based on data quality
    selectedVars = list(sc.Selector_sanity(report, all_var=num_cols + cat_cols))
    VarReportFinal['VarSanity'] = VarReportFinal['VariableList'].apply(lambda x: 1 if x in selectedVars else 0)
    
    # Save excluded variables summary
    removedVarsPostSanity = list(set(list(summary.index)) - set(selectedVars))
    summary[summary.index.isin(removedVarsPostSanity)].to_csv(path + '/' + 'SummarySanityexluded.csv')
    
    # Filter variables with low variance
    print('Removing Variables with zero or constant Variance')
    sel = VarianceThreshold(threshold=0.02)
    sel.fit(data[num_cols].fillna(0))
    
    FeatureRemoved = [x for x in data[num_cols].columns if x not in data[num_cols].columns[sel.get_support()]]
    selectedVars = list(set(selectedVars) - set(FeatureRemoved))
    print(f"Variables Selected : {len(selectedVars)}")
    
    print('\n')
    print('Approx run time in minutes    : %0.2f' % (float(pd.Timedelta(t2 - t1).total_seconds() / 60)))
    print('\n')
    
    # Make sure target is not in selectedVars
    selectedVars = [vars for vars in selectedVars if Target not in vars]
    selectedVars.append(Target)
    
    # Information Value (IV) calculation
    print('**IV Table Creation in Progress**'.center(100, '-'))
    t1 = datetime.now()
    object_dict = sc.iv_maker(uid=Id, Target=Target, data=data, variable=selectedVars)
    sc.object_dict = object_dict
    
    # Transform data using optimal binning
    woe_df = sc.transformer_other_pandas(data, object_dict, uid=Id, target=Target, metric='woe')
    
    # Get IV values
    iv = sc.get_iv_data(object_dict)
    iv.to_csv(path + '/' + 'IVALL.csv')
    
    # Filter variables based on IV
    new_iv = iv[iv['IV'] > sc.iv_cutoff]
    selectedVarsPostIv = new_iv['Variable'].unique().tolist()
    
    print('\n')
    print(f'Default IV Threshold: {round(sc.iv_cutoff * 100, 2)}%')
    print(f'Variables Included : **{len(selectedVars) - 1}**')
    print(f'Variables Selected after IV cutoff : **{len(selectedVarsPostIv)}**')
    
    print('\n')
    t2 = datetime.now()
    print('Approx run time in minutes    : %0.2f' % (float(pd.Timedelta(t2 - t1).total_seconds() / 60)))
    
    # Handle too few variables
    if len(selectedVarsPostIv) <= 2:
        print('Variable left less than so we proceed directly to Next step. Change in feature space: **1%**')
        selectedVarsPostIv = selectedVars
    
    # Make sure target is in selectedVarsPostIv
    selectedVarsPostIv = [vars for vars in selectedVarsPostIv if Target not in vars]
    selectedVarsPostIv.append(Target)
    
    print('\n')
    
    # Feature importance calculation
    print('**PERMUTATION IMPORTANCE ALGORITHM**'.center(100, '-'))
    print('\n')
    print(f'Variables Included : **{len(selectedVarsPostIv) - 1}**')
    
    t1 = datetime.now()
    
    # Split variables by type
    setVarsPostIv = set(selectedVarsPostIv)
    s2 = set(num_cols)
    s3 = set(cat_cols)
    numericVarSelectedVarsPostIv = list(setVarsPostIv & s2)
    CategoricalVarSelectedVarsPostIv = list(setVarsPostIv & s3)
    
    # Chi-square test for categorical variables
    print(f"Variables before Chi-Square(Categorical) : {len(CategoricalVarSelectedVarsPostIv)}")
    chi_result_df, CatVarSelectedVarsPostIv_ChiSq = sc.chisq_feature_selection(data, CategoricalVarSelectedVarsPostIv)
    print(f"Variables after Chi-Square(Categorical) : {len(CatVarSelectedVarsPostIv_ChiSq)}")
    
    # Random Forest importance for numerical variables
    numeric_features = [vars for vars in numericVarSelectedVarsPostIv if Target not in vars]
    if numeric_features:
        rf_pimp_result_df, VarsPostPimpRf = sc.rf_pimp_feature_importance(
            data=data,
            Id=sc.Id,
            Target=sc.Target,
            selected_features=numeric_features,
            factor=1.5,
            model_params=None
        )
        VarsPostPimpRf.extend(CategoricalVarSelectedVarsPostIv)
    else:
        VarsPostPimpRf = CategoricalVarSelectedVarsPostIv
    
    # Update variable tracking
    VarReportFinal['VarRFpimp'] = VarReportFinal['VariableList'].apply(lambda x: 1 if x in VarsPostPimpRf else 0)
    print(f'Variables Selected after rf_pimp : **{len(VarsPostPimpRf)}**')
    
    print('\n')
    t2 = datetime.now()
    print('Approx run time in minutes    : %0.2f' % (float(pd.Timedelta(t2 - t1).total_seconds() / 60)))
    print('\n')
    
    # Save removed variables
    pd.Series(list(set(selectedVarsPostIv) - set(VarsPostPimpRf))).to_csv(path + '/' + 'RFremvoedVars.csv')
    
    # Make sure target is in VarsPostPimpRf
    if Target not in VarsPostPimpRf:
        VarsPostPimpRf.append(Target)
    
    # Correlation filter
    t1 = datetime.now()
    print('**Correlation Stats**'.center(100, '-'))
    print('\n')
    print(f'Default Cutoff = **{sc.corr_cutoff}**')
    print(f'Variables Included : **{len(VarsPostPimpRf) - 1}**')
    
    # Split variables by type
    s1 = set(VarsPostPimpRf)
    numericVarsPostPimpRf = list(s1 & s2)
    categoricalVarsPostPimpRf = list(s1 & s3)
    
    # Filter by correlation
    uncorrelated_vars, removed_vars = sc.correlationFilter(data, numericVarsPostPimpRf, sc.corr_cutoff, sc.Target)
    uncorrelated_vars += categoricalVarsPostPimpRf
    
    # Handle too few variables
    if len(uncorrelated_vars) <= 2:
        print('Variable left less than so we proceed directly to Next step. Change in feature space: **1%**')
        uncorrelated_vars = VarsPostPimpRf
    
    # Update variable tracking
    VarReportFinal['VarCorr'] = VarReportFinal['VariableList'].apply(lambda x: 1 if x in uncorrelated_vars else 0)
    print(f'Variables Selected after Correlation : **{len(uncorrelated_vars)}**')
    
    print('\n')
    t2 = datetime.now()
    print('Approx run time in minutes    : %0.2f' % (float(pd.Timedelta(t2 - t1).total_seconds() / 60)))
    
    # Save removed variables
    pd.Series(list(set(VarsPostPimpRf) - set(uncorrelated_vars))).to_csv(path + '/' + 'CorrremvoedVars.csv')
    
    # SHAP feature elimination
    print('\n')
    print('**Backward Features Elimination Using SHAP**'.center(100, '-'))
    print('\n')
    print(f'Variable Included:**{len(uncorrelated_vars)}**')
    
    # Split variables by type
    setVarsPostCorrelation = set(uncorrelated_vars)
    numVarsPostCorrelation = list(setVarsPostCorrelation & s2)
    catVarsPostCorrelation = list(setVarsPostCorrelation & s3)
    
    t1 = datetime.now()
    
    # Apply SHAP feature elimination
    if numVarsPostCorrelation:
        shapSelectedFeatures, shapReport = sc.shapRFECV(data, numVarsPostCorrelation)
        shapSelectedFeatures.extend(catVarsPostCorrelation)
    else:
        shapSelectedFeatures = catVarsPostCorrelation
    
    # Final selected features
    binnableFeats = shapSelectedFeatures
    
    # Handle too few variables
    if len(binnableFeats) <= 2:
        print('Variable left less than so we proceed directly to Next step. Change in feature space: **1%**')
        binnableFeats = shapSelectedFeatures
    
    # Update variable tracking
    VarReportFinal['VarEnsemble'] = VarReportFinal['VariableList'].apply(lambda x: 1 if x in binnableFeats else 0)
    print(f'Variables Selected after SHAP : **{len(binnableFeats)}**')
    
    # Save removed variables
    pd.Series(list(set(shapSelectedFeatures) - set(binnableFeats))).to_csv(path + '/' + 'StepwiseremvoedVars.csv')
    
    t2 = datetime.now()
    print('\n')
    print('Approx run time in minutes    : %0.2f' % (float(pd.Timedelta(t2 - t1).total_seconds() / 60)))
    print('\n')
    
    # Variable selection completed
    print('**VAR SELECTION PROCESS COMPLETED**'.center(100, '-'))
    print('\n')
    print('**IMAC MODEL DEVELOPMENT STARTS**'.center(100, '-'))
    
    t1 = datetime.now()
    print('\n')
    print('**Model Summary**'.center(100, '-'))
    print('\n')
    print('**ROC SCORE** Displayed Below for Each Model'.center(100, '-'))
    print('\n')
    print('Best **Model Pickle** Object dumped Along with List Of Variables'.center(100, '-'))
    
    # Final feature list
    model_vars = binnableFeats
    model_num_cols = list(set(binnableFeats) & set(num_cols))
    model_cat_cols = list(set(binnableFeats) & set(cat_cols))
    
    # Prepare logistic data
    woe_var_list = [Target]
    for var in model_vars:
        if var is None:
            continue
        woe_var_list.append("woe_" + var.upper())
    
    woe_data = woe_df[woe_var_list]
    train_woe, test_woe = train_test_split(woe_data, test_size=sc.size)
    
    # Prepare ML data
    ml_data = data[binnableFeats + [Target]]
    train_ml, test_ml = sc.mlDataImputer(ml_data, model_num_cols, model_cat_cols)
    ml_cols = [col for col in train_ml.columns if col not in [sc.Target]]
    
    # ML model training
    FeatImpList, model, name, df_dev, df_oot = sc.MlSummary(train_ml, test_ml, binnableFeats=ml_cols, Target=Target)
    
    # Calculate KS for ML model
    print('DEV KS TABLE'.center(100, '-'))
    max_ks, ks_decile, flag_dev_ml, ml_ks_dev = sc.Gen_ks_calculator(data=df_dev, target=Target, prob='prob_ml')
    print(ml_ks_dev[['min_prob', 'max_prob', 'events', 'count', 'event_rate', 'cum_eventrate', 'KS']])
    
    max_ks, ks_decile, flag_oot_ml, ml_ks_oot = sc.Gen_ks_calculator(data=df_oot, target=Target, prob='prob_ml')
    print('OOT KS TABLE'.center(100, '-'))
    print(ml_ks_oot[['min_prob', 'max_prob', 'events', 'count', 'event_rate', 'cum_eventrate', 'KS']])
    
    # Save KS tables
    try:
        if not ml_ks_dev.empty:
            ml_ks_dev.to_csv(path + '/' + 'DEVKSML.csv')
        if not ml_ks_oot.empty:
            ml_ks_oot.to_csv(path + '/' + 'OOTKSML.csv')
    except:
        print('No Stable KS for ML model could be found')
        print('\n')
    
    # Save ML model
    filename = str(name) + '.' + 'sav'
    with open(os.path.join(path, filename), 'wb') as temp_file:
        pickle.dump(model, temp_file)
    
    # Save feature importance
    FeatImpList.to_csv(path + '/' + 'FeatureImportance.csv')
    
    print('\n')
    t2 = datetime.now()
    print('Approx run time in minutes ' + str(float(pd.Timedelta(t2 - t1).total_seconds() / 60)))
    print('\n')
    
    # Logistic modeling
    print('**Logistic Modeling Starts**'.center(100, '-'))
    print('\n')
    t1 = datetime.now()
    
    # Store columns to use for logistic
    sc.cols_to_use = [feat for feat in woe_var_list if Target not in feat]
    
    # Initial model
    result = sc.Modelling(no_of_iter=10, varlist=sc.cols_to_use, data=train_woe, Target=Target)
    
    # Model tuning
    output = sc.ReTuneModels(result=result, oot=test_woe, data=train_woe, varlist=sc.cols_to_use, Target=Target, path=path)
    output.to_csv(path + '/' + 'output.csv')
    
    # Select best model
    print('Best Model As per IMAC is as below '.center(100))
    print('\n')
    bestModel, data_pred, oot_pred, ks_table_dev, ks_table_oot = sc.BestModelSelector(output, train_woe, test_woe, Target)
    
    # Save variable report
    VarReportFinal.to_csv(path + '/' + 'VarReport.csv')
    
    print('\n')
    print('**To run Next Iteration on your Data Please reload the Data**'.center(100, '-'))
    print('\n')
    print('**IMAC ENGINE STOPPED**'.center(100, '-'))
    
    return bestModel, path, sc.cols_to_use, model_vars, data_pred 
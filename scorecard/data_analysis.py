import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def data_report(data, numerical_cols):
    """Generate a summary report for numerical data
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing numerical data
    numerical_cols : list
        List of column names with numerical data
        
    Returns:
    --------
    pd.DataFrame
        A summary report with statistics, fill rates, and missing values
    """
    data = data[numerical_cols]
    
    # Calculate fill rates
    fill_rate = pd.DataFrame(data.count(), columns=["Fill Rate"]).rename_axis('Variable')
    null_value_count = pd.DataFrame(data.isnull().sum(), columns=["N_Miss"]).rename_axis('Variable')
    
    # Calculate descriptive statistics
    proc_means = pd.DataFrame(
        data.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]).transpose()
    ).rename_axis('Variable')
    
    # Merge fill rates and null counts
    fill_rate = pd.merge(fill_rate, null_value_count, how='left', on='Variable')
    fill_rate['Fill_Rate_%'] = round((fill_rate['Fill Rate'] / data.shape[0]) * 100, 2)
    fill_rate['N_miss_%'] = round((fill_rate['N_Miss'] / data.shape[0]) * 100, 2)
    fill_rate.sort_values('N_miss_%', ascending=False, inplace=True)
    
    # Create summary by merging descriptive stats with null counts
    summary = pd.merge(proc_means, null_value_count, how='left', on='Variable')
    summary['Fill_Rate_%'] = round((summary['count'] / data.shape[0]) * 100, 2)
    summary['N_miss_%'] = round((summary['N_Miss'] / data.shape[0]) * 100, 2)
    summary.sort_values('N_miss_%', ascending=False, inplace=True)
    
    # Reorder columns
    new_order = [17, 18, 0, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    summary = summary[summary.columns[new_order]]
    
    return summary


def report_builder(summary):
    """Build a report based on a summary of numerical data
    
    Parameters:
    -----------
    summary : pd.DataFrame
        Summary report of numerical data
        
    Returns:
    --------
    pd.DataFrame
        A report with checks for missing values, distribution, special conditions, variance
    """
    columns = list(summary.index)
    report = pd.DataFrame(
        columns=['miss_check', 'dist_check', 'has_spcial', 'variance_check', 'miss_yes'],
        index=columns
    )
    report = report.fillna(0)
    
    for col in columns:
        # Check for high missing percentage
        if summary.loc[col, 'N_miss_%'] >= 70:
            report.loc[col, 'miss_check'] = 1
            
        # Check for distribution issues
        if summary.loc[col, '90%'] == 0:
            if summary.loc[col, '90%'] == summary.loc[col, '50%']:
                report.loc[col, 'dist_check'] = 1
                
        # Check for special values
        if summary.loc[col, 'mean'] <= 0:
            report.loc[col, 'has_spcial'] = 1
            
        # Check for variance issues
        if ((summary.loc[col, '99%'] == 0 and summary.loc[col, '90%'] == 0) or
            ((summary.loc[col, '20%'] == summary.loc[col, '30%']) and
             (summary.loc[col, '30%'] == summary.loc[col, '40%']) and
             (summary.loc[col, '40%'] == summary.loc[col, '50%']) and
             (summary.loc[col, '50%'] == summary.loc[col, '60%']))):
            report.loc[col, 'variance_check'] = 1
            
        # Flag variables with >5% missing values
        if summary.loc[col, 'N_miss_%'] > 5:
            report.loc[col, 'miss_yes'] = 1
            
    return report


def selector_sanity(report, all_var=None):
    """Perform variable selection based on sanity checks
    
    Parameters:
    -----------
    report : pd.DataFrame
        Report containing data quality checks
    all_var : list, optional
        List of all variable names
        
    Returns:
    --------
    list
        List of selected variables after removing those flagged by sanity checks
    """
    # Find variables to remove that have both variance and distribution issues
    vars_to_remove = list(report[(report['variance_check'] == 1) & 
                                 (report['dist_check'] == 1)].index)
    
    # Remove "flag" variables from the removal list
    vars_to_remove_1 = [var for var in vars_to_remove if 'flag' not in var.lower()]
    
    # Return variables after removing problematic ones
    selected_vars = list(set(all_var) - set(vars_to_remove))
    
    return selected_vars


def correlation_filter(data, varlist, corr_cutoff, target=None):
    """Filter variables based on correlation
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the dataset
    varlist : list
        List of variable names to consider
    corr_cutoff : float
        Correlation threshold
    target : str, optional
        Name of the target variable
        
    Returns:
    --------
    tuple
        Selected variables and dropped variables
    """
    variable_dropped = []  # List to store variables to be dropped
    
    if target:
        target_correlations = data[varlist].corrwith(data[target])
        
        for i in range(len(varlist)):
            for j in range(i + 1, len(varlist)):
                var1 = varlist[i]
                var2 = varlist[j]
                corr_var1_var2 = data[var1].corr(data[var2])
                
                if abs(corr_var1_var2) > corr_cutoff:
                    corr_var1_target = abs(target_correlations[var1])
                    corr_var2_target = abs(target_correlations[var2])
                    
                    # Keep the variable with higher correlation to target
                    if corr_var1_target > corr_var2_target:
                        variable_dropped.append(var2)
                    else:
                        variable_dropped.append(var1)
        
        # Remove duplicates from variable_dropped
        variable_dropped = list(set(variable_dropped))
        
        # Create list of selected variables
        selected_variables = [var for var in varlist if var not in variable_dropped]
        
        return selected_variables, variable_dropped
    
    return varlist, []


def chisq_feature_selection(data, cat_cols, target):
    """Select categorical features using chi-square test
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    cat_cols : list
        List of categorical columns
    target : str
        Target column name
        
    Returns:
    --------
    tuple
        Chi-square result DataFrame and selected categorical columns
    """
    from sklearn.feature_selection import chi2
    from sklearn.preprocessing import LabelEncoder
    
    if len(cat_cols) > 1:
        cat_df = data[cat_cols].copy()  # Create explicit copy to avoid SettingWithCopyWarning
        X = cat_df  # Independent variables
        y = data[target].astype(str)  # Target variable
        
        # Encode categorical variables
        label_encoder = LabelEncoder()
        for column in X.columns:
            X.loc[:, column] = label_encoder.fit_transform(X[column])
        
        # Suppress warnings during chi-square calculation
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Calculate chi-square statistics
            chi_scores, p_values = chi2(X, y)
        
        # Create a DataFrame with results
        chi_result_df = pd.DataFrame({
            'Variable': X.columns, 
            'Chi-Squared': chi_scores, 
            'p-value': p_values
        })
        
        # Filter variables with p-value <= 0.05
        cat_cols_after_chi2 = chi_result_df[chi_result_df['p-value'] <= 0.05]['Variable'].tolist()
        
        return chi_result_df, cat_cols_after_chi2
    
    return pd.DataFrame(), cat_cols 
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from scorecard.data_analysis import data_report, report_builder, selector_sanity, correlation_filter, chisq_feature_selection
from scorecard.feature_engineering import (
    iv_maker, get_iv_data, transformer_other_pandas, 
    rf_pimp_feature_importance, shap_rfecv, ml_data_imputer
)
from scorecard.modeling import (
    modeling, modeling2, predict_method, ml_summary, 
    gen_ks_calculator, gen_ks_calculator_groups, log_loss_calc
)
from scorecard.tuning import re_tune_models, best_model_selector
from scorecard.utils import create_directory
from scorecard.scorecard_builder import scorecard_builder


class ScoreCardRisk:
    """
    ScoreCardRisk is a comprehensive class for building and evaluating scorecard models
    for credit risk assessment.
    """
    
    def __init__(self):
        """Initialize ScoreCardRisk with default parameters"""
        # Thresholds and parameters
        self.iv_cutoff = 0.01
        self.corr_cutoff = 0.65
        self.split = 0.3
        self.alpha = 0.05
        self.max_bin = 20
        self.force_bin = 3
        self.depth = 5
        self.TreeDepth = 100
        self.size = 0.3
        
        # Target and ID variables
        self.Target = None
        self.Id = None
        
        # Other variables
        self.ml_ks_dev = None
        self.m1_ks_oot = None
        self.cols_to_use = None
        self.object_dict = {}
        
    def formula_from_cols(self, df, y):
        """Create formula string from column names for statsmodels
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        y : str
            Target variable name
            
        Returns:
        --------
        str
            Formula string
        """
        return y + ' ~ ' + ' + '.join([col for col in df.columns if col not in ([y, 'weights'])])
    
    def predict_method(self, data, vars1, result):
        """Add predicted probabilities to data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        vars1 : list
            Variables to use for prediction
        result : statsmodels model
            Fitted model
            
        Returns:
        --------
        pd.DataFrame
            Data with predictions
        """
        import statsmodels.api as sm
        
        x = data[vars1]
        x = sm.add_constant(x)
        data['prob'] = result.predict(x)
        
        return data
    
    def LogLoss(self, model, X, y):
        """Calculate log loss for model evaluation
        
        Parameters:
        -----------
        model : scikit-learn model
            Fitted model
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns:
        --------
        float
            Log loss value
        """
        from sklearn.metrics import log_loss
        
        pred = model.predict_proba(X)
        Logloss = log_loss(y, pred)
        
        return Logloss
    
    def volumeAndDir(self):
        """Create a directory for storing results
        
        Returns:
        --------
        str
            Path to the created directory
        """
        return create_directory()
    
    def DataSelector(self, data):
        """Differentiate data into numerical and categorical categories
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        tuple
            (Target, ID, Numerical columns, Categorical columns)
        """
        self.Target = str(input("ENTER RESPONSE VARIABLE : ")).upper()
        print('\n')
        self.Id = str(input("ENTER UNIQUE ID :  ")).upper()
        print('\n')
        
        try:
            iv_cut = str(input("ENTER IV-CUTOFF(In Decimal)(Default : 0.01): "))
            self.iv_cutoff = round(float(iv_cut), 4)
        except ValueError:
            print("Invalid input for IV-CUTOFF. Default value is considered.")
            self.iv_cutoff = 0.01
        
        print('\n')
        
        try:
            correlation_cutoff = str(input("ENTER CORRELATION-CUTOFF(In Decimal)(Default : 0.65): "))
            self.corr_cutoff = round(float(correlation_cutoff), 4)
        except ValueError:
            print("Invalid input for CORRELATION-CUTOFF. Default value is considered.")
            self.corr_cutoff = 0.65
        
        print('\n')
        
        # Separate numerical and categorical columns
        NumericalCols = data.select_dtypes(exclude='object').columns
        NumericalColsLst = [col for col in NumericalCols if col not in [self.Target, self.Id]]
        
        CatCols = data.select_dtypes(include='object').columns
        CatColsLst = [col for col in CatCols if col not in [self.Target, self.Id]]
        
        return self.Target, self.Id, NumericalColsLst, CatColsLst
    
    # Re-use the imported functions
    def data_report(self, data, numerical):
        """Generate summary report for numerical data"""
        return data_report(data, numerical)
    
    def report_builder(self, summary):
        """Build report based on summary of numerical data"""
        return report_builder(summary)
    
    def Selector_sanity(self, report, all_var=None):
        """Select variables based on data quality checks"""
        return selector_sanity(report, all_var)
    
    def correlationFilter(self, data, varlist, corr_cutoff, Target=None):
        """Filter variables based on correlation"""
        return correlation_filter(data, varlist, corr_cutoff, Target)
    
    def chisq_feature_selection(self, data, catCol):
        """Select categorical features using chi-square test"""
        return chisq_feature_selection(data, catCol, self.Target)
    
    def iv_maker(self, uid, Target, data, variable=None):
        """Calculate Information Value for variables"""
        return iv_maker(uid, Target, data, variable)
    
    def get_iv_data(self, object_dict):
        """Generate Information Value data frames"""
        return get_iv_data(object_dict)
    
    def transformer_other_pandas(self, df, optb_fit_objects, uid, target, metric='bins'):
        """Transform a pandas DataFrame"""
        return transformer_other_pandas(df, optb_fit_objects, uid, target, metric)
    
    def rf_pimp_feature_importance(self, data, Id, Target, selected_features, factor=None, model_params=None):
        """Calculate feature importance using Random Forest"""
        return rf_pimp_feature_importance(data, Id, Target, selected_features, factor, model_params)
    
    def shapRFECV(self, df, features):
        """Perform SHAP-based recursive feature elimination"""
        return shap_rfecv(df, features, self.Target)
    
    def mlDataImputer(self, data, NumericalCols, CatCols):
        """Prepare data for machine learning"""
        return ml_data_imputer(data, NumericalCols, CatCols, self.Target, self.size)
    
    def MlSummary(self, data, oot, binnableFeats=None, Target=None):
        """Generate ML model summary"""
        return ml_summary(data, oot, binnableFeats, Target)
    
    def Gen_ks_calculator(self, data=None, target=None, prob=None):
        """Generate KS table for scored population"""
        return gen_ks_calculator(data, target, prob)
    
    def Gen_ks_calculator_groups(self, data=None, target=None, prob=None, groups=None):
        """Generate KS table with custom grouping"""
        return gen_ks_calculator_groups(data, target, prob, groups)
    
    def Modelling(self, no_of_iter=10, varlist=None, data=None, Target=None):
        """Perform logistic regression"""
        return modeling(no_of_iter, varlist, data, Target)
    
    def Modelling2(self, n=10, varlist=None, data=None, Target=None):
        """Alternative logistic regression implementation"""
        return modeling2(n, varlist, data, Target)
    
    def ReTuneModels(self, result=None, oot=None, data=None, varlist=None, Target=None, path=None):
        """Perform iterative model tuning"""
        return re_tune_models(result, oot, data, varlist, Target, path)
    
    def BestModelSelector(self, output, data, oot, Target):
        """Select best model based on performance metrics"""
        return best_model_selector(output, data, oot, Target)
    
    def p_val_calc(self, result):
        """Extract significant variables based on p-values"""
        cov = result.cov_params()
        sum1 = pd.DataFrame({
            "Z_SCORE": result.params / np.sqrt(np.diag(cov)),
            'pvals': result.pvalues,
            'p_values': (result.pvalues < 0.05)
        }).reset_index()
        
        sum1['z_abs'] = abs(sum1['Z_SCORE'])
        sum2 = pd.DataFrame(sum1.sort_values(by='pvals', ascending=True).reset_index().drop('level_0', axis=1))
        sum2 = sum2.rename(columns={'index': 'Vars'})
        
        significant_vars = list(sum2[sum2["p_values"] == True]["Vars"])
        
        return [col for col in significant_vars if col != 'const']
    
    def plot_roc_curve(self, y_oot, P_base_learners, P_ensemble, labels, ens_label):
        """Plot ROC curve for base learners and ensemble"""
        from scorecard.utils import plot_roc_curve
        return plot_roc_curve(y_oot, P_base_learners, P_ensemble, labels, ens_label)
    
    def prob_to_score(self, data=None, pdo=None, constant=None, prob_column=None):
        """Convert probabilities to scores"""
        from scorecard.modeling import prob_to_score
        return prob_to_score(data, pdo, constant, prob_column)
    
    def ScoreCardBuilder(self, data=None, columns_to_remove=None):
        """
        Main function to build a scorecard model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        columns_to_remove : list, optional
            List of columns to remove
            
        Returns:
        --------
        tuple
            (best_model, path, cols_to_use, model_vars, data)
        """
        return scorecard_builder(self, data, columns_to_remove) 
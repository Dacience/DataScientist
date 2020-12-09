import pandas as pd
import numpy as np

class PearsonCorrelation:
    '''
    A feature Selection method using Pearson Correlation.
    
    A Pearson Correlation Coefficient is calculated using a feature and target
    of a dataset and is calculated for each feature. Coefficient can take value
    between 1 and -1. If the value is near 1 then that feature is positively 
    correlated(directly proportional) to the target, if the value is near -1 
    then that feature is negatively correlated(inversly proportinal) to the
    target and if the value is near 0 then that feature is not related to the
    target.
    
    Parameters:
    ----------
    features: DataFrame
        Features are individual independent variables that act as the input in 
        your system. The features DataFrame should only contain numerical value
        (if categorical value present then encode them to numerical labels).
        
    target: Series
        The target is whatever the output of the input variables.It could be 
        the individual classes that the input variables maybe mapped to in case
        of a classification problem or the output value range in a regression 
        problem. The target series should also contain numerical value like 
        features DataFrame.
    '''
    
    def __init__(self,features,target):
        
        self.X = features
        self.y = target
    
    def corr_score(self,sort = False,reset_index = False):
        '''
        Evaluate the Pearson Correlation Coefficient of each feature column 
        with the target column.
        
        Parameters
        ----------
        sort: bool, default=False
            Whether to sort the order of features according to the coefficient
            value. If True, the return DataFrame is sorted in descending order
            according to correlation coefficients value.
            
        reset_index: bool, default=False
            Whether to reset the index of the return DataFrame. If True, then 
            resets the index of output dataframe.
        
        Returns
        -------
        cor_score: DataFrame with one column as the features name and the other
        as Coefficient value.
        '''
        cor_list = []
        feature_name = self.X.columns.tolist()
        cor_score = pd.DataFrame(columns = ['feature','cor_score'])
        for i in feature_name:
            cor = np.corrcoef(self.X[i], self.y)[0,1]
            cor_list.append(cor)
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        cor_list = np.around(cor_list,decimals = 6)
        cor_score['feature'] = feature_name
        cor_score['cor_score'] = cor_list
        if sort == True:
            cor_score = cor_score.sort_values(by = ['cor_score']
                                              ,ascending = False)
        if reset_index == True:
            cor_score = cor_score.reset_index(drop = True)
        
        return cor_score
        
    def top_corr_featurenames(self,feat_num = 1,ascending = True):
        '''
        Evaluates the name of features with top values of Pearson Correlation
        coefficient.
        
        Parameters
        ----------
        feat_num: int, default=1
            The number of top features names to return according to the 
            correlation coefficient value.
            
        ascending: bool, default=True
            Whether order of return list is in ascending order according to the
            correlation coefficient value. If False, then the feature with 
            higher coefficient value is first element followed by the elements 
            with lower values. 
        
        Returns
        -------
        cor_feature: List containing top feat_num features name.
        '''
        cor_list = []
        feature_name = self.X.columns.tolist()
        for i in feature_name:
            cor =np.corrcoef(self.X[i], self.y)[0,1]
            cor_list.append(cor)
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        cor_list = np.around(cor_list,decimals = 6)
        cor_feature = self.X.iloc[:,np.argsort(np.abs(cor_list))[-(feat_num):]].columns.tolist()
        if ascending == False:
            cor_feature = cor_feature[::-1]
        return cor_feature
        
    def top_corr_features(self,feat_num = 1,ascending = True):
        '''
        Evaluates the features along with all the rows with top values of 
        Pearson Correlation coefficient.
        
        Parameters
        ----------
        feat_num: int, default=1
            The number of top features(with all the data) to return according 
            to the correlation coefficient value.
            
        ascending: bool, default=True
            Whether the columns of return DataFrame is in ascending order
            according to the correlation coefficient value. If False, then the
            feature with higher coefficient value is first column followed by 
            the features with lower values. 
        
        Returns
        -------
        out_feature: DataFrame containing top feat_num features with all the 
        data(rows).
        '''
        cor_list = []
        feature_name = self.X.columns.tolist()
        for i in feature_name:
            cor = np.corrcoef(self.X[i], self.y)[0,1]
            cor_list.append(cor)        
        cor_list = np.around(cor_list,decimals = 6)
        cor_feature = self.X.iloc[:,np.argsort(np.abs(cor_list))[-(feat_num):]].columns.tolist()
        if ascending == False:
            cor_feature = cor_feature[::-1]
        out_features = self.X[cor_feature]
        return out_features


class ChiSquare:
    '''
    A feature Selection method using Chi Square Statistic.
    
    A chi-square test is used in statistics to test the independence of two 
    events. In feature selection, the aim is to select the features which are 
    highly dependent on the response. When two features are independent, the 
    observed count is close to the expected count, thus we will have smaller 
    Chi-Square value. So high Chi-Square value indicates that the hypothesis 
    of independence is incorrect. In simple words, higher the Chi-Square value
    the feature is more dependent on the response and it can be selected for 
    model training.
    
    Parameters:
    ----------
    features: DataFrame
        Features are individual independent variables that act as the input in 
        your system. The features DataFrame should only contain numerical value
        (if categorical value present then encode them to numerical labels).
                
    target: Series
        The target is whatever the output of the input variables.It could be 
        the individual classes that the input variables maybe mapped to in case
        of a classification problem or the output value range in a regression 
        problem. The target series should also contain numerical value like 
        features DataFrame.
    '''
    
    def __init__(self,features,target):
        
        self.X = features
        self.y = target
    
    def chi2_score(self,sort = False,reset_index = False):
        '''
        Evaluate the Chi-square statistic of each feature column with the
        target column. It also auto scales the features data.
        
        Parameters
        ----------
        sort: bool, default=False
            Whether to sort the order of features according to the coefficient
            value. If True, the return DataFrame is sorted in descending order
            according to correlation coefficients value.
            
        reset_index: bool, default=False
            Whether to reset the index of the return DataFrame. If True, then 
            resets the index of output dataframe.
        
        Returns
        -------
        chi2_score: DataFrame with one column as the features name and the other
        as Chi-square statistics.
        '''
        from sklearn.feature_selection import chi2
        from sklearn.preprocessing import MinMaxScaler
        X_norm = MinMaxScaler().fit_transform(self.X)
        chi2_list = chi2(X_norm,self.y)[0]
        feature_name = self.X.columns.tolist()
        chi2_list = np.around(chi2_list,decimals = 6)
        chi2_score = pd.DataFrame(columns = ['feature','chi2_score'])
        chi2_score['feature'] = feature_name
        chi2_score['chi2_score'] = chi2_list
        if sort == True:
            chi2_score = chi2_score.sort_values(by = ['chi2_score']
                                              ,ascending = False)
        if reset_index == True:
            chi2_score = chi2_score.reset_index(drop = True)
        
        return chi2_score
        
    def top_chi2_featurenames(self,feat_num = 1):
        '''
        Evaluates the name of features with top values of Chi-square
        statistics.
        
        Parameters
        ----------
        feat_num: int, default=1
            The number of top features names to return according to the 
            chi-square statistics.
            
        Returns
        -------
        chi2_feature: List containing top feat_num features name.
        '''
        from sklearn.feature_selection import chi2
        from sklearn.feature_selection import SelectKBest
        from sklearn.preprocessing import MinMaxScaler
        X_norm = MinMaxScaler().fit_transform(self.X)
        chi_selector = SelectKBest(chi2, k=feat_num)
        chi_selector.fit(X_norm, self.y)
        chi_support = chi_selector.get_support()
        chi_feature = self.X.loc[:,chi_support].columns.tolist()
        return chi_feature
        
    def top_chi2_features(self,feat_num = 1):
        '''
        Evaluates the features along with all the rows with top values of 
        Chi-square statistics.
        
        Parameters
        ----------
        feat_num: int, default=1
            The number of top features(with all the data) to return according 
            to the chi-square statistics.
        
        Returns
        -------
        out_feature: DataFrame containing top feat_num features with all the 
        data(rows).
        '''
        from sklearn.feature_selection import chi2
        from sklearn.feature_selection import SelectKBest
        from sklearn.preprocessing import MinMaxScaler
        X_norm = MinMaxScaler().fit_transform(self.X)
        chi_selector = SelectKBest(chi2, k=feat_num)
        chi_selector.fit(X_norm, self.y)
        chi_support = chi_selector.get_support()
        chi_feature = self.X.loc[:,chi_support].columns.tolist()
        out_feature = self.X[chi_feature]
        return out_feature
       
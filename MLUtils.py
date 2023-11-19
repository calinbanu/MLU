import numpy as np
import plotly.subplots as ps
import statsmodels.api as sm
import pandas as pd

from ISLP.models import (ModelSpec as MS, summarize, sklearn_sm)

from sklearn.model_selection import (cross_validate, KFold)

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

class LinearModelAnalytics:

    # Simple Linear Regression data
    # These information are gathered from the simple linear regression fit of each predictor
    # Intercept for each predictor
    slr_intercept_df = pd.DataFrame({'coef' : [], 'std err' : [], 't-statistic' : [], 'P>|t|' : []})
    # Parameter/Slope for each predictor
    slr_param_df = pd.DataFrame({'coef' : [], 'std err' : [], 't-statistic' : [], 'P>|t|' : []})
    # Statistics for each model
    slr_stats_df = pd.DataFrame({'RSE' : [], 'R-squared' : [], 'F-statistic': []})
    # Dictionary { predictor : result } with the results, key is the name of the predictor
    slr_results = {}

    # Multiple Linear Regression data
    # This model incorporates all predictors
    # Parameter/Slope, Intercept 
    mlr_df = pd.DataFrame({'coef' : [], 'std err' : [], 't-statistic' : [], 'P>|t|' : []})
    # Statistics for model
    mlr_stats_df = pd.DataFrame({'RSE' : [], 'R-squared' : [], 'F-statistic': []})
    # VIF from MLR
    mlr_vif_df = pd.DataFrame({'VIF' : []})

    def _slr_process(self, dataframe, predictors, response) -> None:
        for predictor in predictors:
            X = MS([predictor]).fit_transform(dataframe)
            y = dataframe[response]
            result = sm.OLS(y, X).fit()

            self.slr_results[predictor] = result

            intercept_entry = pd.DataFrame(
                {
                    'coef' : [result.params['intercept']], # Get intercept
                    'std err' : [result.bse['intercept']], # get SE(B_0)
                    't-statistic' : [result.tvalues['intercept']],   # Get t-statistic
                    'P>|t|': [result.pvalues['intercept']] # get p-value
                },
                index=([predictor+'_intercept']))
            self.slr_intercept_df = pd.concat([self.slr_intercept_df, intercept_entry])

            for index, _ in result.params.items():
                if index == 'intercept':
                    continue
                param_entry = pd.DataFrame(
                    {
                        'coef' : [result.params[index]], # Get slope/parameter
                        'std err' : [result.bse[index]], # Get SE(B_1)
                        't-statistic' : [result.tvalues[index]],   # Get t-statistic
                        'P>|t|': [result.pvalues[index]] # get p-value
                    },
                    index=([index]))
                self.slr_param_df = pd.concat([self.slr_param_df, param_entry])

            slr_stats_entry = pd.DataFrame(
                {
                    'RSE' : [np.sqrt(result.scale)], # Get RSE
                    'R-squared' : [result.rsquared], # Get R-squared
                    'F-statistic' : [result.fvalue], # Get F-statistic
                },
                index=([predictor]))
            self.slr_stats_df = pd.concat([self.slr_stats_df, slr_stats_entry])

    def _mlr_process(self, dataframe, predictors, response) -> None:
        X = MS(predictors).fit_transform(dataframe)
        y = dataframe[response]
        result = sm.OLS(y, X).fit()

        for i, predictor in enumerate(X.columns.drop('intercept')):
            mlr_vif_entry = pd.DataFrame(
                {
                    'VIF' : [VIF(X.values, (i + 1))], # Get slope/parameter
                },
                index=[predictor])
            self.mlr_vif_df = pd.concat([self.mlr_vif_df, mlr_vif_entry])

        for index, _ in result.params.items():
            mlr_entry = pd.DataFrame(
                {
                    'coef' : [result.params[index]], # Get slope/parameter
                    'std err' : [result.bse[index]], # Get SE(B_1)
                    't-statistic' : [result.tvalues[index]],   # Get t-statistic
                    'P>|t|': [result.pvalues[index]] # get p-value
                },
                index=([index]))
            self.mlr_df = pd.concat([self.mlr_df, mlr_entry])

        mlr_stats_entry = pd.DataFrame(
        {
            'RSE' : [np.sqrt(result.scale)], # Get RSE
            'R-squared' : [result.rsquared], # Get R-squared
            'F-statistic' : [result.fvalue], # Get F-statistic
        })
        self.mlr_stats_df = pd.concat([self.mlr_stats_df, mlr_stats_entry])

    def __init__(self, dataframe, predictors, response) -> None:
        self.predictors = predictors
        self.response = response
        self.dataframe = dataframe

        self._slr_process(dataframe, predictors, response)
        self._mlr_process(dataframe, predictors, response)
    
    def __str__(self) -> str:
        str_predictors = ",".join(str(el) for el in self.predictors)
        str_response = self.response
        return str("Predictors: " + str_predictors + 
                   "\nResponse: " + str_response + 
                   "\n\n#### SLR Intercept ####\n" + self.slr_intercept_df.__str__() + 
                   "\n\n#### SLR Slope/Parameter ####\n" + self.slr_param_df.__str__() + 
                   "\n\n#### SLR Stats ####\n" + self.slr_stats_df.__str__() +
                   "\n\n#### MLR Stats ####\n" + self.mlr_df.__str__() + 
                   "\n\n#### MLR Stats ####\n" + self.mlr_stats_df.__str__() +
                   "\n\n#### MLR VIF ####\n" + self.mlr_vif_df.__str__())

    def _poly_cv_error(self, predictor, max_grade=10, k_splits=10):
        cv_error = np.zeros (max_grade)
        model = sklearn_sm(sm.OLS)
        cv = KFold(n_splits=k_splits, shuffle=True, random_state=0) # use same splits for each degree
        for i, d in enumerate(range(1,(max_grade+1))):
            X = np.power.outer(np.array(self.dataframe[predictor]), np.arange(d+1))
            y = self.dataframe[self.response]
            M_CV = cross_validate(model, X, y, cv=cv)
            cv_error[i] = np.mean(M_CV['test_score'])
        return cv_error
    
    def plot_slr(self, max_grade=10, k_splits=10):
        # graph column titles
        titles=[
            "Values",
            "Residual vs Fitted",
            "Residual vs Index/Time",
            "Leverage statistic",
            "Input vs Output vs Regression",
            "Polynomial KFold CV MSE",
        ]

        rows = len(self.predictors)
        cols = 6
        height = rows * 500
        width = cols * 500
        
        fig = ps.make_subplots(rows=rows, cols=cols, column_titles=titles)
        
        # Remove legend
        fig.update_layout(showlegend=False)
        fig.update_layout(height=height, width=width)
        fig.update_layout(template='plotly_dark')

        slr_intercept_df = pd.DataFrame({'coef' : [], 'std err' : [], 't-statistic' : [], 'P>|t|' : []})
        slr_param_df = pd.DataFrame({'coef' : [], 'std err' : [], 't-statistic' : [], 'P>|t|' : []})
        
        for row, predictor in enumerate(self.predictors):
            row = row + 1
            X = MS([predictor]).fit_transform(self.dataframe)
            y = self.dataframe[self.response]
            
            # Values
            fig.add_scatter(y=self.dataframe[predictor], mode='markers', row=row, col=1, marker_color="blue")
            fig.update_xaxes(title_text="Index", row=row, col=1)
            fig.update_yaxes(title_text=predictor, row=row, col=1)
            fig.add_hline(y=0, line_width=1, line_color="grey", line_dash="dash", row=row, col=1)

            # Residual vs fitted (y_pred) plot
            fig.add_scatter(x=self.slr_results[predictor].fittedvalues, y=self.slr_results[predictor].resid, mode='markers', row=row, col=2, marker_color="blue")
            fig.update_xaxes(title_text="Fitted vales", row=row, col=2)
            fig.update_yaxes(title_text="Residuals", row=row, col=2)
            fig.add_hline(y=0, line_width=1, line_color="grey", line_dash="dash", row=row, col=2)
            
            # Residual vs time/index
            # Residual vs fitted (y_pred) plot
            fig.add_scatter(y=self.slr_results[predictor].resid, mode='lines', row=row, col=3, marker_color="blue")
            fig.update_xaxes(title_text="Index", row=row, col=3)
            fig.update_yaxes(title_text="Residuals", row=row, col=3)
            fig.add_hline(y=0, line_width=1, line_color="grey", line_dash="dash", row=row, col=3)

            # Leverage satistic
            infl = self.slr_results[predictor].get_influence()
            fig.add_scatter(x=np.arange(X.shape[0]), y=infl.hat_matrix_diag, mode='markers', row=row, col=4, marker_color="blue")
            fig.update_xaxes(title_text="Index", row=row, col=4)
            fig.update_yaxes(title_text="Leverage(h_i)", row=row, col=4)
            # average leverage = (p+1)/n
            average_leverage=(X.shape[1])/X.shape[0]
            fig.add_hline(y=average_leverage, line_width=1, line_color="red", row=row, col=4)
            
            values = self.dataframe[predictor]
            y_predict = self.slr_results[predictor].predict(X)
            fig.add_scatter(x=values, y=y, mode='markers', row=row, col=5, marker_color="blue")
            fig.add_scatter(x=values, y=y_predict, row=row, col=5, marker_color="red")
            fig.update_xaxes(title_text=predictor, row=row, col=5)
            fig.update_yaxes(title_text=self.response, row=row, col=5)

            if self.dataframe.dtypes[predictor].name != 'category':
                cv_error = self._poly_cv_error(predictor, max_grade, k_splits)
                grade_vec = np.array(range(1,(max_grade+1)))
                fig.add_scatter(y=cv_error, x=grade_vec, mode='lines+markers', row=row, col=6, marker_color="blue")
                fig.update_xaxes(title_text="Grade", row=row, col=6)
                fig.update_yaxes(title_text="MSE", row=row, col=6)

        fig.show()

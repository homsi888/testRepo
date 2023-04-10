#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercises 11

Private vs Public school
"""

import numpy as np
import numpy.linalg as la
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
# Helper functions
###############################################################################


def redundant_correlation_pairs(df):
    """
    Get the lower triangular and diagonal elements of a correlation matrix.

    Parameters
    ----------
    df : Pandas DataFrame
        Correlation matrix.

    Returns
    -------
    DataFrame with pair-wise correlations.

    """
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop

###############################################################################


def top_correlations(df, absolute=False, n=5):
    """
    Create ranking of the highest (possibly absolute) correlations.

    Parameters
    ----------
    df : Pandas DataFrame
        Correlation matrix.

    Returns
    -------
    DataFrame with pair-wise correlations.

    """
    if absolute:
        au_corr = df.abs().unstack()
    else:
        au_corr = df.unstack()
    labels_to_drop = redundant_correlation_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]

###############################################################################


def data_frame_to_latex_table_file(file_name, df):
    """
    Take a pandas DataFrame and creates a file_name.tex with LaTeX table data.

    Parameters
    ----------
    file_name : string
                name of the file
    df : Pandas DataFrame
        Correlation matrix.

    Returns
    -------
    saves DataFrame to disk.
    """
    # create and open file
    text_file = open(file_name, "w")
    # data frame to LaTeX
    df_latex = df.to_latex()
    # Consider extensions (see later in class)
    # write latex string to file
    text_file.write(df_latex)
    # close file
    text_file.close()

###############################################################################


def summary_to_latex_table_file(file_name, summary):
    """
    Take a pandas DataFrame and creates a file_name.tex with LaTeX table data.

    Parameters
    ----------
    file_name : string
                name of the file
    df : statsmodels.iolib.summary.Summary
        Summary

    Returns
    -------
    saves Summary to disk.
    """
    # create and open file
    text_file = open(file_name, "w")
    # data frame to LaTeX
    df_latex = summary.as_latex()
    # Consider extensions (see later in class)
    # write latex string to file
    text_file.write(df_latex)
    # close file
    text_file.close()

###############################################################################


def results_summary_to_dataframe(results, rounding=2):
    """
    Transform the result of an statsmodel results table into a dataframe.

    Parameters
    ----------
    results : string
                name of the file
    rounding : int
                rounding

    Returns
    -------
    returns a pandas DataFrame with regression results.
    """
    # get the values from results
    # if you want, you can of course generalize this.
    # e.g. if you don't have normal error terms
    # you could change the pvalues and confidence bounds
    # see exercise session 9?!
    pvals = results.pvalues
    tvals = results.tvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    # create a pandas DataFrame from a dictionary
    results_df = pd.DataFrame({"pvals": np.round(pvals, rounding),
                               "tvals": np.round(tvals, rounding),
                               "coeff": np.round(coeff, rounding),
                               "conf_lower": np.round(conf_lower, rounding),
                               "conf_higher": np.round(conf_higher, rounding)})
    # This is just to show you how to re-order if needed
    # Typically you should put them in the order you like straigh away
    # Reordering...
    results_df = results_df[["coeff", "tvals", "pvals", "conf_lower",
                             "conf_higher"]]

    return results_df

###############################################################################
# Load the data set
###############################################################################


# set the folder for reporting
# This makes it easier if you want to change the folder in the future
# Do not forget the trailing forward slash if you change this.
report_folder = '../report/'
# set the folder for excel files
excel_folder = '../excel/'
# folder for figures
figure_folder = '../figures/'

# setting for output printing
print_line_length = 90
print_line_start = 5

# print start of question 1
print_statement = 'Q1: Load the data'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

# name of data file
data_file = '../data/private_public.csv'
# load data (assumed in current folder)
try:
    data = pd.read_csv(data_file, index_col=0)
    print('Data loaded')
except:
    print('Data could not be loaded')

num_obs = data.shape[0]
###############################################################################
# Check the descriptive statistics
###############################################################################

print_statement = 'Q2: Descriptive Statistics'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

d_stats = data.describe()

print(d_stats)

# writes the d_stats data frame to a tex file that can be easily inputted in a
# LaTeX document
data_frame_to_latex_table_file(report_folder + 'describe.tex',
                               np.round(d_stats.T, 2))

# writes the d_stats data frame to a excel file that can be easily inputted in
# a word document
np.round(d_stats.T, 2).to_excel(excel_folder + 'describe.xlsx')

###############################################################################
# Run the dummy model
###############################################################################

print_statement = 'Q3: Regression model on private school dummy'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

# set the dependent variable to equal final test
y = data['final_test']

# our regressors only contain the private dummy variable
x_q3 = data['private']
X_q3 = sm.add_constant(x_q3)

# set-up the model
model_q3 = sm.OLS(y, X_q3)
# estimate the model
results_q3 = model_q3.fit()

print(results_q3.summary())

summary_to_latex_table_file(report_folder + 'dummy_model.tex',
                            results_q3.summary())

###############################################################################
# Plot residuals vs entry task
###############################################################################

print_statement = 'Q4: Residuals vs entry task'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

fignum = 1
fig, ax = plt.subplots(1, 1, num=fignum)
ax.scatter(data['entry_test'], results_q3.resid)
ax.grid(True, linestyle=':')
ax.set_xlabel(r'entry_task')
ax.set_ylabel(r'$e$')
ax.set_title(r'Omitted Variable Bias')
plt.savefig(figure_folder + 'residuals_vs_entry_task.png')
plt.show()
# increment the figure number
fignum += 1

###############################################################################
# entry_test correlations
###############################################################################

print_statement = 'Q5: Correlation entry task with private and final_test'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

corr_q5 = data[['final_test', 'private', 'entry_test']].corr()

print(corr_q5)

data_frame_to_latex_table_file(report_folder + 'correlation_q5.tex',
                               np.round(corr_q5, 2))

fig, ax = plt.subplots(1, 1, num=fignum)
sns.heatmap(corr_q5, ax=ax)
ax.grid(True, linestyle=':')
ax.set_title(r'Correlation matrix')
plt.savefig('../figures/correlation_heatmap.png')
plt.show()
fignum += 1

###############################################################################
# Impact of correlation
###############################################################################

print_statement = 'Q6: Impact of correlation on estimate'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

# this is not necessary, of cours
print('The estimate would be biased.')

###############################################################################
# Estimate full model
###############################################################################

print_statement = 'Q7: Estimate the full model'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

x_full = data.drop(['final_test'], axis=1)

X_full = sm.add_constant(x_full)

# set-up model
full_model = sm.OLS(y, X_full)
results_full = full_model.fit()

print(results_full.summary())

summary_to_latex_table_file(report_folder + 'full_model.tex',
                            results_full.summary())

# alternatively, you can send for only the coefficients part
# Preferred method
data_frame_to_latex_table_file(report_folder + 'full_model2.tex',
                               results_full.summary2().tables[1].round(2))

results_full.summary2().tables[1].round(2).to_excel(excel_folder +
                                                    'full_model2.xlsx')

###############################################################################
# Checking high correlations
###############################################################################

print_statement = 'Q8: Checking high correlations'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

top_correlations = top_correlations(data.corr(), absolute=(True))
print(top_correlations)

# writes the top_correlations data frame to a tex file that can be easily
# inputted in a LaTeX document
data_frame_to_latex_table_file(report_folder + 'top_correlations.tex',
                               np.round(top_correlations, 2))

# writes the top_correlations data frame to a excel file that can be easily
# inputted in a Word document
np.round(d_stats, 2).to_excel(excel_folder + 'top_correlations.xlsx')

###############################################################################
# Remove variables
###############################################################################

print_statement = 'Q9: Remove variables'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

x = data.drop(['final_test', 'prim_exit_test', 'sh_2'], axis=1)
X = sm.add_constant(x)

###############################################################################
# Estimate the model
###############################################################################

print_statement = 'Q10: Estimate the model'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

# set-up model
model = sm.OLS(y, X)
results = model.fit()

summary_to_latex_table_file(report_folder + 'smaller_model.tex',
                            results.summary())

# a data frame for only the estimatation part
coefficients_table = results.summary2().tables[1]

# alternatively, you can send for only the coefficients part
data_frame_to_latex_table_file(report_folder + 'smaller_model2.tex',
                               coefficients_table.round(2))

coefficients_table.round(2).to_excel(excel_folder + 'smaller_model.xlsx')

###############################################################################
# Rescaling the variables and re-estimate
###############################################################################

print_statement = 'Q11: Rescaling the variables and re-estimate'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

data['entry_test'] /= 100
data['class_size'] /= 20
data['distance_to_school'] /= 20

x = data.drop(['final_test', 'prim_exit_test', 'sh_2'], axis=1)
X = sm.add_constant(x)

# set-up model
model = sm.OLS(y, X)
# the use_t variable is used to specify whether to use standard errors based
# on the t distribution or the normal distribution.
# use_t=True is the default. use_t=False -> normal distribution
results = model.fit(use_t=False)

print(results.summary())

summary_to_latex_table_file(report_folder + 'rescaled_model.tex',
                            results.summary())

###############################################################################
# Plot squares of the residuals
###############################################################################

print_statement = 'Q12: Plot the squared residuals vs fitted values'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')


fig, ax = plt.subplots(1, 1, num=fignum)
ax.scatter(results.fittedvalues, results.resid**2, label='e')
ax.grid(True, linestyle=':')
ax.set_xlabel(r'$\hat{y}$')
ax.set_ylabel(r'$e^2$')
ax.set_title(r'fitted values vs residuals squared')
plt.savefig('../figures/fitted_values_vs_squared_residuals.png')
plt.show()
fignum += 1

###############################################################################
# Test for heteroskedasticity
###############################################################################

print_statement = 'Q13: Test for heteroskedasticity using White test'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')


# calculate the White test
white_test = het_white(results.resid, X)

print('The F-value equals {} with a p-value equal to {}.'.format(
        white_test[2], white_test[3]))

df_f_test = pd.Series(white_test[2:], index=['F test',
                                             'p-value']).astype(float)
df_f_test.name = 'White test'

data_frame_to_latex_table_file(report_folder + 'white_test.tex',
                               np.round(df_f_test, 2))

np.round(df_f_test, 2).to_excel(excel_folder + 'white_test.xlsx')

###############################################################################
# Correct the standard errors
###############################################################################

print_statement = 'Q14: Correct the standard errors'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

# use a different covariance matrix
# https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC0_se.html
# cov_type = HC0 is the White covariance matrix we have seen in class.
results = model.fit(cov_type='HC0')

print(results.summary())

summary_to_latex_table_file(report_folder + 'rescaled_model_hc.tex',
                            results.summary())

coefficients_table = results.summary2().tables[1]

# alternatively, you can send for only the coefficients part
data_frame_to_latex_table_file(report_folder + 'rescaled_model_hc2.tex',
                               coefficients_table.round(2))


###############################################################################
# Model test
###############################################################################

print_statement = 'Q15: Run the model test'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

results = model.fit(cov_type='HC0')

print('The F-value equals {} with a p-value equal to {}.'.format(
        results.fvalue, results.f_pvalue))

model_test = pd.Series(np.array([results.fvalue, results.f_pvalue]),
                       index=['F test', 'p-value']).astype(float)
model_test.name = 'model test'

data_frame_to_latex_table_file(report_folder + 'model_test.tex',
                               np.round(model_test, 2))

###############################################################################
# Remove insignificant variables
###############################################################################

print_statement = 'Q16: Remove insignificant variables'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

x.drop(['uni_ed_mom', 'distance_to_school'], axis=1, inplace=True)
X = sm.add_constant(x)

model = sm.OLS(y, X)
results = model.fit(cov_type='HC0')

print(results.summary())

summary_to_latex_table_file(report_folder + 'significant_model.tex',
                            results.summary())

###############################################################################
# Test impact of reading books vs museum visits
###############################################################################

print_statement = 'Q17: Test if the impact of reading books is bigger' + \
    ' than museum visits'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

# create R matrix for difference between books_read and museum_visits
R = np.array([0, 0, 0, 0, 0, 0, 1, -1, 0])

# calculate the difference between the coefficient for books_read and
# museum_visits in the numerator.
# use the formula for the Wald-test statistic.
# Note that it is not multiplied by n (number of observations) as
# this is already done in results.cov_HC0 (1 / (1/n) = n)
test =  (R @ results.params)**2 / (R @ (results.cov_HC0) @ R.T)

p_value = 1 - stats.chi2.cdf(test, 1)

# see also
print(results.wald_test(R))

books_vs_museum_test = pd.Series(np.array([test, p_value]),
                                 index=['chi-value', 'p-value']).astype(float)
# name the series so the table has a name heading.
books_vs_museum_test.name = 'books vs museums'

print(books_vs_museum_test)

data_frame_to_latex_table_file(report_folder + 'books_vs_museum_test.tex',
                               np.round(books_vs_museum_test, 2))

###############################################################################
# Predict values
###############################################################################

print_statement = 'Q18: Prediction for an individual'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')


# provide the information for the different schools
schools = pd.DataFrame(index=results.params.index,
                       columns=['private', 'public A', 'public B'])

schools.loc['const', :] = 1
schools.loc['private', 'private'] = 1
schools.loc['private', ['public A', 'public B']] = 0
schools.loc['entry_test', :] = 120
schools.loc['income', :] = np.log(5000)
schools.loc['edu_holidays', :] = 5
schools.loc['books_read', :] = np.log(1 + 12)
schools.loc['museum_visits', :] = np.log(1 + 10)
schools.loc['sports_hours', :] = np.log(1 + 4)
schools.loc['class_size', 'private'] = 20
schools.loc['class_size', 'public A'] = 25
schools.loc['class_size', 'public B'] = 30

# rescaling
schools.loc['entry_test', :] /= 100
schools.loc['class_size', :] /= 20

expected_scores = schools.T @ results.params
expected_scores.name = 'expected scores'
expected_scores.index.name = 'school'

# cast type to float
expected_scores = expected_scores.astype(float)

print(expected_scores)

data_frame_to_latex_table_file(report_folder + 'school_info.tex',
                               np.round(schools, 2))

data_frame_to_latex_table_file(report_folder + 'expected_scores.tex',
                               expected_scores)

###############################################################################
# Analyze best spending
###############################################################################

print_statement = 'Q19: Decision making'
print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

schools.loc['edu_holidays', 'private'] = 0

expected_scores_cost = schools.T @ results.params
expected_scores_cost.name = 'expected scores'
expected_scores_cost.index.name = 'school'

# cast type to float
expected_scores_cost = expected_scores_cost.astype(float)

print(expected_scores_cost)

data_frame_to_latex_table_file(report_folder + 'school_info_cost.tex',
                               np.round(schools, 2))

data_frame_to_latex_table_file(report_folder + 'expected_scores_cost.tex',
                               np.round(expected_scores_cost, 2))

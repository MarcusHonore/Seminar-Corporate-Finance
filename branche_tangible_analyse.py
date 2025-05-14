import numpy as np
import statsmodels.formula.api as smf

##############################################################################
# Her laver vi tangible analyse af ROE på market to Book, for tangible assets
# lig med high, medium og low
# modellerne er lavet på FE
############################################################################ 


def roe_book_tang_high(data_specificity_tang_high, data_merge):
    """
    Run a fixed-effects regression of ROE on debt, debt squared, and controls,
    with year and industry fixed effects.

    Parameters
    ----------
    data_specificity_tang_high : pandas.DataFrame
        Dataframe with high specificity observations (tangible).
    data_merge : pandas.DataFrame
        Merged dataframe used to extract industry labels ('NAICS Sector Name').

    Returns
    -------
    model : RegressionResultsWrapper
        Fitted OLS model with HC3 robust covariance.
    turning_point : float
        Debt ratio at which ROE is maximized/minimized.
    """
    # 1. Copy data frame
    data = data_specificity_tang_high.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROE_market_value',
        'debt_book_value',
        'Debt_squared_book',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROE_market_value") ~ Q("debt_book_value") + Q("Debt_squared_book") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roe_book_tang_high = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roe_book_tang_high.params['Q("debt_book_value")']
    beta2 = model_roe_book_tang_high.params['Q("Debt_squared_book")']
    turning_point_roe_book_tang_high = -beta1 / (2 * beta2)

    return model_roe_book_tang_high, turning_point_roe_book_tang_high


def roe_book_tang_medium(data_specificity_tang_medium, data_merge):
    # 1. Copy data frame
    data = data_specificity_tang_medium.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROE_market_value',
        'debt_book_value',
        'Debt_squared_book',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROE_market_value") ~ Q("debt_book_value") + Q("Debt_squared_book") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roe_book_tang_medium = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roe_book_tang_medium.params['Q("debt_book_value")']
    beta2 = model_roe_book_tang_medium.params['Q("Debt_squared_book")']
    turning_point_roe_book_tang_medium = -beta1 / (2 * beta2)

    return model_roe_book_tang_medium, turning_point_roe_book_tang_medium

def roe_book_tang_low(data_specificity_tang_low, data_merge):
    # 1. Copy data frame
    data = data_specificity_tang_low.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROE_market_value',
        'debt_book_value',
        'Debt_squared_book',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROE_market_value") ~ Q("debt_book_value") + Q("Debt_squared_book") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roe_book_tang_low = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roe_book_tang_low.params['Q("debt_book_value")']
    beta2 = model_roe_book_tang_low.params['Q("Debt_squared_book")']
    turning_point_roe_book_tang_low = -beta1 / (2 * beta2)

    return model_roe_book_tang_low, turning_point_roe_book_tang_low
##############################################################################

##############################################################################
# Her laver vi tangible analyse af ROE på market to market
############################################################################ 

def roe_market_tang_high(data_specificity_tang_high, data_merge):
    
    # 1. Copy data frame
    data = data_specificity_tang_high.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROE_market_value',
        'debt_market_value',
        'Debt_squared_market',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROE_market_value") ~ Q("debt_market_value") + Q("Debt_squared_market") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roe_market_tang_high = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roe_market_tang_high.params['Q("debt_market_value")']
    beta2 = model_roe_market_tang_high.params['Q("Debt_squared_market")']
    turning_point_roe_market_tang_high = -beta1 / (2 * beta2)

    return model_roe_market_tang_high, turning_point_roe_market_tang_high


def roe_market_tang_medium(data_specificity_tang_medium, data_merge):
    # 1. Copy data frame
    data = data_specificity_tang_medium.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROE_market_value',
        'debt_market_value',
        'Debt_squared_market',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROE_market_value") ~ Q("debt_market_value") + Q("Debt_squared_market") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roe_market_tang_medium = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roe_market_tang_medium.params['Q("debt_market_value")']
    beta2 = model_roe_market_tang_medium.params['Q("Debt_squared_market")']
    turning_point_roe_market_tang_medium = -beta1 / (2 * beta2)

    return model_roe_market_tang_medium, turning_point_roe_market_tang_medium

def roe_market_tang_low(data_specificity_tang_low, data_merge):
    # 1. Copy data frame
    data = data_specificity_tang_low.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROE_market_value',
        'debt_market_value',
        'Debt_squared_market',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROE_market_value") ~ Q("debt_market_value") + Q("Debt_squared_market") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roe_market_tang_low = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roe_market_tang_low.params['Q("debt_market_value")']
    beta2 = model_roe_market_tang_low.params['Q("Debt_squared_market")']
    turning_point_roe_market_tang_low = -beta1 / (2 * beta2)

    return model_roe_market_tang_low, turning_point_roe_market_tang_low
##############################################################################

##############################################################################
# Her laver vi tangible analyse af ROA på market to Book, for tangible assets
# lig med high, medium og low
# modellerne er lavet på FE
############################################################################ 


def roa_book_tang_high(data_specificity_tang_high, data_merge):
    """
    Run a fixed-effects regression of roa on debt, debt squared, and controls,
    with year and industry fixed effects.

    Parameters
    ----------
    data_specificity_tang_high : pandas.DataFrame
        Dataframe with high specificity observations (tangible).
    data_merge : pandas.DataFrame
        Merged dataframe used to extract industry labels ('NAICS Sector Name').

    Returns
    -------
    model : RegressionResultsWrapper
        Fitted OLS model with HC3 robust covariance.
    turning_point : float
        Debt ratio at which roa is maximized/minimized.
    """
    # 1. Copy data frame
    data = data_specificity_tang_high.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROA_market_value',
        'debt_book_value',
        'Debt_squared_book',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROA_market_value") ~ Q("debt_book_value") + Q("Debt_squared_book") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roa_book_tang_high = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roa_book_tang_high.params['Q("debt_book_value")']
    beta2 = model_roa_book_tang_high.params['Q("Debt_squared_book")']
    turning_point_roa_book_tang_high = -beta1 / (2 * beta2)

    return model_roa_book_tang_high, turning_point_roa_book_tang_high


def roa_book_tang_medium(data_specificity_tang_medium, data_merge):
    # 1. Copy data frame
    data = data_specificity_tang_medium.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROA_market_value',
        'debt_book_value',
        'Debt_squared_book',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROA_market_value") ~ Q("debt_book_value") + Q("Debt_squared_book") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roa_book_tang_medium = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roa_book_tang_medium.params['Q("debt_book_value")']
    beta2 = model_roa_book_tang_medium.params['Q("Debt_squared_book")']
    turning_point_roa_book_tang_medium = -beta1 / (2 * beta2)

    return model_roa_book_tang_medium, turning_point_roa_book_tang_medium

def roa_book_tang_low(data_specificity_tang_low, data_merge):
    # 1. Copy data frame
    data = data_specificity_tang_low.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROA_market_value',
        'debt_book_value',
        'Debt_squared_book',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROA_market_value") ~ Q("debt_book_value") + Q("Debt_squared_book") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roa_book_tang_low = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roa_book_tang_low.params['Q("debt_book_value")']
    beta2 = model_roa_book_tang_low.params['Q("Debt_squared_book")']
    turning_point_roa_book_tang_low = -beta1 / (2 * beta2)

    return model_roa_book_tang_low, turning_point_roa_book_tang_low
##############################################################################

##############################################################################
# Her laver vi tangible analyse af roa på market to market
############################################################################ 

def roa_market_tang_high(data_specificity_tang_high, data_merge):
    
    # 1. Copy data frame
    data = data_specificity_tang_high.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROA_market_value',
        'debt_market_value',
        'Debt_squared_market',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROA_market_value") ~ Q("debt_market_value") + Q("Debt_squared_market") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roa_market_tang_high = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roa_market_tang_high.params['Q("debt_market_value")']
    beta2 = model_roa_market_tang_high.params['Q("Debt_squared_market")']
    turning_point_roa_market_tang_high = -beta1 / (2 * beta2)

    return model_roa_market_tang_high, turning_point_roa_market_tang_high


def roa_market_tang_medium(data_specificity_tang_medium, data_merge):
    # 1. Copy data frame
    data = data_specificity_tang_medium.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROA_market_value',
        'debt_market_value',
        'Debt_squared_market',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROA_market_value") ~ Q("debt_market_value") + Q("Debt_squared_market") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roa_market_tang_medium = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roa_market_tang_medium.params['Q("debt_market_value")']
    beta2 = model_roa_market_tang_medium.params['Q("Debt_squared_market")']
    turning_point_roa_market_tang_medium = -beta1 / (2 * beta2)

    return model_roa_market_tang_medium, turning_point_roa_market_tang_medium

def roa_market_tang_low(data_specificity_tang_low, data_merge):
    # 1. Copy data frame
    data = data_specificity_tang_low.copy()

    # 2. Add fixed effect variables
    data['year'] = data['year'].astype(str)
    data['branche'] = data_merge['NAICS Sector Name'].astype(str)

    # 3. Define variables
    model_vars = [
        'ROA_market_value',
        'debt_market_value',
        'Debt_squared_market',
        'Effective Tax Rate, (%)',
        'EBITDA Margin, Percent',
        'Total Cash Dividends Paid, Cumulative',
        'Growth',
        'risk',
        'year',
        'branche'
    ]

    # 4. Clean data
    data = (
        data[model_vars]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 5. Run regression with fixed effects
    formula = (
        'Q("ROA_market_value") ~ Q("debt_market_value") + Q("Debt_squared_market") + risk + '
        'Q("Effective Tax Rate, (%)") + Q("EBITDA Margin, Percent") + '
        'Q("Total Cash Dividends Paid, Cumulative") + Growth + C(year) + C(branche)'
    )
    model_roa_market_tang_low = smf.ols(formula=formula, data=data).fit(cov_type='HC3')

    # 6. Compute turning point
    beta1 = model_roa_market_tang_low.params['Q("debt_market_value")']
    beta2 = model_roa_market_tang_low.params['Q("Debt_squared_market")']
    turning_point_roa_market_tang_low = -beta1 / (2 * beta2)

    return model_roa_market_tang_low, turning_point_roa_market_tang_low
##############################################################################
##############################################################################


if __name__ == "__main__":
    # Example usage (requires data_specificity_tang_high, data_merge in scope)
    import pandas as pd
    # data_specificity_tang_high = pd.read_csv("...")  # load your data
    # data_merge = pd.read_csv("...")
    model_roe_book_tang_high, tp = roe_book_tang_high(data_specificity_tang_high, data_merge)
    print(model_roe_book_tang_high.summary())
    print(f"Turning point (Debt Ratio where ROE is max/min): {tp:.2f}")

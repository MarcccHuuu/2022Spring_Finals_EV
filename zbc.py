
import pandas as pd
from pandas import DataFrame
import os.path
from pylab import *
import doctest
import pandas as pd
from pandas import DataFrame
import os.path
import matplotlib.pyplot as plt
from pylab import *
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import doctest

# H1 function：

def split_date_make(file, date_split_symbol, make_split_symbol, date_split_loc, make_split_loc, year_of_extraction):
    """
    The split_date_make function is to extract the year and brand from columns 'Registration valid date' and 'Vehicle name'.
    :param file: the initial data
    :param date_split_symbol: Year, month and day separators in 'Registration valid date'
    :param make_split_symbol: Car make and model separator in 'Vehicle name'
    :param date_split_loc: The location of the database we want to separate in 'Registration valid date'
    :param make_split_loc: The location of the database we want to separate in 'Vehicle name'
    :param year_of_extraction: The earliest year we want to extract
    :return: clean_file with two columns ('Date' and 'Make')

    >>> a = DataFrame(data =[[11797, '2020-02-21', '5YJXCBE2', 39, 'NY DATA.NY.GOV (4/2/2020)','J', '2020-12-19', 'NY', 'Tesla Model X', 'BEV']] ,columns=['ZIP Code', 'Registration Valid Date', 'VIN Prefix', 'DMV ID','DMV Snapshot', 'VIN Model Year', 'Registration Expiration Date','State', 'Vehicle Name', 'Technology'],index=[0] )
    >>> test = split_date_make(a, '-',' ',0,0,2020)
    >>> type(test)
    <class 'pandas.core.frame.DataFrame'>
    >>> test.columns
    Index(['Date', 'Make'], dtype='object')
    """
    file.insert(file.shape[1], 'Date',
                        file['Registration Valid Date'].str.split(date_split_symbol, expand=True)[date_split_loc])
    file.insert(file.shape[-1], 'Make',
                        file['Vehicle Name'].str.split(make_split_symbol, expand=True)[make_split_loc])
    file[['Date']] = file[['Date']].astype(int)
    file = file[file['Date'] >= year_of_extraction]
    clean_file = file[['Date', 'Make']]
    return clean_file

def my_autopct(pct):
    """ The my_autopct function is to extract the percentage which is larger than 1%.

    :param pct: pct statistics
    :return: the result which is larger than 1%
    """
    return ('%.1f%%' % pct) if pct > 1 else ''

def h2_EV_analyze(data_file, state_name):
    """
    The h2_EV_analyze function is to count different brands and show as a pie plot and bar plot.
    :param data_file: The clean file from the split_date_make function.
    :param state_name: The state name of the data we use.
    :return: A data frame that counting different brands, a pie plot and a bar plot according to the data frame.

    >>> a = DataFrame(data = [[2020 ,'Tesla'],[2020 ,'Tesla'],[2020 ,'Tesla']], columns= ['Date','Make'])
    >>> test = a[pd.notnull(a['Make'])].groupby(by='Make').count()
    >>> type(test)
    <class 'pandas.core.frame.DataFrame'>
    >>> for i in test.values:
    ...     print(i)
    [3]
    """
    clean_file_count = data_file[pd.notnull(data_file['Make'])].groupby(by='Make').count()
    clean_file_count.columns = ['Brand Count']
    sorted_clean_file_count = clean_file_count.sort_values('Brand Count',ascending= False)
    sorted_clean_file_count = sorted_clean_file_count.head(20)
    print(sorted_clean_file_count)
    sorted_clean_file_count.plot.pie(subplots=True, figsize=(17, 17), autopct=my_autopct,
                                 title=state_name + ' Pie Chart of Different Brands of Car',labeldistance=0.9,fontsize = 15)
    sorted_clean_file_count.plot.bar(figsize=(20, 10),
                                 title=state_name + ' Bar Chart of Different Brands of Car',fontsize = 30)



# H3 function:

def ev_station_count_df(file,state):
    """ Count the ev station of each data and return a dataframe

    :param file: ev station file
    :param state: string for three state
    :return: the dataframe of the count of ev station of each state

    >>> a1 = DataFrame(data =[['NY','12866'],['NY','14203'],['NY','12866'],['TX','12334']] ,columns=['State', 'ZIP'])
    >>> b1 = ev_station_count_df(a1,'NY')
    >>> print(b1['Count'][0])
    2
    >>> a2 = DataFrame(data =[['NY','12866'],['NY','14203'],['NY','12866'],['TX','12334']] ,columns=['State', 'ZIP'])
    >>> b2 = ev_station_count_df(a2,'All')
    >>> print(b2['ZIP Code'][0])
    12866
    >>> print(type(b2))
    <class 'pandas.core.frame.DataFrame'>
    """
    if state == 'All':
        file_clean = file
    else:
        file_clean = file[file['State']==state]
    file_count = file_clean.loc[:,'ZIP'].value_counts()
    file_count_df = DataFrame(file_count)
    file_count_df = file_count_df.rename_axis('ZIP Code').reset_index()
    ev_station_count = file_count_df.rename(columns={'ZIP': 'Count'})
    return ev_station_count


def ev_car_count_df(file):
    """ Calculate the number of car in each zip code

    :param file:state car data
    :return:the car count of each state
    >>> a1 = DataFrame(data =[['NY','12866'],['NY','14203'],['NY','12866']] ,columns=['State', 'ZIP Code'])
    >>> b1 = ev_car_count_df(a1)
    >>> print(b1['Count'][0])
    2
    >>> print(type(b1))
    <class 'pandas.core.frame.DataFrame'>
    """
    file = DataFrame(file)
    file_count = file.loc[:,'ZIP Code'].value_counts()
    file_count_df = DataFrame(file_count)
    file_count_df = file_count_df.rename(columns={'ZIP Code': 'Count'})
    ev_car_count = file_count_df.rename_axis('ZIP Code').reset_index()
    return ev_car_count


def merge_count_data(x_data, y_data, x_col_name, y_col_name):
    """ Merge car_count_data and station_count_file

    :param x_data: dataframe of x (count in each zip code)
    :param y_data: dataframe of y (count in each zip code)
    :param x_col_name: the column name in the dataframe x
    :param y_col_name: the column name in the dataframe y
    :return: a dataframe of the merging result

    >>> a1 = DataFrame(data =[['98110','28280'],['98029','24224'],['12345','0']] ,columns=['ZIP Code', 'Count'])
    >>> a2 = DataFrame(data =[['98110','3'],['98029','4'],['12345','0']] ,columns=['ZIP Code', 'Count'])
    >>> b1 = merge_count_data(a2,a1,'Station Count','Car Count')
    >>> print(b1['Station Count'][0])
    3
    >>> print(type(b1))
    <class 'pandas.core.frame.DataFrame'>
    """
    merge_data = pd.merge(x_data, y_data, how='left', on='ZIP Code')
    x_colume= merge_data.iloc[:,[1]]
    y_colume= merge_data.iloc[:,[2]]
    x_list= x_colume.columns.values.tolist()
    y_list= y_colume.columns.values.tolist()
    merge_data = merge_data.rename(columns={x_list[0]: x_col_name, y_list[0]: y_col_name})
    merge_data = merge_data.fillna(0)
    #ny_merge_data = ny_merge_data[['Station Count', 'Car Count']]
    merge_data_clean = merge_data.drop(merge_data[merge_data[y_col_name] == 0].index)
    return merge_data_clean


def linear_model_main(data, x_str, y_str):
    """ Show the linear regression plot of the relation of car_count and station_count in each zip code.

    :param data: a dataframe of the merging file (car_count and ev-station_count)
    :param x_str: the column name of x
    :param y_str: the column name of y
    :return: the linear regression plot
    """
    x = data[[x_str]]
    y = data[[y_str]]
    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    plt.scatter(data[x_str],data[y_str])
    y_train = model.predict(x)
    plt.plot(x,y_train,color = 'red',label = 'line')
    plt.ylabel(y_str), plt.xlabel(x_str)
    plt.show()


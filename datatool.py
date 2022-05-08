import os
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# H1 function
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

    >>> percent = my_autopct(10)
    >>> print(percent)
    10.0%
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
    sorted_clean_file_count = clean_file_count.sort_values('Brand Count', ascending=False)
    sorted_clean_file_count = sorted_clean_file_count.head(20)
    print(sorted_clean_file_count)
    sorted_clean_file_count.plot.pie(subplots=True, figsize=(17, 17), autopct=my_autopct,
                                     title=state_name + ' Pie Chart of Different Brands of Car', labeldistance=0.9,
                                     fontsize=15)
    sorted_clean_file_count.plot.bar(figsize=(20, 10),
                                     title=state_name + ' Bar Chart of Different Brands of Car', fontsize=30)


# H2 function
def download_file(file_name: str, url: str) -> None:
    """ Download the big csv data file from link if not in the project path

    :param url: the download url
    :param file_name: file name
    :return: None
    >>> if os.path.exists('data/wa_ev_registrations_public.csv'):
    ...     os.remove('data/wa_ev_registrations_public.csv')
    >>> download_file('wa_ev_registrations_public', 'https://www.atlasevhub.com/public/dmv/wa_ev_registrations_public.csv')
    >>> os.path.exists('data/wa_ev_registrations_public.csv')
    True
    >>> if os.path.exists('data/wa_test1.csv'):
    ...     os.remove('data/wa_test1.csv')
    >>> download_file('wa_ev_registrations_public', 'https://www.atlasevhub.com/public/dmv/wa_ev_registrations_public.csv')
    >>> os.path.exists('data/wa_test1.csv')
    False
    """
    if not os.path.exists('data/{}.csv'.format(file_name)):
        import requests
        file = requests.get(url)
        with open('data/{}.csv'.format(file_name), 'wb') as text:
            text.write(file.content)


def preprocess_dataframe(df: pd.DataFrame, date: []) -> tuple:
    """ Pre-process the dataframe, add some valid columns for later operation, and count for recent data

    :param df: the dataframe that need to be processed
    :param date: a list for the beginning date we count for recently data, each three elements represent year, month and day
    :return tuple of (df, df_recently) in which - df: the dataframe after process,
    df_recently: the recently dataframe beginning from the date

    >>> wa_test = pd.read_csv('data/wa_test.csv')
    >>> test, test_recently = preprocess_dataframe(wa_test, [2012, 12, 31])
    >>> 'Valid Datetime' in test and 'Valid Month' in test and 'Valid Quarter' in test
    True
    >>> 'Dynamic Type' in test
    True
    >>> test_recently[test_recently['Valid Datetime'] < pd.Timestamp(2012, 12, 31)].count()
    Vehicle ID                      0
    ZIP Code                        0
    Registration Valid Date         0
    VIN Prefix                      0
    VIN Model Year                  0
    DMV ID                          0
    DMV ID Complete                 0
    DMV Snapshot                    0
    Registration Expiration Date    0
    State Abbreviation              0
    Geography                       0
    Vehicle Name                    0
    Technology                      0
    Valid Datetime                  0
    Valid Month                     0
    Valid Quarter                   0
    Valid Year                      0
    Dynamic Type                    0
    dtype: int64
    """
    df['Valid Datetime'] = pd.to_datetime(df['Registration Valid Date'])
    df['Valid Month'] = df['Registration Valid Date'].str[:7]
    df['Valid Quarter'] = pd.PeriodIndex(df['Valid Datetime'], freq='Q')
    df['Valid Year'] = df['Registration Valid Date'].str[:4]
    # count the Dynamic Type(Technology) for BEV/PHEV
    df['Dynamic Type'] = df['Technology'].apply(lambda x: 1 if x == 'BEV' else 0)

    assert max(df['Valid Datetime']) > pd.Timestamp(2015, 1, 1)
    df_recently = df[df['Valid Datetime'] >= pd.Timestamp(date[0], date[1], date[2])]
    return df, df_recently


def get_monthly_report(df: pd.DataFrame) -> pd.DataFrame:
    """ get the dataframe containing resorted by each month

    :param df: dataframe
    :return: resorted dataframe with month

    >>> test_df = pd.read_csv('data/wa_test.csv')
    >>> test_df, test_df_recently = preprocess_dataframe(test_df, [2015, 1, 1])
    >>> get_monthly_report(test_df)
      Valid Month  Vehicles Total Number  BEV Number  PHEV Number
    0     2021-08                     49          39           10
    1     2021-10                     24          21            3
    2     2021-12                      4           3            1
    3     2022-03                     12          10            2
    """
    df_month = df.groupby(['Valid Month']) \
        .agg({'DMV ID': 'count', 'Dynamic Type': 'sum'}).reset_index() \
        .rename(columns={'DMV ID': 'Vehicles Total Number', 'Dynamic Type': 'BEV Number'})

    df_month['PHEV Number'] = df_month['Vehicles Total Number'] - df_month['BEV Number']
    return df_month


def draw_month_report(df_month: pd.DataFrame) -> None:
    """ draw the month curve report

    :param df_month: the dataframe contains the registration month information
    :return: None (showing plots)
    """
    df_month.plot(x='Valid Month', y={'PHEV Number', 'BEV Number'}, kind='line', figsize=(10, 5), grid=True)
    df_month[df_month['Valid Month'].str.startswith('202')].plot(x='Valid Month', y={'PHEV Number', 'BEV Number'},
                                                                 kind='line', figsize=(10, 5), grid=True)
    df_month[df_month['Valid Month'].str.startswith('202')].plot(x='Valid Month', y={'PHEV Number', 'BEV Number'},
                                                                 kind='bar', stacked=True, figsize=(10, 5))
    plt.show()


def get_vehicle_quarterly_data(df: pd.DataFrame, car_list: [], min_amount=0) -> pd.DataFrame:
    """ get certain car's development trend whose registration number is above min_amount

    :param df: dataframe of registration record
    :param car_list: list of the searching car make
    :param min_amount: minimum car registration number that is taken into consideration
    :return: datafrome of quarterly data for specific car

    >>> test_df = pd.read_csv('data/wa_test.csv')
    >>> test_df, test_df_recently = preprocess_dataframe(test_df, [2015, 1, 1])
    >>> get_vehicle_quarterly_data(test_df, [], 4000)
    Empty DataFrame
    Columns: []
    Index: []
    >>> get_vehicle_quarterly_data(test_df, ['Porsche Taycan', 'Kia Niro EV', 'Polestar 2'])
    Vehicle Name   Kia Niro EV  Polestar 2  Porsche Taycan
    Valid Quarter                                         
    2021Q3                 2.0         NaN             NaN
    2022Q1                 NaN         1.0             1.0
    """
    df_quarter = df.groupby(['Vehicle Name', 'Valid Quarter'])\
        .agg({'DMV ID': 'count'}).reset_index()\
        .rename(columns={'DMV ID': 'Vehicles Amount'})
    if len(car_list) > 0:
        df_quarter = df_quarter[df_quarter['Vehicle Name'].isin(car_list)]
    df_quarter = df_quarter[df_quarter['Vehicles Amount'] >= min_amount]
    df_pivot_car_in_quarter = pd.pivot(df_quarter, index='Valid Quarter', columns='Vehicle Name',
                                       values='Vehicles Amount')
    return df_pivot_car_in_quarter


def get_mom_yoy(df_recently: pd.DataFrame, search_month: str, min_count) -> tuple:
    """ Get data for three plot (2 curve and 1 bar plot) of models rank in certain month

    :param df_recently: the dataframe of recent registration
    :param search_month: the searching month
    :param min_count: the minimum registration count in the month which is valid
    :return: dataframe of month-on-month, dataframe of year-on-year, dataframe of month-on-month by search month

    >>> test_df = pd.read_csv('data/wa_test.csv')
    >>> test_df, test_df_recently = preprocess_dataframe(test_df, [2015, 1, 1])
    >>> df_test_mom, df_test_yoy, df_test_mom_t = get_mom_yoy(test_df_recently, '2021-08', 0)
    >>> df_test_mom.count()
    Vehicle Name
    BMW 3-Series Plug in            3
    BMW i3 REx                      3
    Cadillac ELR                    3
    Chevrolet Bolt EV               3
    Chevrolet Volt                  3
    Chrysler Pacifica               3
    Fiat 500e                       3
    Fisker Karma                    3
    Ford Mustang Mach-E             2
    Jaguar I-Pace                   3
    Kia Niro EV                     3
    Kia Soul EV                     0
    Mitsubishi Outlander Plug In    3
    Nissan Leaf                     3
    Polestar 2                      0
    Porsche Taycan                  0
    Smart forTwo EV                 3
    Tesla Model 3                   3
    Tesla Model S                   3
    Tesla Model X                   3
    Tesla Model Y                   3
    Toyota Prius Prime              1
    Toyota RAV4 Prime               3
    dtype: int64
    >>> df_test_mom_t.count()
    Valid Month
    2021-08    0
    2021-10    0
    2021-12    0
    2022-03    0
    dtype: int64
    >>> df_test2_mom, df_test2_yoy, df_test2_mom_t = get_mom_yoy(test_df_recently, '2021-08', 100)
    >>> df_test2_yoy
    Empty DataFrame
    Columns: []
    Index: []
    """
    df_month = df_recently.groupby(['Vehicle Name', 'Valid Month']) \
        .agg({'DMV ID': 'count'}).reset_index() \
        .rename(columns={'DMV ID': 'Vehicles Amount'})
    df_month = df_month[df_month['Vehicles Amount'] >= min_count]
    df_pivot_car_in_month = pd.pivot(df_month, index='Valid Month', columns='Vehicle Name', values='Vehicles Amount')

    # Month-on-Month & Year-on-Year
    df_ev_mom = df_pivot_car_in_month.transform(func=lambda x: x.pct_change(periods=1))
    df_ev_yoy = df_pivot_car_in_month.transform(func=lambda x: x.pct_change(periods=12))

    df_ev_mom_t = df_ev_mom.T
    if search_month not in df_ev_mom_t.columns:
        return df_ev_mom, df_ev_yoy, None
    df_ev_mom_t.sort_values(by=search_month, inplace=True, ascending=False)
    df_ev_mom_t = df_ev_mom_t[df_ev_mom_t[search_month].notna()]
    return df_ev_mom, df_ev_yoy, df_ev_mom_t


def draw_mom_yoy(df_ev_mom: pd.DataFrame, df_ev_yoy: pd.DataFrame, df_ev_mom_t: pd.DataFrame,
                 state_name: str, search_month: str):
    """ draw three plots(2 curve and 1 bar plot) of models rank in certain month

    :param df_ev_mom: dataframe of month-on-month
    :param df_ev_yoy: dataframe of month-on-month
    :param df_ev_mom_t: dataframe of month-on-month
    :param state_name: the state name
    :param search_month: the search month
    :return: None
    """
    df_ev_mom.plot(figsize=(10, 5), title='EV Rigstration Brand Month on Month Development', grid=True,
                   xlabel='Quarter', ylabel='Vehicle Amount')
    df_ev_yoy.plot(figsize=(10, 5), title='EV Rigstration Brand Year on Year Development', grid=True, xlabel='Quarter',
                   ylabel='Vehicle Amount')
    fig = df_ev_mom_t.plot.bar(figsize=(20, 10), fontsize=25, y=search_month, color='#c0ded9',
                               title=' {} Bar Chart of Different Brands Top MoM in {}' .format(state_name, search_month))
    fig.axes.title.set_size(40)


# H3 function
def ev_station_count_df(file, state):
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
    # ny_merge_data = ny_merge_data[['Station Count', 'Car Count']]
    merge_data_clean = merge_data.drop(merge_data[merge_data[y_col_name] == 0].index)
    return merge_data_clean


def linear_model_main(data, x_str, y_str):
    """ Show the linear regression plot of the relation of car_count and station_count in each zip code.

    :param data: a dataframe of the merging file (car_count and ev-station_count)
    :param x_str: the column name of x
    :param y_str: the column name of y
    :return: the linear regression plot
    """
    x = data[[x_str]].values.reshape(-1, 1)
    y = data[[y_str]].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    plt.scatter(data[x_str], data[y_str])
    plt.plot(x, model.predict(x), color='red', label='line')
    plt.ylabel(y_str), plt.xlabel(x_str)
    plt.show()


# H3(cont.) function
def ev_car_phev_count_df(data: str) -> pd.DataFrame:
    """ Calculate the number of car and PHEV rate in each zip code

    :param data:state car registration data
    :return:the car count and phev rate of each state

    >>> test_data = pd.read_csv('data/wa_test.csv')
    >>> test_car_count = ev_car_phev_count_df(test_data)
    >>> test_car_count['PHEV Rate'][0] == test_car_count['PHEV Count'][0] / test_car_count['Count'][0]
    True
    >>> test_car_count.count()
    ZIP Code      65
    Count         65
    PHEV Count    65
    PHEV Rate     65
    dtype: int64
    """
    data_count = data.loc[:, 'ZIP Code'].value_counts().sort_index(axis=0)
    phev_count = data[data['Technology'] == 'PHEV'].loc[:, 'ZIP Code'].value_counts().sort_index(axis=0)
    file_count_df = DataFrame(data_count)
    file_count_df = file_count_df.rename(columns={'ZIP Code': 'Count'})
    ev_car_count = file_count_df.rename_axis('ZIP Code').reset_index()
    phev_count_df = DataFrame(phev_count)
    phev_rate = phev_count_df.rename(columns={'ZIP Code': 'PHEV Count'})
    phev_rate = phev_rate.rename_axis('ZIP Code').reset_index()
    ev_car_count = pd.merge(ev_car_count, phev_rate, how='left', on='ZIP Code')
    ev_car_count.fillna(0, inplace=True)
    ev_car_count['PHEV Rate'] = ev_car_count['PHEV Count'] / ev_car_count['Count']
    return ev_car_count


def filter_car_station_data(ev_station_count, state_data, us_zip_df):
    """ filter specific columns for the linear regression model

    :param ev_station_count: the dataframe for ev station count by zip code
    :param state_data: the dataframe for specific state registration data
    :param us_zip_df: the dataframe for the state zip code data
    :return: the dataframe for all the merged message

    >>> test_data = pd.read_csv('data/wa_test.csv')
    >>> ev_station_data = pd.read_csv('data/ev_stations_v1.csv')
    >>> ev_station_count_test = ev_station_count_df(ev_station_data, 'WA')
    >>> us_zip_df = pd.read_csv('data/uszips.csv')
    >>> us_zip_df = us_zip_df[['zip', 'state_id', 'population', 'density']]
    >>> us_zip_df['area'] = us_zip_df['population'] / us_zip_df['density']
    >>> us_zip_df = us_zip_df.rename(columns={'zip': 'ZIP Code'})
    >>> test_result = filter_car_station_data(ev_station_count_test, test_data, us_zip_df)
    >>> test_result.columns
    Index(['ZIP Code', 'Station Count', 'Count', 'PHEV Rate', 'population',
           'area'],
          dtype='object')
    >>> len(test_result)
    0
    """
    ev_station_count = ev_station_count.rename(columns={'Count': 'Station Count'})
    ev_car_count = ev_car_phev_count_df(state_data)
    ev_car_station = pd.merge(ev_car_count, ev_station_count, how='left', on='ZIP Code')
    ev_car_station.dropna(axis=0, how='any', inplace=True)
    ev_car_station['ZIP Code'] = ev_car_station['ZIP Code'].astype('int64')

    ev_car_station_pop = pd.merge(ev_car_station, us_zip_df, how='left', on='ZIP Code')
    ev_car_station_pop.fillna(0, inplace=True)
    ev_car_station_pop = ev_car_station_pop[ev_car_station_pop['Count'] > 10]
    ev_car_station_pop = ev_car_station_pop[ev_car_station_pop['Station Count'] > 2]
    ev_car_station_pop = ev_car_station_pop[ev_car_station_pop['population'] > 500]
    ev_car_station_pop = ev_car_station_pop[['ZIP Code', 'Station Count', 'Count', 'PHEV Rate', 'population', 'area']]
    return ev_car_station_pop

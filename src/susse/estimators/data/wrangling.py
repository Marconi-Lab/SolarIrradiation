import pandas as pd

def wrangle_CAMSRAD_data(path):
    df = pd.read_csv(path, skiprows=42, delimiter=';')

    df.rename(columns={
      'TOA': 'TAO (Wh/m2/day)',
      'GHI': 'GHI (Wh/m2/day)',
      'DHI': 'DHI (Wh/m2/day)',
      'BHI': 'BHI (Wh/m2/day)',
      'BNI': 'DNI (Wh/m2/day)',
      'Clear sky GHI': 'Clear sky GHI (Wh/m2/day)',
      'Clear sky DHI': 'Clear sky DHI (Wh/m2/day)',
      'Clear sky BHI': 'Clear sky BHI (Wh/m2/day)',
      'Clear sky BNI': 'Clear sky DNI (Wh/m2/day)',
    }, inplace=True)

    df[['Time', 'End Day']] = df['# Observation period'].str.split('/', expand=True)
    df['Time'] = pd.to_datetime(df['Time'])
    df.drop(columns= ['# Observation period', 'End Day'], inplace=True, axis=1)

    return df

def wrangle_NREL_Data(filepath):
    df = pd.read_csv(filepath, skiprows=2)

    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']].assign(Day=lambda x: x.Day.astype(str).str.zfill(2)))
    df = df[['Date'] + list(df.columns[:-1])]
    df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)

    df.rename(columns={
        'Solar Zenith Angle': 'Solar Zenith Angle (Degrees)',
        'Clearsky DHI': 'Clearsky DHI (W/m2)',
        'Clearsky DNI': 'Clearsky DNI (W/m2)',
        'Clearsky GHI': 'Clearsky GHI (W/m2)',
        'Dew Point': 'Dew Point (C)',
        'Temperature': 'Temperature (C)',
        'Wind Speed': 'Wind Speed (m/s)',
        'Wind Direction': 'Wind Direction (Degrees)',
        'Temperature': 'Temperature (C)',
        'Pressure': 'Pressure (mbar)',
        'Relative Humidity': 'Relative Humidity (%)',
        'Precipitable Water': 'Precipitable Water (cm)',
        'DHI': 'DHI (W/m2)',
        'DNI': 'DNI (W/m2)',
        'GHI': 'GHI (W/m2)',
    }, inplace=True)
    df.reset_index(inplace=True)
    return df

def wrangle_Solcast_data(path):
    df = pd.read_csv(path)
    columns_to_keep = ['air_temp', 'albedo', 'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 
                       'clearsky_gti', 'dhi', 'dni', 'ghi', 'gti', 'period_end', 'period']
    df = df[columns_to_keep]
    df['period_end'] = pd.to_datetime(df['period_end'])

    df.rename(columns={
        'air_temp': 'Temperature (C)',
        'albedo': 'Albedo',
        'clearsky_dhi': 'Clearsky DHI (W/m2)',
        'clearsky_dni': 'Clearsky DNI (W/m2)',
        'clearsky_ghi': 'Clearsky GHI (W/m2)',
        'clearsky_gti': 'Clearsky GTI (W/m2)',
        'dhi': 'DHI (W/m2)',
        'dni': 'DNI (W/m2)',
        'ghi': 'GHI (W/m2)',
        'gti': 'GTI (W/m2)',
        'period_end': 'Date',
        'period': 'Period',
    }, inplace=True)

    return df
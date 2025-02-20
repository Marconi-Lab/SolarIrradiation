import pandas as pd

def wrangle_CAMSRAD_data(path):
    df = pd.read_csv(path, skiprows=42, delimiter=';')

    irra_cols = [
      'TOA',
      'GHI',
      'DHI',
      'BHI',
      'BNI',
      'Clear sky GHI',
      'Clear sky DHI',
      'Clear sky BHI',
      'Clear sky BNI']

    df[irra_cols] = df[irra_cols] / 1000

    df.rename(columns={
      'TOA': 'TAO (kWh/m2/day)',
      'GHI': 'GHI (kWh/m2/day)',
      'DHI': 'DHI (kWh/m2/day)',
      'BHI': 'BHI (kWh/m2/day)',
      'BNI': 'DNI (kWh/m2/day)',
      'Clear sky GHI': 'Clear sky GHI (kWh/m2/day)',
      'Clear sky DHI': 'Clear sky DHI (kWh/m2/day)',
      'Clear sky BHI': 'Clear sky BHI (kWh/m2/day)',
      'Clear sky BNI': 'Clear sky DNI (kWh/m2/day)',
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

    df.set_index('Date', inplace=True)
    irr_cols = ['Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'DHI', 'DNI', 'GHI']

    df = df[irr_cols].resample('D').sum()
    df[irr_cols] = df[irr_cols] / 1000
    df.rename(columns={
        'Clearsky DHI': 'Clearsky DHI (kWh/m2/day)',
        'Clearsky DNI': 'Clearsky DNI (kWh/m2/day)',
        'Clearsky GHI': 'Clearsky GHI (kWh/m2/day)',
        'DHI': 'DHI (kWh/m2/day)',
        'DNI': 'DNI (kWh/m2/day)',
        'GHI': 'GHI (kWh/m2/day)',
    }, inplace=True)
    df.reset_index(inplace=True)
    return df

def wrangle_Solcast_data(path):
    df = pd.read_csv(path)
    columns_to_keep = ['air_temp', 'albedo', 'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 'clearsky_gti', 'dhi', 'dni', 'ghi', 'gti', 'period_end', 'period']
    df = df[columns_to_keep]
    df['period_end'] = pd.to_datetime(df['period_end']).dt.tz_convert('Africa/Nairobi')
    df.set_index('period_end', inplace=True)

    df.drop(columns=['period', 'albedo'], inplace=True)

    irr_cols = df.columns[df.columns != 'air_temp']
    df_temp = df[['air_temp']].resample('D').mean()
    df_irr = df[irr_cols].resample('D').sum()
    df = df_temp.join(df_irr)

    df.index = pd.to_datetime(df.index)
    df[irr_cols] = df[irr_cols] / 1000

    df.columns = [
    'air_temp (°C)',
    'clearsky_dhi (kWh/m2/day)',
    'clearsky_dni (kWh/m2/day)',
    'clearsky_ghi (kWh/m2/day)',
    'clearsky_gti (kWh/m2/day)',
    'dhi (kWh/m2/day)',

    'dni (kWh/m2/day)',
    'ghi (kWh/m2/day)',
    'gti (kWh/m2/day)'
    ]

    df = df.iloc[:-1] 
    return df
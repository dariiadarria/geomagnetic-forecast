import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def prepare_noaa_data():
   
 
    data_dir = Path('data')
    
    plasma_files = sorted(data_dir.glob('solar_wind_plasma_*.csv'))
    mag_files = sorted(data_dir.glob('solar_wind_mag_*.csv'))
    kp_files = sorted(data_dir.glob('kp_index_*.csv'))

    if not plasma_files or not mag_files or not kp_files: 
        print("Нет данных. Запустите data_collector.py")
        return None
    
    plasma_df = pd.read_csv(plasma_files[-1], parse_dates=['time_tag'])
    mag_df = pd.read_csv(mag_files[-1], parse_dates=['time_tag'])
    kp_df = pd.read_csv(kp_files[-1], parse_dates=['time_tag'])
    kp_df['Kp'] = pd.to_numeric(kp_df['Kp'], errors='coerce')
  
    merged = pd.merge(plasma_df, mag_df, on='time_tag', how='inner')
    merged = merged.sort_values('time_tag')
    
    latest_time = merged['time_tag'].max()
    time_threshold = latest_time - timedelta(hours=3)
    recent_wind_data = merged[merged['time_tag'] >= time_threshold]
    
    if len(recent_wind_data) == 0:
        recent_wind_data = merged.tail(100)
    
    relevant_kp = kp_df[kp_df['time_tag'] <= latest_time]
    if relevant_kp.empty:
        print("Нет доступных данных Kp")
        return None
    
    latest_kp_row = relevant_kp.iloc[-1]
    kp_value = latest_kp_row['Kp']

    features = {
        'kp_index': round(kp_value, 2),
        'wind_speed': round(recent_wind_data['speed'].mean(), 2),
        'wind_density': round(recent_wind_data['density'].mean(), 2),
        'threat_level': None
    }
    
    noaa_df = pd.DataFrame([features])
    noaa_df['timestamp'] = latest_time
    noaa_df['data_source'] = 'noaa'
    
    output_path = Path('data/processed/noaa_ready.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_feature_columns = ['kp_index', 'wind_speed', 'wind_density']
    other_columns = ['threat_level', 'timestamp', 'data_source']
    final_columns = model_feature_columns + other_columns
    noaa_df[final_columns].to_csv(output_path, index=False)
    
    print(f"NOAA данные сохранены: {output_path}")
    return noaa_df

if __name__ == "__main__":
    prepare_noaa_data()
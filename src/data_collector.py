import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path
import glob


DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
LOGS_DIR = DATA_DIR / 'logs'

for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

API_URLS = {
    'solar_wind_plasma': 'https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json',
    'solar_wind_mag': 'https://services.swpc.noaa.gov/products/solar-wind/mag-6-hour.json',
    'kp_index': 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json',
    'dst_index': 'https://services.swpc.noaa.gov/products/kyoto-dst.json'
}

def fetch_data(url, name, retries=2, delay=5):
    
    for attempt in range(retries):
        try:
            print(f"Загрузка {name}" + (f" (попытка {attempt+1}/{retries})" if attempt > 0 else ""))
            
            response = requests.get(url, timeout=(10, 30)) 
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
                df['_collected_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                filename = f"data/{name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                df.to_csv(filename, index=False)
                print(f" Успешно: {len(df)} строк")
                print(f" Файл: {filename}")
                return df
            else:
                print(f"Неожиданный формат данных")
                return None
                
        except requests.exceptions.Timeout:
            print(f"    Таймаут...")
            if attempt < retries - 1:
                print(f"    Жду {delay} секунд перед повторной попыткой")
                time.sleep(delay)
            else:
                print(f"    Превышено количество попыток")
                return None
                
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'status_code') and e.response.status_code == 404:
                print(f"    URL не найден (404)")
            else:
                print(f"    Ошибка сети: {e}")
            return None
        except Exception as e:
            print(f"    Ошибка: {e}")
            return None
    
    return None

if __name__ == "__main__":
    print(f"Дата сбора: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    all_data = {}
    for name, url in API_URLS.items():
        df = fetch_data(url, name)
        if df is not None:
            all_data[name] = df

    print("\nСбор данных завершен")

    if all_data:
        print("\n итог сбора данных:")
        for name, df in all_data.items():
            columns_preview = ', '.join(df.columns[:3]) + '...' if len(df.columns) > 3 else ', '.join(df.columns)
            print(f"  {name}: {len(df):4} строк | Столбцы: {columns_preview}")
            
        print("\n проверка полей:")
        critical_fields = {
            'solar_wind_plasma': ['speed', 'density'],
            'solar_wind_mag': ['bz_gsm', 'bt'],
            'kp_index': ['Kp']
        }
        
        for dataset, fields in critical_fields.items():
            if dataset in all_data:
                missing = [field for field in fields if field not in all_data[dataset].columns]
                if missing:
                    print(f"    В {dataset} отсутствуют: {', '.join(missing)}")
                else:
                    print(f"    {dataset}: все ключевые поля на месте")
            else:
                print(f"    {dataset}: данные не были загружены")
    else:
        print("Не удалось собрать данные")

    print("\nпроверка источников:")
    for name, url in API_URLS.items():
        if name in all_data:
            print(f"   {name}: Загружен ({len(all_data[name])} строк)")
        else:
            print(f"   {name}: Не загружен (URL: {url})")
            

    cleanup_age_minutes = 1 
    cutoff_time = datetime.now() - timedelta(minutes=cleanup_age_minutes)


    file_patterns_to_clean = [
        os.path.join(DATA_DIR, 'solar_wind_plasma_*.csv'),
        os.path.join(DATA_DIR, 'solar_wind_mag_*.csv'),
        os.path.join(DATA_DIR, 'kp_index_*.csv'),
        os.path.join(DATA_DIR, 'dst_index_*.csv'),
       
    ]

    print(f"\n Очистка старых файлов (старше {cleanup_age_minutes} минуты)")
    files_deleted_total = 0
    for pattern in file_patterns_to_clean:
        old_files = glob.glob(pattern)
        files_deleted_for_pattern = 0
        for file_path in old_files:
            try:
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime < cutoff_time:
                    os.remove(file_path)
                    print(f"  Удалён старый файл: {file_path}")
                    files_deleted_for_pattern += 1
                    files_deleted_total += 1
            except OSError as e:
                print(f"  Ошибка при удалении {file_path}: {e}")
            except Exception as e:
                print(f"  Неожиданная ошибка при обработке {file_path}: {e}")
        if files_deleted_for_pattern > 0:
            print(f"  Удалено файлов по паттерну '{pattern}': {files_deleted_for_pattern}")

    if files_deleted_total == 0:
        print("  Нет файлов, подлежащих удалению.")
    else:
        print(f"  Всего удалено файлов: {files_deleted_total}")

    print("Очистка завершена.\n")
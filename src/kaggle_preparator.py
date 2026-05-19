import pandas as pd
from pathlib import Path

def prepare_kaggle_data():
    
    kaggle_path = Path('data/raw/solar_storm_impact_dataset.csv')
    df = pd.read_csv(kaggle_path)
    print(f"Загружено: {len(df)} событий")
    
    features_to_drop = [
        'event_id',
        'solar_flare_class',
        'flare_intensity',
        'flare_duration_minutes',
        'event_date'
    ]
    df = df.drop(columns=[col for col in features_to_drop if col in df.columns])

    rename_map = {
        'solar_wind_speed': 'wind_speed',
        'solar_wind_density': 'wind_density',
        'geomagnetic_index_Kp': 'kp_index',
        'power_grid_disruption': 'threat_level'
    }
    df = df.rename(columns=rename_map)
    
    class_counts = df['threat_level'].value_counts().sort_index()
    print("\nРаспределение классов (threat_level):")
    for class_num, count in class_counts.items():
        percentage = count / len(df) * 100
        print(f"  Класс {class_num}: {count} событий ({percentage:.1f}%)")
    
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > 5:
        print(f"\nВнимание: Дисбаланс классов {imbalance_ratio:.1f}:1")
        print("Рекомендуется использовать class_weight='balanced' при обучении")
    
    df['kp_index'] = df['kp_index'].astype('float64')
    
    output_path = Path('data/kaggle/processed/kaggle_ready.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nСохранено: {output_path}")
    print(f"   Признаки: {', '.join(df.columns.tolist())}")
    print(f"   Размер: {len(df)} строк × {len(df.columns)} столбцов")
    
    return df

if __name__ == "__main__":
    prepare_kaggle_data()
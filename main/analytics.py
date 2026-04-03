import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from parser import TelemetryParser
import argparse

def haversine_vectorized(df):
    """
    Haversine для всіх рядків DataFrame одночасно.
    Повертає Series з відстанями між сусідніми точками у метрах.
    """
    lat1 = np.radians(df['lat'].values)
    lon1 = np.radians(df['lon'].values)

    # shift(1) — зсув на 1 рядок
    lat2 = np.radians(df['lat'].shift(1).values)
    lon2 = np.radians(df['lon'].shift(1).values)

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6_371_000

    distances = pd.Series(c * r, index=df.index)
    distances.iloc[0] = 0  # перший рядок не має попередньої точки
    return distances


def total_distance(df_gps):
    """Загальна дистанція польоту у метрах через haversine."""
    distances = haversine_vectorized(df_gps)
    return round(distances.sum(), 2)


def max_horizontal_speed(df_gps):
    """
    Максимальна горизонтальна швидкість у км/год.
    Рахується як відстань між точками / час між точками.
    """
    df = df_gps.copy()

    # Відстань між сусідніми точками (метри)
    df['dist'] = haversine_vectorized(df)

    # Час між сусідніми точками (секунди)
    df['dt'] = df['timestamp'].diff()

    # Швидкість = відстань / час (м/с) → км/год
    df['speed_ms'] = df['dist'] / df['dt']

    # Прибираємо некоректні значення
    df = df[df['dt'] > 0]

    max_speed_ms  = df['speed_ms'].max()
    max_speed_kmh = max_speed_ms * 3.6

    return round(max_speed_kmh, 2)


def max_vertical_speed(df_gps):
    """
    Максимальна вертикальна швидкість у м/с.
    Рахується як різниця висот / час між точками.
    """
    df = df_gps.copy()

    df['alt_smooth'] = df['alt'].rolling(window=11, center=True, min_periods=1).median()
    df['dt']         = df['timestamp'].diff()
    df['d_alt']      = df['alt_smooth'].diff()

    df = df[(df['dt'] > 0)].dropna()
    df['v_speed'] = df['d_alt'] / df['dt']

    # Відкидаємо фізично неможливі значення
    df = df[df['v_speed'].abs() < 50]

    return round(df['v_speed'].max(), 2)


def max_altitude_gain(df_gps):
    """Максимальний набір висоти відносно точки старту (метри)."""
    start_alt = df_gps['alt'].iloc[0]
    return round(df_gps['alt'].max() - start_alt, 2)


def flight_duration(df_gps):
    """Загальна тривалість польоту у секундах."""
    start = df_gps['timestamp'].iloc[0]
    end   = df_gps['timestamp'].iloc[-1]
    return round(end - start, 2)

def get_metrics(df_gps, df_imu):
    """
    Повертає словник з усіма метриками польоту.
    Саме цю функцію викликає Людина 4 у Streamlit.
    """
    return {
        'total_distance_m':     total_distance(df_gps),
        'duration_s':           flight_duration(df_gps),
        'max_speed_h_kmh':      max_horizontal_speed(df_gps),
        'max_speed_v_ms':       max_vertical_speed(df_gps),
        'max_altitude_gain_m':  max_altitude_gain(df_gps),
        'max_acceleration_ms2': '',
        'max_speed_imu_ms':     ''
    }


if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser()

    parser_arg.add_argument(
        'input',
        type=str,
    )

    args = parser_arg.parse_args()


    parser = TelemetryParser(args.input)
    df_gps, df_imu = parser.parse()

    metrics = get_metrics(df_gps, df_imu)

    print("\n" + "="*45)
    print("         МЕТРИКИ ПОЛЬОТУ")
    print("="*45)
    print(f"  Загальна дистанція    : {metrics['total_distance_m']} м")
    print(f"  Тривалість польоту    : {metrics['duration_s']} сек")
    print(f"  Макс. гориз. швидкість: {metrics['max_speed_h_kmh']} км/год")
    print(f"  Макс. верт. швидкість : {metrics['max_speed_v_ms']} м/с")
    print(f"  Макс. набір висоти    : {metrics['max_altitude_gain_m']} м")
    print(f"  Макс. прискорення     : {metrics['max_acceleration_ms2']} м/с²")
    print(f"  Макс. швидкість (IMU) : {metrics['max_speed_imu_ms']} м/с")
    print("="*45)

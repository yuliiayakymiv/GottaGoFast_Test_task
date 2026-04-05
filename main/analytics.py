import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from bin_parser import TelemetryParser
import argparse
from scipy.integrate import cumulative_trapezoid

def haversine_vectorized(df):
    """
    Vectorized Haversine formula — distances between consecutive GPS points (meters).

    Great-circle distance between (φ₁,λ₁) and (φ₂,λ₂) in radians:
        a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
        c = 2·arcsin(√a)      ← central angle
        d = R·c               ← arc length,  R = 6 371 000 m
    """
    lat1 = np.radians(df['lat'].values)
    lon1 = np.radians(df['lon'].values)

    # shift(1) — previous point
    lat2 = np.radians(df['lat'].shift(1).values)
    lon2 = np.radians(df['lon'].shift(1).values)

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2  # haversine
    c = 2 * np.arcsin(np.sqrt(a))   # central angle
    r = 6_371_000                   # Earth mean radius, m

    distances = pd.Series(c * r, index=df.index)
    distances.iloc[0] = 0  # first row has no previous point
    return distances


def total_distance(df_gps):
    """
    Total flight path length (meters).

    Discrete approximation of the arc-length integral:
        D = Σᵢ d(pᵢ, pᵢ₋₁)  ≈  ∫ ||dp/dt|| dt
    """
    distances = haversine_vectorized(df_gps)
    return round(distances.sum(), 2)


def max_horizontal_speed(df_gps):
    """
    Maximum horizontal speed (km/h).

    Forward finite difference of position:
        vᵢ = Δsᵢ / Δtᵢ  [m/s]  →  × 3.6  →  [km/h]
    """
    df = df_gps.copy()

    df['dist']     = haversine_vectorized(df)   # Δsᵢ, meters
    df['dt']       = df['timestamp'].diff()     # Δtᵢ, seconds
    df['speed_ms'] = df['dist'] / df['dt']      # vᵢ = Δsᵢ / Δtᵢ

    df = df[df['dt'] > 0]  # drop zero-interval rows

    return round(df['speed_ms'].max(), 2)


def max_vertical_speed(df_gps):
    """
    Maximum vertical speed (m/s).

    Altitude is median-filtered (window=11) to suppress GPS spikes,
    then differentiated:
        v_z[i] = Δalt_smooth[i] / Δt[i]

    Values |v_z| ≥ 50 m/s are discarded as physically impossible.
    """
    df = df_gps.copy()

    # Median filter — non-linear rank estimator, robust to outliers
    df['alt_smooth'] = df['alt'].rolling(window=11, center=True, min_periods=1).median()
    df['dt']         = df['timestamp'].diff()
    df['d_alt']      = df['alt_smooth'].diff()

    df = df[(df['dt'] > 0)].dropna()
    df['v_speed'] = df['d_alt'] / df['dt']      # v_z = Δalt / Δt
    df = df[df['v_speed'].abs() < 50]           # plausibility filter

    return round(df['v_speed'].max(), 2)


def max_altitude_gain(df_gps):
    """
    Peak altitude gain above takeoff point (meters).

        ΔH = max(alt[i]) − alt[0]

    Note: this is the highest point reached, not the cumulative climb.
    """
    start_alt = df_gps['alt'].iloc[0]
    return round(df_gps['alt'].max() - start_alt, 2)


def flight_duration(df_gps):
    """
    Total flight duration (seconds).

        T = t_end − t_start
    """
    start = df_gps['timestamp'].iloc[0]
    end   = df_gps['timestamp'].iloc[-1]
    return round(end - start, 2)


def get_max_acceleration(df_imu):
    """
    Peak net acceleration (m/s²) with gravity removed.

    IMU measures specific force, so at rest ||a|| ≈ g = 9.81 m/s².
    Net kinematic acceleration:
        a_net = | √(ax²+ay²+az²) − g |
    """
    if df_imu.empty:
        return 0.0

    acc_magnitude = np.linalg.norm(df_imu[['acc_x', 'acc_y', 'acc_z']].values, axis=1)
    return round(np.max(np.abs(acc_magnitude - 9.81)), 2)


def get_max_speed_imu(df_imu, gamma=0.995):
    """
    Peak speed (m/s) via Leaky Integrator over IMU accelerations.

    Naive integration  vᵢ = vᵢ₋₁ + aᵢ·Δtᵢ  drifts unboundedly due to bias ε:
        v(t) → ε·t

    Leaky Integrator adds a forgetting factor γ:
        vᵢ = γ·vᵢ₋₁ + aᵢ·Δtᵢ,   0 < γ < 1

    Steady-state bias response is now bounded:  v_ss = ε·Δt / (1−γ)
    Time constant:  τ ≈ Δt / (1−γ)  — with γ=0.995, Δt≈0.01 s → τ ≈ 2 s

    Bias removal — first 50 samples assumed stationary:
        b = mean(a[0:50]);   a_corrected = a_raw − b

    Final speed:  ||v|| = √(vx²+vy²+vz²)
    """
    if df_imu.empty:
        return 0.0

    t  = df_imu['timestamp'].values
    dt = np.diff(t, prepend=t[0])   # Δtᵢ = tᵢ − tᵢ₋₁

    # Static bias: b = mean(a[0:50])
    bias_x = np.mean(df_imu['acc_x'].iloc[:50])
    bias_y = np.mean(df_imu['acc_y'].iloc[:50])
    bias_z = np.mean(df_imu['acc_z'].iloc[:50])

    acc_x = df_imu['acc_x'].values - bias_x   # a_corrected = a_raw − b
    acc_y = df_imu['acc_y'].values - bias_y
    acc_z = df_imu['acc_z'].values - bias_z

    n = len(t)
    v_x, v_y, v_z = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(1, n):          # vᵢ = γ·vᵢ₋₁ + aᵢ·Δtᵢ
        v_x[i] = v_x[i-1] * gamma + acc_x[i] * dt[i]
        v_y[i] = v_y[i-1] * gamma + acc_y[i] * dt[i]
        v_z[i] = v_z[i-1] * gamma + acc_z[i] * dt[i]

    v_total = np.sqrt(v_x**2 + v_y**2 + v_z**2)   # ||v||
    return round(np.max(v_total), 2)


def get_metrics(df_gps, df_imu):
    """
    Returns all flight metrics as a dictionary (called by Streamlit frontend)
    """
    return {
        'total_distance_m':     total_distance(df_gps),
        'duration_s':           flight_duration(df_gps),
        'max_speed_h_kmh':      max_horizontal_speed(df_gps),
        'max_speed_v_ms':       max_vertical_speed(df_gps),
        'max_altitude_gain_m':  max_altitude_gain(df_gps),
        'max_acceleration_ms2': get_max_acceleration(df_imu),
        'max_speed_imu_ms':     get_max_speed_imu(df_imu)
    }


if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('input', type=str)
    args = parser_arg.parse_args()

    parser = TelemetryParser(args.input)
    df_gps, df_imu = parser.parse()

    metrics = get_metrics(df_gps, df_imu)

    print("\n" + "="*45)
    print("           FLIGHT METRICS")
    print("="*45)
    print(f"  Total distance        : {metrics['total_distance_m']} m")
    print(f"  Flight duration       : {metrics['duration_s']} sec")
    print(f"  Max horizontal speed  : {metrics['max_speed_h_kmh']} m/s")
    print(f"  Max vertical speed    : {metrics['max_speed_v_ms']} m/s")
    print(f"  Max altitude gain     : {metrics['max_altitude_gain_m']} m")
    print(f"  Max acceleration      : {metrics['max_acceleration_ms2']} m/s²")
    print(f"  Max speed (IMU)       : {metrics['max_speed_imu_ms']} m/s")
    print("="*45)

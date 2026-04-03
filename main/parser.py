import pandas as pd
import numpy as np
from pymavlink import mavutil
import argparse


class TelemetryParser:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df_gps = pd.DataFrame()
        self.df_imu = pd.DataFrame()
        self.metadata = {}

    def parse(self):
        mlog = mavutil.mavlink_connection(self.file_path)
        gps_list, imu_list = [], []

        while True:
            msg = mlog.recv_msg()
            if msg is None:
                break

            msg_type = msg.get_type()

            if msg_type == 'GPS':
                lat = msg.Lat
                lon = msg.Lng

                # Якщо значення > 180 або < -180 — це цілі числа
                if abs(lat) > 180:
                    lat = lat / 1e7
                    lon = lon / 1e7

                gps_list.append({
                    'timestamp':  msg.TimeUS / 1e6,
                    'lat':        lat,
                    'lon':        lon,
                    'alt':        msg.Alt / 100 if msg.Alt > 1000 else msg.Alt,
                    'speed':      msg.Spd,
                    'satellites': msg.NSats,
                    'hdop':       msg.HDop / 100 if msg.HDop > 100 else msg.HDop,
                })

            elif msg_type in ('IMU', 'IMU2', 'IMU3'):
                imu_list.append({
                    'timestamp': msg.TimeUS / 1e6,
                    'sensor':    msg_type,
                    'acc_x':     msg.AccX,
                    'acc_y':     msg.AccY,
                    'acc_z':     msg.AccZ,
                    'gyr_x':     msg.GyrX,
                    'gyr_y':     msg.GyrY,
                    'gyr_z':     msg.GyrZ,
                })

        self.df_gps = pd.DataFrame(gps_list)
        self.df_imu = pd.DataFrame(imu_list)

        self._filter_gps()
        self._extract_metadata()

        return self.df_gps, self.df_imu

    def _filter_gps(self):
        """Прибирає записи без GPS фіксу."""
        if self.df_gps.empty:
            return

        before = len(self.df_gps)

        self.df_gps = self.df_gps[
            (self.df_gps['lat'].abs() > 0.1) &
            (self.df_gps['lon'].abs() > 0.1) &
            (self.df_gps['satellites'] >= 6) &
            (self.df_gps['hdop'] < 5.0)
        ].reset_index(drop=True)

        removed = before - len(self.df_gps)

    def _extract_metadata(self):
        if not self.df_imu.empty:
            imu_primary = self.df_imu[self.df_imu['sensor'] == 'IMU']
            if not imu_primary.empty:
                duration = imu_primary['timestamp'].iloc[-1] - imu_primary['timestamp'].iloc[0]
                self.metadata['imu_freq'] = f"{len(imu_primary) / duration:.2f} Hz"
            self.metadata['imu_units'] = "m/s²"
            self.metadata['duration_sec'] = round(
                self.df_imu['timestamp'].iloc[-1] - self.df_imu['timestamp'].iloc[0], 2
            )

        if not self.df_gps.empty:
            gps_duration = self.df_gps['timestamp'].iloc[-1] - self.df_gps['timestamp'].iloc[0]
            self.metadata['gps_freq'] = f"{len(self.df_gps) / gps_duration:.2f} Hz"
            self.metadata['gps_units'] = "WGS-84 (degrees), Alt: metres"
            self.metadata['gps_count'] = len(self.df_gps)

if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser()

    parser_arg.add_argument(
        'input',
        type=str,
    )

    parser_arg.add_argument(
        '--save',
        action='store_true',
    )

    args = parser_arg.parse_args()


    try:

        parser = TelemetryParser(args.input)
        df_gps, df_imu = parser.parse()

        print("\nMETADATA")
        for key, value in parser.metadata.items():
            print(f"  {key}: {value}")

        print("\nCHECK GPS")
        if not df_gps.empty:

            print(df_gps[['timestamp', 'lat', 'lon', 'alt', 'satellites', 'hdop']].head())
            if args.save:
                df_gps.to_csv("gps_data.csv", index=False)
                print("Файл gps_data.csv збережено.")
        else:
            print("Помилка: Дані GPS не знайдені.")

        print("\nCHECK IMU")
        if not df_imu.empty:

            print(df_imu.head())
            if args.save:
                df_imu.to_csv("imu_data.csv", index=False)
                print("Файл imu_data.csv збережено.")
        else:
            print("Помилка: Дані IMU не знайдені.")

    except Exception as e:
        print(f"Сталася помилка при читанні файлу: {e}")

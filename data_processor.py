import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gwl_manual_quarterly_cgwb_tn_2021_2025.csv')

class GroundwaterDataProcessor:
    def __init__(self, csv_path=DATA_PATH):
        self.csv_path = csv_path
        self.df = None
        self.stations_df = None
        self.districts_df = None
        self.load_and_clean_data()

    def load_and_clean_data(self):
        """Loads and cleans the CGWB Tamil Nadu manual quarterly dataset."""
        df = pd.read_csv(self.csv_path)
        
        target_col = 'Groundwater Level Quarterly Manual (meter)'
        # Drop rows where target is NaN
        df = df.dropna(subset=[target_col]).copy()
        
        # Parse Acquisition Time (format: 'DD-MM-YYYY HH:MM')
        df['date'] = pd.to_datetime(df['Data Acquisition Time'], format='%d-%m-%Y %H:%M', errors='coerce')
        df = df.dropna(subset=['date']).copy()
        
        # Sort chronologically by station
        df = df.sort_values(by=['Station', 'date']).reset_index(drop=True)
        
        # Extract calendar and seasonal attributes
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['quarter_label'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        
        # Cyclical temporal encoding
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_quarter'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['cos_quarter'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # CGWB Season naming
        def get_season_name(m):
            if m == 1:
                return 'Post-Monsoon / Rabi (Jan)'
            elif m == 5:
                return 'Pre-Monsoon / Summer (May)'
            elif m == 8:
                return 'Southwest Monsoon (Aug)'
            elif m == 11:
                return 'Northeast Monsoon (Nov)'
            elif m in [12, 1, 2]:
                return 'Winter / Post-Monsoon'
            elif m in [3, 4, 5]:
                return 'Pre-Monsoon / Summer'
            elif m in [6, 7, 8, 9]:
                return 'SW Monsoon'
            else:
                return 'NE Monsoon'
                
        df['season'] = df['month'].apply(get_season_name)
        
        # District level quarterly statistics for hierarchical baseline
        district_quarter_stats = df.groupby(['District', 'quarter'])[target_col].agg(['mean', 'std']).reset_index()
        district_quarter_stats.rename(columns={'mean': 'district_quarter_mean', 'std': 'district_quarter_std'}, inplace=True)
        df = df.merge(district_quarter_stats, on=['District', 'quarter'], how='left')
        
        # Station historical baseline
        station_stats = df.groupby('Station')[target_col].agg(['count', 'mean', 'min', 'max', 'std']).reset_index()
        station_stats.rename(columns={
            'count': 'station_obs_count',
            'mean': 'station_mean',
            'min': 'station_min',
            'max': 'station_max',
            'std': 'station_std'
        }, inplace=True)
        station_stats['station_std'] = station_stats['station_std'].fillna(0.0)
        df = df.merge(station_stats, on='Station', how='left')
        
        # Lag features (previous quarter & 1 year prior)
        df['lag_1'] = df.groupby('Station')[target_col].shift(1)
        df['lag_4'] = df.groupby('Station')[target_col].shift(4)
        
        # Hierarchical imputation for initial observations where lag is missing
        df['lag_1'] = df['lag_1'].fillna(df['station_mean']).fillna(df['district_quarter_mean'])
        df['lag_4'] = df['lag_4'].fillna(df['station_mean']).fillna(df['district_quarter_mean'])
        
        self.df = df
        
        # Build station directory
        self._build_stations_summary()
        self._build_districts_summary()

    def _build_stations_summary(self):
        """Builds a cached table of unique stations with location, latest reading, and count."""
        target_col = 'Groundwater Level Quarterly Manual (meter)'
        latest_idx = self.df.groupby('Station')['date'].idxmax()
        latest_records = self.df.loc[latest_idx].copy()
        
        stations = []
        for _, row in latest_records.iterrows():
            st_name = row['Station']
            st_df = self.df[self.df['Station'] == st_name]
            
            # Trend calculation (difference between last reading and previous reading)
            trend = 0.0
            if len(st_df) >= 2:
                sorted_vals = st_df.sort_values('date')[target_col].values
                trend = round(float(sorted_vals[-1] - sorted_vals[-2]), 2)
                
            depth = float(row[target_col])
            # Classification according to CGWB depth categories
            if depth < 5.0:
                health = 'Safe'
                health_color = 'emerald'
            elif depth < 10.0:
                health = 'Semi-Critical'
                health_color = 'amber'
            elif depth < 20.0:
                health = 'Critical'
                health_color = 'orange'
            else:
                health = 'Over-Exploited'
                health_color = 'rose'
                
            stations.append({
                'station': st_name,
                'district': row['District'],
                'district_code': int(row['District LGD Code']) if pd.notna(row['District LGD Code']) else 0,
                'latitude': float(row['Latitude']),
                'longitude': float(row['Longitude']),
                'latest_depth': round(depth, 2),
                'latest_date': row['date'].strftime('%d %b %Y'),
                'latest_date_iso': row['date'].strftime('%Y-%m-%d'),
                'latest_quarter': row['quarter_label'],
                'total_readings': int(row['station_obs_count']),
                'min_depth': round(float(row['station_min']), 2),
                'max_depth': round(float(row['station_max']), 2),
                'mean_depth': round(float(row['station_mean']), 2),
                'trend_vs_prior': trend,
                'health_status': health,
                'health_color': health_color
            })
            
        self.stations_df = pd.DataFrame(stations).sort_values(by=['district', 'station'])

    def _build_districts_summary(self):
        """Builds a district-level summary of groundwater conditions."""
        target_col = 'Groundwater Level Quarterly Manual (meter)'
        districts = []
        
        for district_name, group in self.df.groupby('District'):
            unique_stations = group['Station'].nunique()
            mean_depth = group[target_col].mean()
            min_depth = group[target_col].min()
            max_depth = group[target_col].max()
            
            # Latest quarter mean
            latest_year = group['year'].max()
            latest_group = group[group['year'] == latest_year]
            latest_mean = latest_group[target_col].mean()
            
            # CGWB Health classification
            if mean_depth < 5.0:
                status = 'Safe'
            elif mean_depth < 10.0:
                status = 'Semi-Critical'
            elif mean_depth < 20.0:
                status = 'Critical'
            else:
                status = 'Over-Exploited'
                
            districts.append({
                'district': district_name,
                'stations_count': unique_stations,
                'total_observations': len(group),
                'mean_depth': round(float(mean_depth), 2),
                'min_depth': round(float(min_depth), 2),
                'max_depth': round(float(max_depth), 2),
                'latest_mean_depth': round(float(latest_mean), 2),
                'status': status
            })
            
        self.districts_df = pd.DataFrame(districts).sort_values(by='district')

    def get_districts_list(self):
        """Returns sorted list of all districts with counts and summary."""
        return self.districts_df.to_dict(orient='records')

    def get_stations_for_district(self, district=None):
        """Returns list of stations, optionally filtered by district."""
        if district and district in self.stations_df['district'].values:
            subset = self.stations_df[self.stations_df['district'] == district]
        else:
            subset = self.stations_df
        # Sort by total_readings descending so best monitored wells show up first
        subset = subset.sort_values(by=['total_readings', 'station'], ascending=[False, True])
        return subset.to_dict(orient='records')

    def get_station_details(self, station_name):
        """Returns comprehensive information and time-series for a single station."""
        st_rows = self.stations_df[self.stations_df['station'] == station_name]
        if st_rows.empty:
            # Fallback to first station if not found
            station_name = self.stations_df.iloc[0]['station']
            st_rows = self.stations_df[self.stations_df['station'] == station_name]
            
        station_info = st_rows.iloc[0].to_dict()
        
        # Get historical time series
        target_col = 'Groundwater Level Quarterly Manual (meter)'
        ts_df = self.df[self.df['Station'] == station_name].sort_values('date').copy()
        
        time_series = []
        for _, row in ts_df.iterrows():
            time_series.append({
                'date': row['date'].strftime('%b %Y'),
                'date_full': row['date'].strftime('%d %b %Y'),
                'date_iso': row['date'].strftime('%Y-%m-%d'),
                'year': int(row['year']),
                'quarter': int(row['quarter']),
                'quarter_label': row['quarter_label'],
                'depth_m': round(float(row[target_col]), 2),
                'season': row['season']
            })
            
        station_info['history'] = time_series
        return station_info

    def get_statewide_overview(self):
        """Returns high-level statistics across all of Tamil Nadu."""
        target_col = 'Groundwater Level Quarterly Manual (meter)'
        vals = self.df[target_col]
        
        # Categorize stations by CGWB health
        st_df = self.stations_df
        safe_count = int((st_df['latest_depth'] < 5.0).sum())
        semi_count = int(((st_df['latest_depth'] >= 5.0) & (st_df['latest_depth'] < 10.0)).sum())
        crit_count = int(((st_df['latest_depth'] >= 10.0) & (st_df['latest_depth'] < 20.0)).sum())
        over_count = int((st_df['latest_depth'] >= 20.0).sum())
        
        return {
            'total_observations': int(len(self.df)),
            'total_stations': int(len(self.stations_df)),
            'total_districts': int(len(self.districts_df)),
            'date_start': self.df['date'].min().strftime('%d %b %Y'),
            'date_end': self.df['date'].max().strftime('%d %b %Y'),
            'overall_mean_depth': round(float(vals.mean()), 2),
            'overall_median_depth': round(float(vals.median()), 2),
            'overall_std_depth': round(float(vals.std()), 2),
            'overall_min_depth': round(float(vals.min()), 2),
            'overall_max_depth': round(float(vals.max()), 2),
            'health_breakdown': {
                'safe': safe_count,
                'semi_critical': semi_count,
                'critical': crit_count,
                'over_exploited': over_count
            }
        }

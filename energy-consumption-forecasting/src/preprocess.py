"""
preprocess.py
=============
Module for downloading, loading, cleaning, and aggregating minute-level smart meter
readings into qualified daily energy consumption (in kWh).

Physical Conversion Principle:
------------------------------
Global_active_power is sampled in Kilowatts (kW), representing an instantaneous power rate.
For a nominal 1-minute sampling interval:
    Energy (kWh) for 1 minute = Power (kW) * (1 hour / 60 minutes) = Power (kW) / 60
For a full day with N valid 1-minute observations:
    Daily Energy (kWh) = sum(Global_active_power_i) / 60

Data Quality & Completeness Policy:
-----------------------------------
A standard solar day has 1,440 minutes (24 hours * 60 minutes).
Strict completeness requirement:
    - Minimum valid readings required: 1,368 minutes (>= 95.0% coverage).
    - Days with < 95% coverage are marked as incomplete (is_eligible = False)
      and excluded from model training to prevent artificial underestimation.
    - Missing minutes are NEVER silently imputed as zero.
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_MINUTES_PER_DAY = 1440
MINIMUM_COVERAGE_THRESHOLD = 0.95  # 95% threshold = 1,368 minutes


def load_raw_uci_data(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads raw UCI household power consumption dataset.
    
    The raw text file is semicolon-delimited (';'), uses '?' for missing values,
    and formats dates as 'dd/mm/yyyy' and time as 'hh:mm:ss'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}. "
            "Please download household_power_consumption.zip from UCI Machine Learning Repository "
            "and extract household_power_consumption.txt into data/raw/."
        )

    logger.info(f"Loading raw data from {filepath} (nrows={nrows})...")
    df = pd.read_csv(
        filepath,
        sep=';',
        na_values=['?'],
        low_memory=False,
        nrows=nrows
    )

    # Validate essential schema
    required_cols = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: {col}")

    # Convert numeric fields
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sub_metering_3 is usually float
    if 'Sub_metering_3' in df.columns:
        df['Sub_metering_3'] = pd.to_numeric(df['Sub_metering_3'], errors='coerce')

    logger.info(f"Loaded {len(df):,} minute-level records.")
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines Date ('dd/mm/yyyy') and Time ('hh:mm:ss') columns into a single
    pandas DatetimeIndex with minute frequency validation.
    """
    logger.info("Parsing dates and timestamps...")
    # Fast format parsing
    datetime_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
    df['datetime'] = pd.to_datetime(datetime_str, format='%d/%m/%Y %H:%M:%S', errors='coerce')
    
    # Check for invalid timestamps
    invalid_dt = df['datetime'].isna().sum()
    if invalid_dt > 0:
        logger.warning(f"Found {invalid_dt} rows with unparseable timestamps.")
        df = df.dropna(subset=['datetime'])

    # Check for duplicate timestamps
    dup_count = df.duplicated(subset=['datetime']).sum()
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate minute timestamps. Deduplicating by keeping first...")
        df = df.drop_duplicates(subset=['datetime'], keep='first')

    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def aggregate_daily_energy(
    df: pd.DataFrame,
    coverage_threshold: float = MINIMUM_COVERAGE_THRESHOLD
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aggregates minute-level active power into daily energy consumption (kWh)
    with strict data quality auditing and incomplete-day flagging.

    Energy Formula:
        daily_energy_kwh = sum(valid_Global_active_power) / 60

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned minute-level dataframe with 'datetime' and 'Global_active_power'.
    coverage_threshold : float
        Fraction of expected 1440 minutes required to qualify (default 0.95).

    Returns:
    --------
    daily_df : pd.DataFrame
        Daily records with 'date', 'daily_energy_kwh', 'valid_minutes',
        'coverage_pct', 'is_eligible'.
    summary : dict
        Audit dictionary with total days, qualified days, and excluded days.
    """
    logger.info("Aggregating minute data to daily energy consumption...")
    df['date'] = df['datetime'].dt.date

    # Group by date
    grouped = df.groupby('date')

    # Aggregations:
    # 1. Total minutes present in dataset for that date
    # 2. Total non-null Global_active_power readings
    # 3. Sum of Global_active_power in kW
    daily_stats = grouped.agg(
        total_rows=('datetime', 'count'),
        valid_minutes=('Global_active_power', lambda x: x.notna().sum()),
        sum_active_power_kw=('Global_active_power', lambda x: x.sum(min_count=1)),
        avg_active_power_kw=('Global_active_power', 'mean'),
        avg_voltage=('Voltage', 'mean'),
        avg_intensity=('Global_intensity', 'mean')
    ).reset_index()

    # Convert date to standard datetime
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    daily_stats = daily_stats.sort_values('date').reset_index(drop=True)

    # Dimensional Unit Conversion: sum(kW) / 60 = kWh
    daily_stats['daily_energy_kwh'] = daily_stats['sum_active_power_kw'] / 60.0

    # Coverage metrics
    daily_stats['missing_minutes'] = EXPECTED_MINUTES_PER_DAY - daily_stats['valid_minutes']
    daily_stats['coverage_pct'] = (daily_stats['valid_minutes'] / EXPECTED_MINUTES_PER_DAY) * 100.0

    # Eligibility threshold (95% = 1368 valid minutes)
    min_minutes_required = int(EXPECTED_MINUTES_PER_DAY * coverage_threshold)
    daily_stats['is_eligible'] = daily_stats['valid_minutes'] >= min_minutes_required

    # Summary audit
    total_days = len(daily_stats)
    qualified_days = int(daily_stats['is_eligible'].sum())
    excluded_days = total_days - qualified_days

    audit_summary = {
        'total_calendar_days': total_days,
        'qualified_days': qualified_days,
        'excluded_incomplete_days': excluded_days,
        'qualification_rate_pct': round((qualified_days / total_days) * 100.0, 2) if total_days > 0 else 0,
        'min_minutes_required': min_minutes_required,
        'mean_daily_energy_kwh': round(daily_stats.loc[daily_stats['is_eligible'], 'daily_energy_kwh'].mean(), 2),
        'min_daily_energy_kwh': round(daily_stats.loc[daily_stats['is_eligible'], 'daily_energy_kwh'].min(), 2),
        'max_daily_energy_kwh': round(daily_stats.loc[daily_stats['is_eligible'], 'daily_energy_kwh'].max(), 2),
    }

    logger.info(
        f"Daily aggregation complete: {qualified_days}/{total_days} days qualified "
        f"({audit_summary['qualification_rate_pct']}%). Excluded {excluded_days} incomplete days."
    )
    return daily_stats, audit_summary


def run_preprocessing_pipeline(
    raw_filepath: str,
    output_dir: str = "data/processed",
    coverage_threshold: float = MINIMUM_COVERAGE_THRESHOLD
) -> str:
    """
    End-to-end execution of the preprocessing pipeline.
    Saves daily_energy.csv and data_quality_audit.csv into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    raw_df = load_raw_uci_data(raw_filepath)
    parsed_df = parse_timestamps(raw_df)
    daily_df, summary = aggregate_daily_energy(parsed_df, coverage_threshold=coverage_threshold)

    output_path = os.path.join(output_dir, "daily_energy.csv")
    daily_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed daily dataset to: {output_path}")

    audit_path = os.path.join(output_dir, "data_quality_audit.csv")
    audit_df = pd.DataFrame([summary])
    audit_df.to_csv(audit_path, index=False)
    logger.info(f"Saved audit metrics to: {audit_path}")

    return output_path


if __name__ == '__main__':
    default_raw = os.path.join("data", "raw", "household_power_consumption.txt")
    if len(sys.argv) > 1:
        default_raw = sys.argv[1]
    
    if os.path.exists(default_raw):
        run_preprocessing_pipeline(default_raw)
    else:
        print(f"Usage: python src/preprocess.py <path_to_raw_dataset>")
        print(f"Note: Default file '{default_raw}' not found yet. See README.md for download instructions.")

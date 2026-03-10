import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_explore(csv_path):
    """
        Load and perform basic inspection of a turbofan dataset.

    Input:
    csv_path : str
        Path to the text file containing the dataset.

    Output:
    df : pandas.DataFrame
        DataFrame containing the loaded data with column names.
        Includes:
        - 'unit_number', 'time_cycles' (engine ID and cycle)
        - 'setting_1' to 'setting_3' (engine operating settings)
        - 's_1' to 's_21' (sensor readings)
    """

    index_names =  ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1,22)]
    column_names = index_names + setting_names + sensor_names
    df = pd.read_csv(csv_path, sep=r"\s+", header=None, index_col=False, names=column_names)

    print("=== Basic Info ===")
    print(f"Shape: {df.shape}")
    print(f"Engines: {df['unit_number'].nunique()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    pd.set_option('display.max_columns', None)
    print(df.describe())
    print(df.head)

    return df

def manage_sensors(df):
    """
    Remove sensor columns with near-constant values.

    Sensors with std < 0.01 carry no meaningful information about engine
    degradation – their value almost does not change across the engine's lifetime,
    so they cannot help detect anomalies or predict RUL.

    Input:
    df : pandas.DataFrame
        DataFrame containing sensor columns.

    Output:
    df : pandas.DataFrame
        DataFrame with low-variance sensor columns removed.
    """
    sensor_cols = [c for c in df.columns if c.startswith('s_')]
    stds = df[sensor_cols].std()
    print("=== Sensor Standard Deviations ===")
    print(stds.sort_values())

    low_std_col = stds[stds < 0.01].index.tolist()
    df = df.drop(columns=low_std_col)
    print(f"\nRemoved {len(low_std_col)} columns: {low_std_col}")
    print(f"Remaining sensors: {[c for c in df.columns if c.startswith('s_')]}")

    return df

def normalize_sensors(df):
    """
    Normalize all remaining sensor columns to the [0, 1] range using MinMaxScaler.

    Normalization is necessary because sensors operate on very different scales
    (e.g. s_11 ranges ~47-48, while s_9 ranges ~9021-9244).

    Input:
    df : pandas.DataFrame
        DataFrame containing sensor columns.

    Output:
    df : pandas.DataFrame
        DataFrame with sensor values scaled to [0, 1].
    """
    sensor_cols = [c for c in df.columns if c.startswith('s_')]
    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df

def add_RUL(df, rul_threshold=125):
    """
    Add Remaining Useful Life (RUL) columns to the training DataFrame.

    RUL describes how many cycles are left before the engine fails.
    In CMAPSS training data, every engine runs until failure, so we can calculate RUL backwards:
    RUL = max_cycle_for_this_engine - current_cycle
    RUL = 0 means failure

    In practice, it is irrelevant whether an engine has RUL=400 or RUL=200 –
    both are far from failure. We only care when RUL drops below
    a critical threshold (commonly 125 cycles).

    Input:
    df : pandas.DataFrame
        DataFrame containing 'unit_number' and 'time_cycles' columns.
    rul_cap : int
        Maximum RUL value to use (default 125).

    Output:
    df : pandas.DataFrame
        DataFrame with two new columns:
        - 'RUL'        : remaining useful life in cycles
        - 'RUL_capped' : RUL capped at rul_threshold
    """

    max_cycles = df.groupby('unit_number')['time_cycles'].max().rename('max_cycles')
    df = df.merge(max_cycles, on='unit_number')
    df['RUL'] = df['max_cycles'] - df['time_cycles']
    df = df.drop(columns=['max_cycles'])

    if rul_threshold is not None:
        df['RUL_capped'] = df['RUL'].clip(upper=rul_threshold)
    else:
        print("RUL added. No cap applied.")

    print(f"RUL range: {df['RUL'].min()} – {df['RUL'].max()} cycles")
    print(f"RUL_capped range: {df['RUL_capped'].min()} – {df['RUL_capped'].max()} cycles")

    pd.set_option('display.max_columns', None)
    print(df.head)

    return df


def run_preprocessing(csv_path, rul_threshold=125):
    """
    Full preprocessing pipeline – runs all steps in order.

    Input:
    csv_path       : str  – path to raw .txt file
    rul_cap        : int  – RUL cap value (default 125)
    rolling_window : int  – rolling statistics window (default 5)

    Output:
    df     : fully preprocessed DataFrame
    scaler : fitted MinMaxScaler (needed to transform test data)
    """
    print("\n--- Step 1: Load ---")
    df = load_explore(csv_path)

    print("\n--- Step 2: Remove low-variance sensors ---")
    df = manage_sensors(df)

    print("\n--- Step 3: Normalize sensors ---")
    df = normalize_sensors(df)

    print("\n--- Step 4: Add RUL ---")
    df = add_RUL(df, rul_threshold=rul_threshold)

    print(f"\n=== Preprocessing complete. Final shape: {df.shape} ===")

    return df


if __name__ == "__main__":
    csv_path = '/home/zoo/TurbofanAI_agent/data/raw/CMaps/train_FD001.txt'
    df = run_preprocessing(csv_path)
    df.to_csv('/home/zoo/TurbofanAI_agent/data/processed/train_FD001.csv', index=False)


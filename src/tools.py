import pandas as pd
import numpy as np
from langchain.tools import tool
import random

DATA_PATH = './data/processed/train_FD001.csv'
KEY_SENSORS = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_11', 's_12', 's_15', 's_17']
SENSOR_DESCRIPTIONS = {
    's_1':  'Fan inlet temperature (°R)',
    's_2':  'LPC outlet temperature (°R)',
    's_3':  'HPC outlet temperature (°R)',
    's_4':  'LPT outlet temperature (°R)',
    's_5':  'Fan inlet pressure (psia)',
    's_6':  'Bypass-duct pressure (psia)',
    's_7':  'HPC outlet pressure (psia)',
    's_8':  'Physical fan speed (rpm)',
    's_9':  'Physical core speed (rpm)',
    's_10': 'Engine pressure ratio P50/P2',
    's_11': 'HPC outlet static pressure (psia)',
    's_12': 'Ratio of fuel flow to Ps30 (pps/psia)',
    's_13': 'Corrected fan speed (rpm)',
    's_14': 'Corrected core speed (rpm)',
    's_15': 'Bypass ratio',
    's_16': 'Burner fuel-air ratio',
    's_17': 'Bleed enthalpy',
    's_18': 'Required fan speed',
    's_19': 'Required fan conversion speed',
    's_20': 'High-pressure turbines cool air flow',
    's_21': 'Low-pressure turbines cool air flow',}

def _load_data(simulate: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    if not simulate:
        return df
    frames = []
    for uid, group in df.groupby("unit_number"):
        group = group.reset_index(drop=True)
        max_idx = len(group)
        min_cut = int(max_idx*0.1)
        max_cut = int(max_idx*0.9)
        cut = random.randint(min_cut, max_cut)
        frames.append(group.iloc[:cut])
    return pd.concat(frames).reset_index(drop=True)

def _get_status(rul: int) -> str:
    if rul < 30:
        return "CRITICAL"
    elif rul < 100:
        return "WARNING"
    return "HEALTHY"

@tool
def get_engines_set_summary() -> str:
    """
    Returns a summary of the entire engine set, including
    total engine count, names of key sensors, average RUL, number of critical engines (RUL < 30) 
    and number of warning engines (RUL between 30 and 100).
    Use this tool first to understand the overall stat of all the engines.
    """
    df = _load_data()
    latest = df.groupby('unit_number').last().reset_index() 
    total = len(latest)
    key_sensors_name = "\n".join([f"{s}: {SENSOR_DESCRIPTIONS.get(s,"Unknown")}"
                                  for s in KEY_SENSORS])
    avg_rul  = latest['RUL'].mean()
    critical = (latest['RUL'] < 30).sum()
    warning  = ((latest['RUL'] >= 30) & (latest['RUL'] < 100)).sum()
    healthy  = (latest['RUL'] >= 100).sum()

    return (
        f"Engines summary:\n"
        f"Total engines: {total}\n"
        f"Key sensors: {KEY_SENSORS}\n"
        f"Key sensors names: {key_sensors_name}\n"
        f"Average RUL: {avg_rul:.1f} cycles\n"
        f"CRITICAL (RUL<30): {critical} engines\n"
        f"WARNING  (RUL<100): {warning} engines\n"
        f"HEALTHY  (RUL>=100): {healthy} engines")

@tool
def get_engine_stats(idx: int) -> str:
    """
    Returns detailed statistics for a single engine unit including
    current cycle, RUL, and the current readings of key sensors.
    Use this when the user asks about a specific engine by its ID number.
    Args:
        idx: The engine unit number (integer).
    """
    df = _load_data()
    engine = df[df['unit_number'] == idx]

    if engine.empty:
        return f"Engine {idx} not found in dataset - there are 100 engines)"

    latest = engine.iloc[-1]
    current_cycle = int(latest['time_cycles'])
    rul = int(latest['RUL'])
    status = _get_status(rul)

    sensor_readings = {
        s: f"{latest[s]:.3f}"
        for s in KEY_SENSORS
        if s in latest.index}
    
    sensor_str = "\n".join([f"{name}: {value}" for name, value in sensor_readings.items()])

    key_sensors_name = "\n".join([f"{s}: {SENSOR_DESCRIPTIONS.get(s,"Unknown")}"
                                  for s in KEY_SENSORS])

    return (
        f"Engine {idx}:\n"
        f"Current cycle: {current_cycle}\n"
        f"RUL: {rul} cycles\n"
        f"Status: {status}\n"
        f"Key sensors readings (normalized 0-1):\n{sensor_str}"
        f"Key sensors names: {key_sensors_name}\n")

@tool
def get_critical_engines() -> str:
    """
    Returns a list of all engines in CRITICAL or WARNING state.
    CRITICAL = RUL below 30 cycles.
    WARNING  = RUL between 30 and 100 cycles.
    Use this when the user asks which engines need attention,
    are at risk of failure, or require maintenance scheduling.
    """
    df = _load_data()
    latest = df.groupby('unit_number').last().reset_index()

    critical = latest[latest['RUL'] < 30][['unit_number', 'RUL']].sort_values('RUL')
    warning  = latest[(latest['RUL'] >= 30) & (latest['RUL'] < 100)][['unit_number', 'RUL']].sort_values('RUL')

    result = "=== CRITICAL ENGINES (RUL < 30) ===\n"
    if critical.empty:
        result += "  None\n"
    else:
        for _, row in critical.iterrows():
            result += f"  Engine {int(row['unit_number'])}: {int(row['RUL'])} cycles remaining\n"

    result += "\n=== WARNING ENGINES (RUL 30-100) ===\n"
    if warning.empty:
        result += "  None\n"
    else:
        for _, row in warning.iterrows():
            result += f"  Engine {int(row['unit_number'])}: {int(row['RUL'])} cycles remaining\n"

    return result

@tool
def detect_anomalies(unit_id: int, z_threshold: float = 2.5) -> str:
    """
    Detects anomaly sensor readings for a specific engine using z-score analysis.
    The Z-score (also known as a standard score) is a statistical measure 
    that tells us how many standard deviations a specific engine's sensor 
    reading is from the fleet-wide average - calculated using the formula:
    Z = (x - mu) / sigma, where:
    x is the current sensor value of the specific engine,
    (mu) is the mean (average) value across the entire fleet,
    (sigma) is the standard deviation of the fleet.
    A sensor reading is considered an anomaly when it deviates significantly
    from the fleet average for that sensor at a similar point in engine life.
    Use this when the user asks if something is wrong with a specific engine,
    or wants to know which sensors are behaving unusually.

    Args:
        unit_id     : The engine unit number.
        z_threshold : Z-score threshold for anomaly detection (default 2.5).
                      Higher = less sensitive, lower = more sensitive.
    """
    df = _load_data()
    engine = df[df['unit_number'] == unit_id]

    if engine.empty:
        return f"Engine {unit_id} not found in dataset."

    #Latest cycle compared to the mean and the std of the entire fleet
    fleet_stats = df.groupby('unit_number').last().reset_index()
    fleet_mean = fleet_stats[KEY_SENSORS].mean()
    fleet_std = fleet_stats[KEY_SENSORS].std().replace(0, np.nan)

    latest = engine.iloc[-1]
    engine_vals = latest[KEY_SENSORS]
    z_scores = ((engine_vals - fleet_mean) / fleet_std).abs()

    anomalies = z_scores[z_scores > z_threshold].dropna()

    if anomalies.empty:
        return (
            f"Engine {unit_id}: No anomalies detected "
            f"(z-score threshold: {z_threshold}).\n"
            f"All key sensors are within normal fleet range.")

    result = f"Engine {unit_id}: {len(anomalies)} anomalous sensor(s) detected:\n"
    for sensor, z in anomalies.sort_values(ascending=False).items():
        actual = f"{engine_vals[sensor]:.3f}"
        fleet_avg = f"{fleet_mean[sensor]:.3f}"
        result += (
            f"{sensor}: z-score={z:.2f} "
            f"(engine={actual}, fleet avg={fleet_avg})\n")

    return result

@tool
def compare_engines(unit_id_1: int, unit_id_2: int) -> str:
    """
    Compares the health and sensor readings of two engines.
    Use this when the user wants to compare two specific engines,
    understand which one is in better condition, or see differences
    in their degradation patterns.

    Args:
        unit_id_1: First engine unit number.
        unit_id_2: Second engine unit number.
    """
    df = _load_data()

    results = {}
    for uid in [unit_id_1, unit_id_2]:
        engine = df[df['unit_number'] == uid]
        if engine.empty:
            return f"Engine {uid} not found in dataset."
        latest = engine.iloc[-1]
        results[uid] = latest

    lines = [f"=== Engine Comparison: {unit_id_1} vs {unit_id_2} ===\n"]

    # RUL and status
    for uid in [unit_id_1, unit_id_2]:
        rul = int(results[uid]['RUL'])
        cycle = int(results[uid]['time_cycles'])
        status = _get_status(rul)
        lines.append(f"Engine {uid}:")
        lines.append(f"Cycle: {cycle}")
        lines.append(f"RUL: {rul} cycles")
        lines.append(f"Status: {status}")

    # Compare sensors
    lines.append("\nSensor comparison (normalized 0-1):")
    lines.append(f"{'Sensor':<8} {'Engine '+ str(unit_id_1):>12} {'Engine ' + str(unit_id_2):>12} {'Diff':>10}")
    lines.append("  " + "-" * 46)

    for s in KEY_SENSORS:
        if s in results[unit_id_1].index and s in results[unit_id_2].index:
            v1 = results[unit_id_1][s]
            v2 = results[unit_id_2][s]
            diff = v1 - v2
            flag = " ← higher" if abs(diff) > 0.1 else ""
            lines.append(f"  {s:<8} {v1:>12.3f} {v2:>12.3f} {diff:>+10.3f}{flag}")

    rul1 = int(results[unit_id_1]['RUL'])
    rul2 = int(results[unit_id_2]['RUL'])
    better = unit_id_1 if rul1 > rul2 else unit_id_2
    worse = unit_id_1 if rul1 < rul2 else unit_id_2
    lines.append(f"\nVerdict: Engine {better} is in better condition  - ({max(rul1, rul2)} cycles remaining for engine {better} vs {min(rul1, rul2)} cycles remaining for engine {worse}).")

    return "\n".join(lines)


@tool
def get_sensor_trend(unit_id: int, sensor: str) -> str:
    """
    Returns the trend of a specific sensor over the engine's lifetime.
    Summarizes the sensor's trajectory in early, mid, and late life stages.
    Use this when the user asks how a sensor has changed over time
    or wants to understand the degradation pattern of a specific engine.

    Args:
        unit_id : The engine unit number.
        sensor  : Sensor name, e.g. 's_11', 's_4', 's_2'.
    """
    df = _load_data()
    engine = df[df['unit_number'] == unit_id]

    if engine.empty:
        return f"Engine {unit_id} not found in dataset."

    if sensor not in engine.columns:
        return f"Sensor '{sensor}' not found. Available: {KEY_SENSORS}"

    total_cycles = len(engine)
    third = total_cycles // 3

    early = engine.iloc[:third][sensor].mean()
    mid = engine.iloc[third:2*third][sensor].mean()
    late = engine.iloc[2*third:][sensor].mean()
    trend = late - early

    direction = "increasing ↑" if trend > 0.05 else "decreasing ↓" if trend < -0.05 else "stable →"

    return (
        f"Engine {unit_id} – Sensor {sensor} trend over {total_cycles} cycles:\n"
        f"Early life avg: {early:.3f}\n"
        f"Mid life avg: {mid:.3f}\n"
        f"Late life avg: {late:.3f}\n"
        f"Overall trend: {direction} (delta={trend:+.3f})\n"
        f"Note: Values normalized to [0,1]."
        f"A strong trend toward 0 or 1 indicates degradation."
    )





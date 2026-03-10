# NASA Turbofan Engine Health Monitoring System

> AI-powered multi-agent system for turbofan engine degradation analysis based on NASA CMAPSS dataset.

## Overview

This project implements an end-to-end AI system for monitoring and analyzing the health of turbofan engines. Using the **NASA CMAPSS FD001** run-to-failure dataset, the system enables natural language querying of engine health status, anomaly detection, and Remaining Useful Life (RUL) estimation with a multi-agent architecture built with LangGraph.

The application simulates a real-time monitoring scenario by randomly sampling each engine's observation frames.

---

## Project Structure

```
TurbofanAI-agent/
│
├── data/
│   ├── raw/                    # NASA CMAPSS raw .txt files
│   └── processed/              # preprocessed CSV files
│
├── src/
│   ├── preprocessing.py        # Data loading, cleaning, RUL calculation, feature engineering
│   ├── tools.py                # Domain-specific tool functions for agents
│   ├── agent.py                # LangGraph multi-agent graph definition
│   ├── prompts.py              # system prompts with turbofan domain knowledge
│   └── test_agents.py          # Simple prompt testing tool
│
├── .env                        # Personal API key
├── requirements.txt
└── README.md
```

---

## Architecture

```
                                          User Question
                                                │
                                                ▼
                                          ┌─────────────┐
                                          │  SUPERVISOR │  ← routes question to the right specialist
                                          └──────┬──────┘
                                                 │
                                    ┌────────────┴─────────────┐
                                    │            │             │
                                    ▼            ▼             ▼
                                  Data        Anomaly      Degradation
                                  Inspector   Detector      Analyst
                                    │            │             │
                                    └────────────┴─────────────┘
                                                 │
                                                 ▼
                                            Final Answer
```
Each specialist agent has access to domain-specific tools (Python functions) and a system prompt with NASA turbofan specific knowledge. 
A **LangGraph StateGraph** manages message flow and conversation history across agents.

---

## Agents

| Agent | Responsibility | Tools |
|---|---|---|
| **Supervisor** | Routes user query to the correct specialist | - |
| **Data Inspector** | Fleet statistics, engine summaries, sensor info | `get_engines_set_summary`, `get_engine_stats` |
| **Anomaly Detector** | Z-score anomaly detection, critical engine identification | `detect_anomalies`, `get_critical_engines` |
| **Degradation Analyst** | RUL trends, sensor trajectory analysis, engine comparison | `get_sensor_trend`, `compare_engines` |


---

## Preprocessing Pipeline

1. **Load & inspect** - parse raw NASA CMAPSS `.txt` files, assign column names
2. **Remove low-variance sensors** - drop sensors with std < 0.01
3. **Normalize** - MinMaxScaler for all sensor columns
4. **Calculate RUL** - `RUL = max_cycle - current_cycle` per engine, capped at 125 cycles (standard CMAPSS benchmark)


## Usage Examples

```text
=== TEST 1 ===

User:
  Give me the summary of the entire dataset

Supervisor:
  routing → data_inspector

Data Inspector:
  Total Engines: 100
  Average RUL: 107.0 cycles

  Key Sensors:
    s_2  - Total temperature at fan inlet (T2) [°R]
    s_3  - Total temperature at LPC outlet (T24) [°R]
    s_4  - Total temperature at HPC outlet (T30) [°R]
    s_7  - Static pressure at fan inlet (P2) [psia]
    s_8  - Physical fan speed (Nf) [rpm]
    s_11 - Static pressure at HPC outlet (P30) [psia]
    s_12 - Fuel flow / Ps30 ratio (Wf/Ps30)
    s_15 - Bypass ratio (BPR)
    s_17 - Bleed enthalpy

  Engine Health Status:
    Healthy (RUL ≥ 100)      : 57
    Warning (30 ≤ RUL < 100) : 43
    Critical (RUL < 30)      : 0

=== TEST 2: ===
Are there any anomalies in the engine number 4?
[Supervisor] routing to: [anomaly_detector]
Engine 4 does not have any anomalous sensor readings based on the z-score threshold of 2.5. All key sensors are within the normal range compared to the fleet average.

If you have any other specific concerns or need to check another engine, let me know!

=== TEST 3: ===
What is z-score?
[Supervisor] routing to: [anomaly_detector]
Certainly! The z-score, also known as a standard score, is a statistical measure that indicates how many standard deviations a particular data point is from the mean of a dataset. It is calculated using the following formula:

\[ \text{Z} = \frac{x - \mu}{\sigma} \]

Where:
- \( x \) is the value of the data point (in this case, the current sensor reading of a specific engine).
- \( \mu \) (mu) is the mean (average) value of the dataset (in this case, the average sensor reading across the entire fleet of engines).
- \( \sigma \) (sigma) is the standard deviation of the dataset (in this case, the standard deviation of the sensor readings across the entire fleet of engines).

### Interpretation of Z-Score:
- **Z = 0**: The data point is exactly at the mean.
- **Z > 0**: The data point is above the mean.
- **Z < 0**: The data point is below the mean.
- **|Z| > 2.5**: The data point is considered an outlier or anomaly. This threshold is commonly used in anomaly detection to flag values that are significantly different from the mean.

In the context of turbofan engine health monitoring:
- A z-score greater than 2.5 or less than -2.5 indicates that the sensor reading for a specific engine is significantly different from the average reading across the fleet. This could be a sign of potential issues or degradation in the engine.

For example, if the z-score for the HPC outlet temperature (s_4) is 3.0, it means that the HPC outlet temperature for that specific engine is 3 standard deviations higher than the average HPC outlet temperature across the fleet. This could indicate that the HPC is experiencing higher temperatures, which might suggest compressor degradation or other issues that require further investigation.

If you have any more questions or need further clarification, feel free to ask!

=== TEST 4: ===
Is engine number 5 better than engine number 4 then?
[Supervisor] routing to: [degradation_analyst]
### Comparison of Engine 4 and Engine 5

#### Engine 4:
- **Current Cycle:** 117
- **Remaining Useful Life (RUL):** 72 cycles
- **Status:** WARNING

#### Engine 5:
- **Current Cycle:** 64
- **Remaining Useful Life (RUL):** 205 cycles
- **Status:** HEALTHY

### Sensor Comparison (Normalized 0-1):

| Sensor | Engine 4 | Engine 5 | Difference |
|--------|----------|----------|------------|
| s_2    | 0.530    | 0.358    | +0.172     | (higher in Engine 4)
| s_3    | 0.370    | 0.330    | +0.039     | (higher in Engine 4)
| s_4    | 0.455    | 0.240    | +0.215     | (higher in Engine 4)
| s_7    | 0.461    | 0.688    | -0.227     | (higher in Engine 5)
| s_8    | 0.242    | 0.121    | +0.121     | (higher in Engine 4)
| s_11   | 0.280    | 0.179    | +0.101     | (higher in Engine 4)
| s_12   | 0.667    | 0.761    | -0.094     | (higher in Engine 5)
| s_15   | 0.405    | 0.252    | +0.153     | (higher in Engine 4)
| s_17   | 0.500    | 0.250    | +0.250     | (higher in Engine 4)

### Verdict:
- **Engine 5 is in better condition.**
  - **RUL:** Engine 5 has 205 cycles remaining, while Engine 4 has only 72 cycles remaining.
  - **Status:** Engine 5 is classified as HEALTHY,

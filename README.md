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



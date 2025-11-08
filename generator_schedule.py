import pulp
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


NODE_LIST: List[str] = [
    "GFD0331",
    "GYT0331",
    "HAY0111",
    "HAY0331",
    "MLG0111",
    "MLG0331",
    "MST0331",
    "PNI0331",
    "TKR0331",
    "UHT0331",
]

BATTERY_CONFIG: Dict[str, Any] = {
    "battery_hours": 2.0,
    "num_batteries": 1,
    "E_max": 2000.0,
    "E_min": 200.0,
    "E_0": 200.0,
    "eta_ch": 0.95,
    "eta_dis": 0.95,
    "lamda_bat": 0.05,
    "M": 10000.0,
    "D": 5.0 / 60.0,
    "e_da_max": 1e8,
}

DEFAULT_INPUT_CSV = Path("model_inputs_0811025/model_input_data_nov8.csv")
DEFAULT_OUTPUT_CSV = Path("model_outputs_08112025/generator_schedule_results.csv")

# Optional toggle for a lightweight test run.
TEST_MODE_ENABLED = True
TEST_MODE_NODES: Optional[List[str]] = ["PNI0331"]
TEST_MODE_START_TIMENUMBER = 49011
TEST_MODE_END_TIMENUMBER = 57472

def build_generator_schedule(params):
    """
    Build the deterministic generator scheduling optimization model using PuLP.

    Parameters
    ----------
    params : dict
        Dictionary containing all model parameters, sets, and scalars.

    Returns
    -------
    tuple[pulp.LpProblem, dict]
        The constructed PuLP model and a dictionary of decision variables.
    """
    # Unpack sets
    time_periods = params["time_periods"]
    battery = params["battery"]

    # Unpack parameters
    price_gxp = params["price_gxp"]
    price_6sec = params["price_6sec"]
    price_60sec = params["price_60sec"]
    

    
    call = params["call"]

    p_w = params["p_w"]
    p_w_bar = params["p_w_bar"]
    p_w_underline = params["p_w_underline"]
    E_0 = params["E_0"]
    E_max = params["E_max"]
    E_min = params["E_min"]
    eta_ch = params["eta_ch"]
    eta_dis = params["eta_dis"]
    DR_bar = params["DR_bar"]
    CR_bar = params["CR_bar"]
    lamda_bat = params["lamda_bat"]
    

    M = params.get("M", 10000)
    D = params.get("D", 1)
    e_da_max = params.get("e_da_max", 100)

    # Create model
    model = pulp.LpProblem("generator_Deterministic_Scheduling", pulp.LpMaximize)

    # Decision variables
    e_da = pulp.LpVariable.dicts("e_da", time_periods, cat="Continuous")
    e_s_da = pulp.LpVariable.dicts("e_s_da", time_periods, lowBound=0, cat="Continuous")
    e_b_da = pulp.LpVariable.dicts("e_b_da", time_periods, upBound=0, cat="Continuous")
    J_s = pulp.LpVariable.dicts("J_s", time_periods, cat="Binary")

    e_6sec = pulp.LpVariable.dicts("e_6sec", time_periods, lowBound=0, cat="Continuous")
    e_60sec = pulp.LpVariable.dicts("e_60sec", time_periods, lowBound=0, cat="Continuous")

    e_ch = pulp.LpVariable.dicts("e_ch", (battery, time_periods), lowBound=0, cat="Continuous")
    e_da_dis = pulp.LpVariable.dicts("e_da_dis", (battery, time_periods), lowBound=0, cat="Continuous")
    e_6sec_dis = pulp.LpVariable.dicts("e_6sec_dis", (battery, time_periods), lowBound=0, cat="Continuous")
    e_60sec_dis = pulp.LpVariable.dicts("e_60sec_dis", (battery, time_periods), lowBound=0, cat="Continuous")
    u_ch = pulp.LpVariable.dicts("u_ch", (battery, time_periods), cat="Binary")
    e_soc = pulp.LpVariable.dicts("e_soc", (battery, time_periods), lowBound=0, cat="Continuous")

    cost_deg = pulp.LpVariable.dicts("cost_deg", (battery, time_periods), lowBound=0, cat="Continuous")
    # Endogenous total energy over horizon per battery (was a parameter; now a variable)
    E_dem = pulp.LpVariable.dicts("E_dem", battery, lowBound=0, cat="Continuous")

    # Objective: maximize total profit
    model += (
        # DA net revenue; e_da[t] is energy (kWh), so no D factor here.
        pulp.lpSum([price_gxp[t] * e_da[t] for t in time_periods])
        + pulp.lpSum([call[t] * price_6sec[t] * e_6sec[t] for t in time_periods])
        + pulp.lpSum([call[t] * price_60sec[t] * e_60sec[t] for t in time_periods])
        - pulp.lpSum([cost_deg[n][t] for n in battery for t in time_periods])
        
    ), "Total_Profit"

    # Constraints
    # generator energy balance
    for t in time_periods:
        model += e_da[t] == p_w[t] * D + pulp.lpSum(
            [
                (e_da_dis[n][t] + e_6sec_dis[n][t] + e_60sec_dis[n][t]) * eta_dis
                - e_ch[n][t] / eta_ch
                for n in battery
            ]
        )

    # DA market buy/sell logic
    for t in time_periods:
        model += e_da[t] == e_s_da[t] + e_b_da[t]
        model += e_s_da[t] <= e_da_max * J_s[t]
        model += e_b_da[t] >= -e_da_max * (1 - J_s[t])

    # Reserve/DA capacity (energy per 5-min interval in kWh)
    for t in time_periods:
        e_cap_bar = p_w_bar[t] * D + pulp.lpSum([DR_bar[n] * D * eta_dis for n in battery])
        # DA sales allowed only when J_s[t]=1, bounded by capacity
        model += e_s_da[t] <= e_cap_bar * J_s[t]

    # Reserve market conditions (energy per 5-min interval in kWh)
    for t in time_periods:
        e_cap_underline = p_w_underline[t] * D + pulp.lpSum(
            [DR_bar[n] * D * 0.1 * eta_dis for n in battery]
        )
        model += e_6sec[t] * call[t] == pulp.lpSum([e_6sec_dis[n][t] * eta_dis for n in battery])
        model += e_60sec[t] * call[t] == pulp.lpSum([e_60sec_dis[n][t] * eta_dis for n in battery])
        # Reserve delivery bound independent of DA sell toggle
        model += e_6sec[t] + e_60sec[t] <= e_cap_underline

    # Battery net energy over entire horizon (always connected)
    for n in battery:
        model += E_dem[n] == pulp.lpSum(
            [e_ch[n][t] / eta_ch - e_da_dis[n][t] - e_6sec_dis[n][t] - e_60sec_dis[n][t] for t in time_periods]
        )

    # Simultaneous charging/discharging prevention (power limits converted to energy per interval via D)
    for n in battery:
        for t in time_periods:
            model += e_da_dis[n][t] + e_6sec_dis[n][t] + e_60sec_dis[n][t] <= DR_bar[n] * D * (1 - u_ch[n][t])
            model += e_ch[n][t] <= CR_bar[n] * D * u_ch[n][t]

    # Battery state of charge (SoC) over entire horizon
    for n in battery:
        for t in time_periods:
            if t == 0:
                model += (
                    e_soc[n][t]
                    == E_0[n] + e_ch[n][t] * eta_ch - e_da_dis[n][t] - e_6sec_dis[n][t] - e_60sec_dis[n][t]
                )
            else:
                model += (
                    e_soc[n][t]
                    == e_soc[n][t - 1]
                    + e_ch[n][t] * eta_ch
                    - e_da_dis[n][t]
                    - e_6sec_dis[n][t]
                    - e_60sec_dis[n][t]
                )

    # Degradation cost calculation
    for n in battery:
        for t in time_periods:
            model += cost_deg[n][t] == lamda_bat * (e_da_dis[n][t] + e_6sec_dis[n][t] + e_60sec_dis[n][t])

    # Battery energy bounds over entire horizon
    for n in battery:
        for t in time_periods:
            model += e_soc[n][t] >= E_min[n]
            model += e_soc[n][t] <= E_max[n]

    # Non-negativity of variables enforced via bounds; no plug-in/out zeros since always plugged

    variables = {
        "e_da": e_da,
        "e_s_da": e_s_da,
        "e_b_da": e_b_da,
        "J_s": J_s,
        "e_6sec": e_6sec,
        "e_60sec": e_60sec,
        "e_ch": e_ch,
        "e_da_dis": e_da_dis,
        "e_6sec_dis": e_6sec_dis,
        "e_60sec_dis": e_60sec_dis,
        "u_ch": u_ch,
        "e_soc": e_soc,
        "cost_deg": cost_deg,
        "E_dem": E_dem,
    }

    return model, variables


def solve_model(model):
    """
    Solve the given PuLP model using the default solver.

    Returns
    -------
    int
        The model status after solving.
    """
    model.solve()
    return model.status


def load_input_data(
    csv_path: Path,
    nodes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load the raw input dataset and return it filtered for the requested nodes.
    The returned DataFrame is sorted by node and timenumber and includes a
    sequential period number per node.
    """
    df = pd.read_csv(csv_path)

    if "TradingPeriodNumber" in df.columns:
        df = df.drop(columns=["TradingPeriodNumber"])

    if "timenumber" not in df.columns:
        raise KeyError("Input data must contain a 'timenumber' column.")

    df["timenumber"] = pd.to_numeric(df["timenumber"], errors="coerce")
    df = df.dropna(subset=["timenumber"])
    df["timenumber"] = df["timenumber"].astype(int)

    effective_nodes: Optional[Sequence[str]] = nodes
    start_tn: Optional[int] = None
    end_tn: Optional[int] = None

    if TEST_MODE_ENABLED:
        effective_nodes = TEST_MODE_NODES or nodes
        start_tn = TEST_MODE_START_TIMENUMBER
        end_tn = TEST_MODE_END_TIMENUMBER

    if effective_nodes is not None:
        df = df[df["node"].isin(effective_nodes)]

    if start_tn is not None and end_tn is not None:
        df = df[(df["timenumber"] >= start_tn) & (df["timenumber"] <= end_tn)]

    df = df.sort_values(["node", "timenumber"]).reset_index(drop=True)
    df["period_number"] = df.groupby("node").cumcount() + 1
    return df


def _series_to_time_dict(series: pd.Series, time_periods: List[int]) -> Dict[int, float]:
    cleaned = series.reset_index(drop=True).fillna(0.0)
    return {t: float(cleaned.iloc[t]) for t in time_periods}


def build_params_for_node(
    node_df: pd.DataFrame,
    battery_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Assemble the parameter dictionary expected by build_generator_schedule for a
    single node.
    """
    node_df = node_df.reset_index(drop=True)
    time_periods = list(range(len(node_df)))
    battery_ids = list(range(int(battery_config["num_batteries"])))

    battery_hours = float(battery_config["battery_hours"])
    e_max = float(battery_config["E_max"])
    e_min = float(battery_config["E_min"])
    e_0 = float(battery_config.get("E_0", e_min))

    dr_bar_value = e_max / battery_hours if battery_hours else 0.0
    cr_bar_value = e_max / battery_hours if battery_hours else 0.0

    params: Dict[str, Any] = {
        "time_periods": time_periods,
        "battery": battery_ids,
        "price_gxp": _series_to_time_dict(node_df["DollarsPerkilowatthour"], time_periods),
        "price_6sec": _series_to_time_dict(node_df["price_6sec"], time_periods),
        "price_60sec": _series_to_time_dict(node_df["price_60sec"], time_periods),
        "call": _series_to_time_dict(
            node_df["call"] if "call" in node_df.columns else pd.Series([0] * len(node_df)),
            time_periods,
        ),
        "p_w": _series_to_time_dict(node_df["power"], time_periods),
        "p_w_bar": _series_to_time_dict(node_df["power"], time_periods),
        "p_w_underline": _series_to_time_dict(node_df["power"], time_periods),
        "E_0": {n: e_0 for n in battery_ids},
        "E_max": {n: e_max for n in battery_ids},
        "E_min": {n: e_min for n in battery_ids},
        "eta_ch": float(battery_config["eta_ch"]),
        "eta_dis": float(battery_config["eta_dis"]),
        "DR_bar": {n: dr_bar_value for n in battery_ids},
        "CR_bar": {n: cr_bar_value for n in battery_ids},
        "lamda_bat": float(battery_config["lamda_bat"]),
        "M": float(battery_config["M"]),
        "D": float(battery_config["D"]),
        "e_da_max": float(battery_config["e_da_max"]),
    }

    return params


def _safe_value(var: Any) -> Optional[float]:
    value = pulp.value(var)
    return float(value) if value is not None else None


def collect_node_results(
    node: str,
    node_df: pd.DataFrame,
    model: pulp.LpProblem,
    variables: Dict[str, Any],
    params: Dict[str, Any],
    solver_status: str,
) -> pd.DataFrame:
    """
    Convert optimisation results for a single node into long-format rows.
    """
    node_df = node_df.reset_index(drop=True)
    period_lookup = node_df[["period_number", "timenumber"]]
    rows: List[Dict[str, Any]] = []

    time_metrics = ["e_da", "e_s_da", "e_b_da", "J_s", "e_6sec", "e_60sec"]
    battery_time_metrics = [
        "e_ch",
        "e_da_dis",
        "e_6sec_dis",
        "e_60sec_dis",
        "u_ch",
        "e_soc",
        "cost_deg",
    ]

    for metric in time_metrics:
        var_map = variables[metric]
        for t in params["time_periods"]:
            period_info = period_lookup.iloc[t]
            rows.append(
                {
                    "node": node,
                    "period_number": int(period_info["period_number"]),
                    "timenumber": int(period_info["timenumber"]),
                    "battery": None,
                    "metric": metric,
                    "value": _safe_value(var_map[t]),
                    "solver_status": solver_status,
                }
            )

    for metric in battery_time_metrics:
        var_map = variables[metric]
        for n in params["battery"]:
            for t in params["time_periods"]:
                period_info = period_lookup.iloc[t]
                rows.append(
                    {
                        "node": node,
                        "period_number": int(period_info["period_number"]),
                        "timenumber": int(period_info["timenumber"]),
                        "battery": n,
                        "metric": metric,
                        "value": _safe_value(var_map[n][t]),
                        "solver_status": solver_status,
                    }
                )

    return pd.DataFrame(rows)


def run_schedule_for_nodes(
    df: pd.DataFrame,
    nodes: Sequence[str],
    battery_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build, solve, and collect schedule results for all requested nodes.
    """
    results: List[pd.DataFrame] = []

    for node in nodes:
        node_df = df[df["node"] == node].reset_index(drop=True)
        if node_df.empty:
            print(f"Node '{node}' not found in input data. Skipping.")
            continue

        params = build_params_for_node(node_df, battery_config)
        model, variables = build_generator_schedule(params)
        status_code = solve_model(model)
        status_label = pulp.LpStatus.get(status_code, str(status_code))
        print(f"Node '{node}': solver status = {status_label}")

        node_results = collect_node_results(
            node=node,
            node_df=node_df,
            model=model,
            variables=variables,
            params=params,
            solver_status=status_label,
        )
        results.append(node_results)

    if not results:
        return pd.DataFrame(
            columns=[
                "node",
                "period_number",
                "timenumber",
                "battery",
                "metric",
                "value",
                "solver_status",
            ]
        )

    combined = pd.concat(results, ignore_index=True)
    return combined


def main(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    nodes: Optional[Sequence[str]] = None,
) -> None:
    """
    Entry point for running the generator scheduling optimisation across nodes.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    target_nodes = list(nodes) if nodes is not None else NODE_LIST

    data = load_input_data(input_csv, target_nodes)
    results = run_schedule_for_nodes(data, target_nodes, BATTERY_CONFIG)

    if results.empty:
        print("No optimisation results to write.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    print(f"Saved optimisation results to '{output_csv}'.")


if __name__ == "__main__":
    main()
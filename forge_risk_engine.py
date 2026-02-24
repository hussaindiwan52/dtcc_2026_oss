from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Iterable
"""Core risk intelligence engine for FORGE.

This module turns the notebook logic into reusable functions for:
- Sample data generation
- Constraint-based remediation scheduling
- Budget timeline calculation
- D3 chart code generation for UI rendering
"""


import json
import math
import numbers
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pulp
from ortools.sat.python import cp_model

SolverName = Literal["cp-sat", "pulp"]

BASE_SAMPLE_DATA: list[dict[str, Any]] = [
    {"ID": 1, "Score": 25, "Res_Score": 5, "CTA": 5000, "LeadTime": 3, "Capacity": 2, "Predecessors": []},
    {"ID": 2, "Score": 20, "Res_Score": 4, "CTA": 3000, "LeadTime": 2, "Capacity": 1, "Predecessors": [1]},
    {"ID": 3, "Score": 30, "Res_Score": 10, "CTA": 8000, "LeadTime": 4, "Capacity": 2, "Predecessors": []},
    {"ID": 4, "Score": 15, "Res_Score": 2, "CTA": 2000, "LeadTime": 2, "Capacity": 1, "Predecessors": [2]},
    {"ID": 5, "Score": 40, "Res_Score": 5, "CTA": 10000, "LeadTime": 5, "Capacity": 3, "Predecessors": [3]},
    {"ID": 6, "Score": 10, "Res_Score": 2, "CTA": 1500, "LeadTime": 2, "Capacity": 1, "Predecessors": []},
    {"ID": 7, "Score": 50, "Res_Score": 10, "CTA": 12000, "LeadTime": 6, "Capacity": 3, "Predecessors": [5]},
    {"ID": 8, "Score": 12, "Res_Score": 6, "CTA": 1000, "LeadTime": 1, "Capacity": 1, "Predecessors": [6]},
    {"ID": 9, "Score": 35, "Res_Score": 8, "CTA": 7500, "LeadTime": 4, "Capacity": 2, "Predecessors": []},
    {"ID": 10, "Score": 18, "Res_Score": 3, "CTA": 2500, "LeadTime": 3, "Capacity": 1, "Predecessors": [9]},
    {"ID": 11, "Score": 22, "Res_Score": 5, "CTA": 4000, "LeadTime": 3, "Capacity": 2, "Predecessors": [8]},
    {"ID": 12, "Score": 28, "Res_Score": 7, "CTA": 6000, "LeadTime": 4, "Capacity": 2, "Predecessors": [4]},
    {"ID": 13, "Score": 14, "Res_Score": 2, "CTA": 1800, "LeadTime": 2, "Capacity": 1, "Predecessors": []},
    {"ID": 14, "Score": 45, "Res_Score": 9, "CTA": 11000, "LeadTime": 5, "Capacity": 3, "Predecessors": [13]},
    {"ID": 15, "Score": 16, "Res_Score": 4, "CTA": 2200, "LeadTime": 2, "Capacity": 1, "Predecessors": [14]},
]

CANONICAL_COLUMNS = ["ID", "Score", "Res_Score", "CTA", "LeadTime", "Capacity", "Predecessors"]
COLUMN_ALIASES = {
    "id": "ID",
    "score": "Score",
    "risk_score": "Score",
    "res_score": "Res_Score",
    "residual_score": "Res_Score",
    "residual": "Res_Score",
    "post_remediation_score": "Res_Score",
    "cta": "CTA",
    "cost": "CTA",
    "cost_to_achieve": "CTA",
    "remediation_cost": "CTA",
    "lead_time": "LeadTime",
    "leadtime": "LeadTime",
    "lead_time_days": "LeadTime",
    "duration": "LeadTime",
    "capacity": "Capacity",
    "fte": "Capacity",
    "team_capacity": "Capacity",
    "predecessors": "Predecessors",
    "dependencies": "Predecessors",
}


@dataclass
class AnalysisSummary:
    solver: SolverName
    feasible: bool
    deadline: int
    team_capacity: int
    num_risks: int
    selected_count: int
    total_original_score: float
    target_score_max: float
    achieved_score: float | None
    achieved_reduction: float | None
    achieved_reduction_pct: float | None
    total_cost: float | None
    schedule: list[dict[str, Any]]
    budget_timeline: list[dict[str, Any]]


def get_sample_data(num_records: int = 15) -> pd.DataFrame:
    if num_records <= 0:
        raise ValueError("num_records must be greater than 0.")
    return pd.DataFrame(BASE_SAMPLE_DATA[:num_records]).copy()


def _standardize_risk_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: COLUMN_ALIASES.get(str(col).lower(), col) for col in df.columns}
    out = df.rename(columns=rename_map).copy()

    # If no dependency column is provided, treat all risks as independent.
    if "Predecessors" not in out.columns:
        out["Predecessors"] = [[] for _ in range(len(out))]

    missing = [col for col in CANONICAL_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required risk fields: {missing}")

    out = out[CANONICAL_COLUMNS].copy()

    for col in ["ID", "LeadTime", "Capacity"]:
        out[col] = out[col].astype(int)
    for col in ["Score", "Res_Score", "CTA"]:
        out[col] = out[col].astype(float)

    def normalize_predecessors(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [int(v) for v in value]
        try:
            nan_value = pd.isna(value)
            if isinstance(nan_value, (bool, int)) and bool(nan_value):
                return []
        except TypeError:
            pass
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return [int(value)]
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return []
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return [int(v) for v in parsed]
                if isinstance(parsed, numbers.Real) and not isinstance(parsed, bool):
                    return [int(parsed)]
                if parsed is None:
                    return []
                if isinstance(parsed, str):
                    pieces = [p.strip() for p in parsed.replace(";", ",").split(",") if p.strip()]
                    return [int(p) for p in pieces]
            except json.JSONDecodeError:
                pieces = [p.strip() for p in cleaned.replace(";", ",").split(",") if p.strip()]
                return [int(p) for p in pieces]
            raise ValueError(f"Invalid predecessors value: {value}")
        if isinstance(value, Iterable):
            return [int(v) for v in value]
        raise ValueError(f"Invalid predecessors value: {value}")

    out["Predecessors"] = out["Predecessors"].apply(normalize_predecessors)
    return out


def _ensure_acyclic_dependency_graph(df: pd.DataFrame) -> None:
    graph = {int(row.ID): [int(v) for v in row.Predecessors] for row in df.itertuples(index=False)}
    visiting: set[int] = set()
    visited: set[int] = set()

    def dfs(node: int) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError("Dependency graph contains a cycle.")
        visiting.add(node)
        for pred in graph.get(node, []):
            if pred in graph:
                dfs(pred)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        dfs(node)


def validate_risk_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Risk dataset is empty.")
    if df["ID"].duplicated().any():
        duplicates = df.loc[df["ID"].duplicated(), "ID"].tolist()
        raise ValueError(f"Duplicate risk IDs found: {duplicates}")
    if (df["LeadTime"] <= 0).any():
        raise ValueError("LeadTime must be greater than 0 for all risks.")
    if (df["Capacity"] <= 0).any():
        raise ValueError("Capacity must be greater than 0 for all risks.")
    if (df["CTA"] < 0).any():
        raise ValueError("CTA must be non-negative for all risks.")
    if (df["Score"] < 0).any() or (df["Res_Score"] < 0).any():
        raise ValueError("Risk scores must be non-negative.")
    if (df["Score"] < df["Res_Score"]).any():
        invalid = df.loc[df["Score"] < df["Res_Score"], "ID"].tolist()
        raise ValueError(f"Res_Score cannot exceed Score. Invalid IDs: {invalid}")

    all_ids = set(df["ID"].astype(int).tolist())
    for row in df.itertuples(index=False):
        for pred in row.Predecessors:
            if pred not in all_ids:
                raise ValueError(f"Risk {row.ID} references missing predecessor ID {pred}.")
            if int(pred) == int(row.ID):
                raise ValueError(f"Risk {row.ID} cannot depend on itself.")

    _ensure_acyclic_dependency_graph(df)


def prepare_risk_data(risks: list[dict[str, Any]] | None = None, num_records: int = 15) -> pd.DataFrame:
    df = get_sample_data(num_records) if risks is None else pd.DataFrame(risks)
    return prepare_risk_frame(df)


def prepare_risk_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = _standardize_risk_frame(df)
    validate_risk_data(out)
    return out


def load_risk_file_csv(
    path: str,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    df = pd.read_csv(file_path, sep=delimiter, encoding=encoding)
    return prepare_risk_frame(df)


def load_risk_file_excel(
    path: str,
    *,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return prepare_risk_frame(df)


def solve_with_cp_sat(
    df: pd.DataFrame,
    deadline: int,
    max_capacity: int,
    target_score_max: float,
) -> pd.DataFrame | None:
    model = cp_model.CpModel()
    x = {int(r.ID): model.NewBoolVar(f"select_{int(r.ID)}") for r in df.itertuples(index=False)}
    risk_vars: dict[int, dict[str, Any]] = {}

    for r in df.itertuples(index=False):
        rid = int(r.ID)
        duration = int(r.LeadTime)
        start = model.NewIntVar(0, int(deadline), f"start_{rid}")
        end = model.NewIntVar(0, int(deadline), f"end_{rid}")
        interval = model.NewOptionalIntervalVar(start, duration, end, x[rid], f"interval_{rid}")
        risk_vars[rid] = {"x": x[rid], "start": start, "end": end, "interval": interval}

    total_original = float(df["Score"].sum())
    score_reduction_needed = max(0, math.ceil(total_original - target_score_max))

    reductions: list[Any] = []
    costs: list[Any] = []
    for r in df.itertuples(index=False):
        rid = int(r.ID)
        reduction_val = int(round(float(r.Score) - float(r.Res_Score)))
        reductions.append(x[rid] * reduction_val)
        costs.append(x[rid] * int(round(float(r.CTA))))

    model.Add(sum(reductions) >= score_reduction_needed)

    for r in df.itertuples(index=False):
        rid = int(r.ID)
        for pred_id in r.Predecessors:
            pred = int(pred_id)
            model.Add(x[rid] <= x[pred])
            model.Add(risk_vars[rid]["start"] >= risk_vars[pred]["end"]).OnlyEnforceIf(x[rid])

    intervals = [risk_vars[int(r.ID)]["interval"] for r in df.itertuples(index=False)]
    demands = [int(r.Capacity) for r in df.itertuples(index=False)]
    model.AddCumulative(intervals, demands, int(max_capacity))

    model.Minimize(sum(costs))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    rows: list[dict[str, Any]] = []
    for r in df.itertuples(index=False):
        rid = int(r.ID)
        if solver.Value(x[rid]) != 1:
            continue
        start = solver.Value(risk_vars[rid]["start"])
        end = solver.Value(risk_vars[rid]["end"])
        rows.append(
            {
                "ID": rid,
                "Start_Day": int(start),
                "End_Day": int(end),
                "Cost": float(r.CTA),
                "Reduction": float(r.Score - r.Res_Score),
                "Capacity": int(r.Capacity),
            }
        )

    return pd.DataFrame(rows)


def solve_with_pulp(
    df: pd.DataFrame,
    deadline: int,
    max_capacity: int,
    target_score_max: float,
) -> pd.DataFrame | None:
    prob = pulp.LpProblem("Risk_Budget_Optimization", pulp.LpMinimize)
    times = list(range(deadline + 1))
    risk_ids = [int(v) for v in df["ID"].tolist()]

    x = pulp.LpVariable.dicts("Start", (risk_ids, times), cat="Binary")
    selected = pulp.LpVariable.dicts("Selected", risk_ids, cat="Binary")

    cta_by_id = {int(row.ID): float(row.CTA) for row in df.itertuples(index=False)}
    lead_time_by_id = {int(row.ID): int(row.LeadTime) for row in df.itertuples(index=False)}
    capacity_by_id = {int(row.ID): int(row.Capacity) for row in df.itertuples(index=False)}
    score_by_id = {int(row.ID): float(row.Score) for row in df.itertuples(index=False)}
    residual_by_id = {int(row.ID): float(row.Res_Score) for row in df.itertuples(index=False)}
    predecessors_by_id = {
        int(row.ID): [int(v) for v in row.Predecessors] for row in df.itertuples(index=False)
    }

    prob += pulp.lpSum(selected[i] * cta_by_id[i] for i in risk_ids)

    for i in risk_ids:
        duration = lead_time_by_id[i]
        prob += pulp.lpSum(x[i][t] for t in times) == selected[i]
        for t in range(deadline - duration + 1, deadline + 1):
            prob += x[i][t] == 0

    current_total = float(df["Score"].sum())
    score_reduction = pulp.lpSum(selected[i] * (score_by_id[i] - residual_by_id[i]) for i in risk_ids)
    prob += (current_total - score_reduction) <= float(target_score_max)

    big_m = deadline * 2
    for i in risk_ids:
        for pred in predecessors_by_id[i]:
            pred_duration = lead_time_by_id[pred]
            start_i = pulp.lpSum(t * x[i][t] for t in times)
            start_pred = pulp.lpSum(t * x[pred][t] for t in times)
            prob += selected[i] <= selected[pred]
            prob += start_i >= start_pred + pred_duration - big_m * (1 - selected[i])

    for t in times:
        resource_usage = []
        for i in risk_ids:
            duration = lead_time_by_id[i]
            cap = capacity_by_id[i]
            relevant_starts = [x[i][k] for k in range(max(0, t - duration + 1), t + 1)]
            if relevant_starts:
                resource_usage.append(cap * pulp.lpSum(relevant_starts))
        prob += pulp.lpSum(resource_usage) <= int(max_capacity)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    rows: list[dict[str, Any]] = []
    for i in risk_ids:
        if pulp.value(selected[i]) != 1:
            continue
        start_day = 0
        for t in times:
            if pulp.value(x[i][t]) == 1:
                start_day = t
                break
        rows.append(
            {
                "ID": i,
                "Start_Day": int(start_day),
                "End_Day": int(start_day + lead_time_by_id[i]),
                "Cost": float(cta_by_id[i]),
                "Reduction": float(score_by_id[i] - residual_by_id[i]),
                "Capacity": int(capacity_by_id[i]),
            }
        )

    return pd.DataFrame(rows)


def _to_primitive_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (int, str, bool)) or value is None:
                clean[key] = value
            elif isinstance(value, float):
                clean[key] = float(value)
            else:
                clean[key] = value
        records.append(clean)
    return records


def budget_timeline(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame(columns=["Start_Day", "Cost"])
    timeline = schedule_df.groupby("Start_Day", as_index=False)["Cost"].sum().sort_values("Start_Day")
    return timeline.reset_index(drop=True)


def analyze_risk_plan(
    *,
    risks: list[dict[str, Any]] | None = None,
    num_records: int = 15,
    solver: SolverName = "cp-sat",
    deadline: int = 20,
    team_capacity: int = 4,
    target_remaining_ratio: float = 0.5,
    target_score_max: float | None = None,
) -> AnalysisSummary:
    if deadline <= 0:
        raise ValueError("deadline must be greater than 0.")
    if team_capacity <= 0:
        raise ValueError("team_capacity must be greater than 0.")
    if target_score_max is None and not (0 <= target_remaining_ratio <= 1):
        raise ValueError("target_remaining_ratio must be between 0 and 1.")

    df = prepare_risk_data(risks=risks, num_records=num_records)
    total_original = float(df["Score"].sum())
    target_max = float(target_score_max) if target_score_max is not None else total_original * target_remaining_ratio

    if solver == "cp-sat":
        schedule_df = solve_with_cp_sat(df, deadline, team_capacity, target_max)
    elif solver == "pulp":
        schedule_df = solve_with_pulp(df, deadline, team_capacity, target_max)
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    if schedule_df is None:
        return AnalysisSummary(
            solver=solver,
            feasible=False,
            deadline=deadline,
            team_capacity=team_capacity,
            num_risks=len(df),
            selected_count=0,
            total_original_score=total_original,
            target_score_max=target_max,
            achieved_score=None,
            achieved_reduction=None,
            achieved_reduction_pct=None,
            total_cost=None,
            schedule=[],
            budget_timeline=[],
        )

    schedule_df = schedule_df.sort_values(["Start_Day", "ID"]).reset_index(drop=True)
    budget_df = budget_timeline(schedule_df)
    total_reduction = float(schedule_df["Reduction"].sum())
    achieved_score = total_original - total_reduction
    reduction_pct = (total_reduction / total_original * 100.0) if total_original else 0.0
    total_cost = float(schedule_df["Cost"].sum())

    return AnalysisSummary(
        solver=solver,
        feasible=True,
        deadline=deadline,
        team_capacity=team_capacity,
        num_risks=len(df),
        selected_count=len(schedule_df),
        total_original_score=total_original,
        target_score_max=target_max,
        achieved_score=achieved_score,
        achieved_reduction=total_reduction,
        achieved_reduction_pct=reduction_pct,
        total_cost=total_cost,
        schedule=_to_primitive_records(schedule_df),
        budget_timeline=_to_primitive_records(budget_df),
    )


def benchmark_solvers(
    *,
    iterations: int = 5,
    num_records: int = 15,
    deadline: int = 20,
    team_capacity: int = 4,
    target_remaining_ratio: float = 0.5,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be greater than 0.")

    df = prepare_risk_data(num_records=num_records)
    target_max = float(df["Score"].sum()) * target_remaining_ratio

    solvers: dict[SolverName, Any] = {"cp-sat": solve_with_cp_sat, "pulp": solve_with_pulp}
    durations: dict[str, list[float]] = {"cp-sat": [], "pulp": []}

    for _ in range(iterations):
        for name, fn in solvers.items():
            t0 = time.perf_counter()
            fn(df, deadline, team_capacity, target_max)
            durations[name].append(time.perf_counter() - t0)

    summary: dict[str, Any] = {"iterations": iterations, "results": {}}
    for name, values in durations.items():
        summary["results"][name] = {
            "min_sec": min(values),
            "max_sec": max(values),
            "avg_sec": sum(values) / len(values),
        }
    return summary


def build_budget_d3_code(budget_rows: list[dict[str, Any]], title: str = "Budget Timeline") -> str:
    data = [{"day": int(row["Start_Day"]), "cost": float(row["Cost"])} for row in budget_rows]
    json_blob = json.dumps(data, separators=(",", ":"))
    return f"""const data = {json_blob};
const width = 900;
const height = 420;
const margin = {{ top: 48, right: 24, bottom: 64, left: 80 }};

const root = d3.select("#chart");
root.selectAll("*").remove();
const svg = root.append("svg").attr("width", width).attr("height", height);

if (!data.length) {{
  svg.append("text")
    .attr("x", width / 2)
    .attr("y", height / 2)
    .attr("text-anchor", "middle")
    .attr("font-size", 18)
    .text("No feasible schedule to visualize.");
}} else {{
  const x = d3.scaleBand()
    .domain(data.map((d) => String(d.day)))
    .range([margin.left, width - margin.right])
    .padding(0.22);

  const yMax = d3.max(data, (d) => d.cost) || 1;
  const y = d3.scaleLinear()
    .domain([0, yMax * 1.1])
    .nice()
    .range([height - margin.bottom, margin.top]);

  svg.append("g")
    .attr("transform", `translate(0,${{height - margin.bottom}})`)
    .call(d3.axisBottom(x))
    .call((g) => g.selectAll("text").attr("font-size", 12));

  svg.append("g")
    .attr("transform", `translate(${{margin.left}},0)`)
    .call(d3.axisLeft(y).ticks(6).tickFormat((v) => `$${{d3.format(",")(v)}}`))
    .call((g) => g.selectAll("text").attr("font-size", 12));

  svg.selectAll("rect.budget-bar")
    .data(data)
    .join("rect")
    .attr("class", "budget-bar")
    .attr("x", (d) => x(String(d.day)))
    .attr("y", (d) => y(d.cost))
    .attr("width", x.bandwidth())
    .attr("height", (d) => y(0) - y(d.cost))
    .attr("fill", "#0f766e");

  svg.selectAll("text.bar-label")
    .data(data)
    .join("text")
    .attr("class", "bar-label")
    .attr("x", (d) => (x(String(d.day)) || 0) + x.bandwidth() / 2)
    .attr("y", (d) => y(d.cost) - 8)
    .attr("text-anchor", "middle")
    .attr("font-size", 11)
    .text((d) => `$${{d3.format(",")(Math.round(d.cost))}}`);

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", 30)
    .attr("text-anchor", "middle")
    .attr("font-size", 20)
    .attr("font-weight", 700)
    .text("{title}");

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", height - 16)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .text("Remediation Start Day");

  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -(height / 2))
    .attr("y", 22)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .text("Budget Required ($)");
}}
"""


def build_gantt_d3_code(schedule_rows: list[dict[str, Any]], title: str = "Remediation Gantt Chart") -> str:
    data = [
        {
            "id": int(row["ID"]),
            "start": int(row["Start_Day"]),
            "end": int(row["End_Day"]),
            "cost": float(row["Cost"]),
            "reduction": float(row["Reduction"]),
            "capacity": int(row["Capacity"]),
        }
        for row in schedule_rows
    ]
    json_blob = json.dumps(data, separators=(",", ":"))
    return f"""const data = {json_blob};
const width = 960;
const height = Math.max(420, data.length * 36 + 140);
const margin = {{ top: 52, right: 24, bottom: 56, left: 110 }};

const root = d3.select("#chart");
root.selectAll("*").remove();
const svg = root.append("svg").attr("width", width).attr("height", height);

if (!data.length) {{
  svg.append("text")
    .attr("x", width / 2)
    .attr("y", height / 2)
    .attr("text-anchor", "middle")
    .attr("font-size", 18)
    .text("No feasible schedule to visualize.");
}} else {{
  data.sort((a, b) => a.start - b.start || a.id - b.id);
  const y = d3.scaleBand()
    .domain(data.map((d) => String(d.id)))
    .range([margin.top, height - margin.bottom])
    .padding(0.24);

  const xMax = d3.max(data, (d) => d.end) || 1;
  const x = d3.scaleLinear()
    .domain([0, xMax])
    .nice()
    .range([margin.left, width - margin.right]);

  svg.append("g")
    .attr("transform", `translate(0,${{height - margin.bottom}})`)
    .call(d3.axisBottom(x).ticks(Math.min(12, xMax + 1)));

  svg.append("g")
    .attr("transform", `translate(${{margin.left}},0)`)
    .call(d3.axisLeft(y));

  svg.selectAll("rect.task-bar")
    .data(data)
    .join("rect")
    .attr("class", "task-bar")
    .attr("x", (d) => x(d.start))
    .attr("y", (d) => y(String(d.id)))
    .attr("width", (d) => Math.max(2, x(d.end) - x(d.start)))
    .attr("height", y.bandwidth())
    .attr("rx", 4)
    .attr("fill", "#b45309");

  svg.selectAll("text.task-label")
    .data(data)
    .join("text")
    .attr("class", "task-label")
    .attr("x", (d) => x(d.start) + 8)
    .attr("y", (d) => (y(String(d.id)) || 0) + y.bandwidth() / 2 + 4)
    .attr("font-size", 11)
    .attr("fill", "#ffffff")
    .text((d) => `$${{d3.format(",")(Math.round(d.cost))}} / -${{d.reduction}} risk`);

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", 30)
    .attr("text-anchor", "middle")
    .attr("font-size", 20)
    .attr("font-weight", 700)
    .text("{title}");

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", height - 16)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .text("Day");

  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -(height / 2))
    .attr("y", 20)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .text("Risk ID");
}}
"""


def _markdown_schedule_table(schedule_rows: list[dict[str, Any]]) -> str:
    if not schedule_rows:
        return "_No selected remediation actions._"

    lines = [
        "| Risk ID | Start Day | End Day | Cost ($) | Score Reduction | Capacity |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in schedule_rows:
        lines.append(
            "| {id} | {start} | {end} | {cost:.0f} | {reduction:.0f} | {capacity} |".format(
                id=int(row["ID"]),
                start=int(row["Start_Day"]),
                end=int(row["End_Day"]),
                cost=float(row["Cost"]),
                reduction=float(row["Reduction"]),
                capacity=int(row["Capacity"]),
            )
        )
    return "\n".join(lines)


def build_visual_report_markdown(summary: AnalysisSummary) -> str:
    if not summary.feasible:
        return (
            "No feasible schedule found for the current constraints.\n\n"
            "Try one or more adjustments:\n"
            "- Increase `deadline`\n"
            "- Increase `team_capacity`\n"
            "- Relax the target (`target_remaining_ratio` closer to 1.0)"
        )

    budget_code = build_budget_d3_code(summary.budget_timeline)
    gantt_code = build_gantt_d3_code(summary.schedule)

    return (
        "## FORGE Risk Intelligence Report\n\n"
        "- Optimization Engine: `FORGE Optimizer`\n"
        f"- Selected Risks: `{summary.selected_count}` / `{summary.num_risks}`\n"
        f"- Total Cost: `${summary.total_cost:,.0f}`\n"
        f"- Achieved Score: `{summary.achieved_score:.1f}` (target max `{summary.target_score_max:.1f}`)\n"
        f"- Risk Reduction: `{summary.achieved_reduction:.1f}` (`{summary.achieved_reduction_pct:.1f}%`)\n\n"
        "### Schedule Table\n"
        f"{_markdown_schedule_table(summary.schedule)}\n\n"
        "### Budget Timeline Graph\n"
        "```d3\n"
        f"{budget_code}"
        "```\n\n"
        "### Remediation Gantt Graph\n"
        "```d3\n"
        f"{gantt_code}"
        "```\n"
    )


def summary_to_dict(summary: AnalysisSummary) -> dict[str, Any]:
    return asdict(summary)


def load_oss_vulnerability_excel(
    path: str,
    *,
    sheet_name: str | int = 0,
    required_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Load an OSS vulnerability Excel file and return a normalized pandas DataFrame.

    Expected logical columns (case-insensitive):
    - repository
    - cve id
    - severity
    - packagename
    - version
    - status

    Parameters
    ----------
    path : str
        Path to the Excel file.
    sheet_name : str | int, default 0
        Sheet name or index to read.
    required_columns : Iterable[str] | None
        Optional override for required columns (lowercase, normalized).

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with canonical column names:
        ['repository', 'cve_id', 'severity', 'package_name', 'version', 'status']
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"OSS vulnerability Excel file not found: {file_path}")

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Normalize column names: lowercase, strip, replace spaces
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Column alias mapping
    column_aliases = {
        "repository": "repository",
        "repo": "repository",

        "cve": "cve_id",
        "cve_id": "cve_id",
        "cveid": "cve_id",

        "severity": "severity",
        "severity_level": "severity",

        "packagename": "package_name",
        "package_name": "package_name",
        "package": "package_name",

        "version": "version",
        "package_version": "version",

        "status": "status",
        "vulnerability_status": "status",
    }

    df = df.rename(columns={c: column_aliases.get(c, c) for c in df.columns})

    # Default required columns
    required = set(
        required_columns
        if required_columns is not None
        else {
            "repository",
            "cve_id",
            "severity",
            "package_name",
            "version",
            "status",
        }
    )

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OSS vulnerability columns: {sorted(missing)}")

    # Keep only canonical columns (preserve order)
    canonical_cols = [
        "repository",
        "cve_id",
        "severity",
        "package_name",
        "version",
        "status",
    ]
    df = df[canonical_cols].copy()

    # Basic cleanup
    for col in ["repository", "cve_id", "severity", "package_name", "version", "status"]:
        df[col] = df[col].astype(str).str.strip()

    return df
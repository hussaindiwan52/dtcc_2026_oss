"""MCP server for FORGE (Framework for Operational Risk & Governance Enhancement)."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

try:
    from .forge_risk_engine import (
        AnalysisSummary,
        analyze_risk_plan,
        build_visual_report_markdown,
        load_risk_file_csv,
        load_risk_file_excel,
        prepare_risk_data,
        summary_to_dict,
    )
except ImportError:
    # Allow running as a direct script: python forge_mcp/forge_mcp_server.py
    from forge_risk_engine import (
        AnalysisSummary,
        analyze_risk_plan,
        build_visual_report_markdown,
        load_risk_file_csv,
        load_risk_file_excel,
        prepare_risk_data,
        summary_to_dict,
    )

TransportName = Literal["stdio", "streamable-http"]
INTERNAL_SOLVER = "cp-sat"

LAST_ANALYSIS: AnalysisSummary | None = None
DATASETS: dict[str, dict[str, Any]] = {}
ACTIVE_DATASET_ID: str | None = None
LAST_VISUAL_REPORT: str | None = None
LAST_VISUAL_REPORT_KEY: str | None = None
LAST_VISUAL_REPORT_TS: float | None = None
VISUAL_REPORT_DEDUP_WINDOW_SEC = float(
    os.getenv("FORGE_VISUAL_REPORT_DEDUP_WINDOW_SEC", "15"),
)


def _make_dataset_id(requested: str | None = None) -> str:
    if requested:
        return requested

    idx = 1
    while True:
        candidate = f"dataset_{idx}"
        if candidate not in DATASETS:
            return candidate
        idx += 1


def _dataset_summary(dataset_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "name": entry.get("name"),
        "source_type": entry.get("source_type"),
        "source_path": entry.get("source_path"),
        "loaded_at_utc": entry.get("loaded_at_utc"),
        "row_count": entry.get("row_count"),
        "total_score": entry.get("total_score"),
        "total_cta": entry.get("total_cta"),
    }


def _register_dataset(
    *,
    risks: list[dict[str, Any]],
    source_type: str,
    source_path: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    global ACTIVE_DATASET_ID

    chosen_id = _make_dataset_id(dataset_id)
    total_score = float(sum(float(row["Score"]) for row in risks))
    total_cta = float(sum(float(row["CTA"]) for row in risks))

    DATASETS[chosen_id] = {
        "name": chosen_id,
        "source_type": source_type,
        "source_path": source_path,
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(risks),
        "total_score": total_score,
        "total_cta": total_cta,
        "risks": risks,
    }
    ACTIVE_DATASET_ID = chosen_id
    return _dataset_summary(chosen_id, DATASETS[chosen_id])


def _resolve_risks_input(
    *,
    dataset_id: str | None,
    risks: list[dict[str, Any]] | None,
    num_records: int,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if dataset_id and risks is not None:
        raise ValueError("Provide either dataset_id or risks, not both.")

    chosen_dataset_id = dataset_id or ACTIVE_DATASET_ID
    if chosen_dataset_id:
        entry = DATASETS.get(chosen_dataset_id)
        if entry is None:
            raise ValueError(f"Unknown dataset_id: {chosen_dataset_id}")
        return entry["risks"], chosen_dataset_id

    return risks, None


def _store_last(summary: AnalysisSummary) -> AnalysisSummary:
    global LAST_ANALYSIS, LAST_VISUAL_REPORT, LAST_VISUAL_REPORT_KEY, LAST_VISUAL_REPORT_TS
    LAST_ANALYSIS = summary
    LAST_VISUAL_REPORT = None
    LAST_VISUAL_REPORT_KEY = None
    LAST_VISUAL_REPORT_TS = None
    return summary


def _serialize_risks_for_key(risks: list[dict[str, Any]] | None) -> str | None:
    if risks is None:
        return None
    try:
        return json.dumps(risks, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return repr(risks)


def _build_visual_report_key(
    *,
    num_records: int,
    dataset_id: str | None,
    deadline: int,
    team_capacity: int,
    target_remaining_ratio: float,
    target_score_max: float | None,
    risks: list[dict[str, Any]] | None,
) -> str:
    payload = {
        "num_records": num_records,
        "dataset_id": dataset_id,
        "active_dataset_id": ACTIVE_DATASET_ID if dataset_id is None else None,
        "deadline": deadline,
        "team_capacity": team_capacity,
        "target_remaining_ratio": target_remaining_ratio,
        "target_score_max": target_score_max,
        "risks": _serialize_risks_for_key(risks),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _run_analysis(
    *,
    dataset_id: str | None = None,
    risks: list[dict[str, Any]] | None = None,
    num_records: int = 15,
    deadline: int = 20,
    team_capacity: int = 4,
    target_remaining_ratio: float = 0.5,
    target_score_max: float | None = None,
) -> AnalysisSummary:
    resolved_risks, _ = _resolve_risks_input(
        dataset_id=dataset_id,
        risks=risks,
        num_records=num_records,
    )
    summary = analyze_risk_plan(
        risks=resolved_risks,
        num_records=num_records,
        solver=INTERNAL_SOLVER,
        deadline=deadline,
        team_capacity=team_capacity,
        target_remaining_ratio=target_remaining_ratio,
        target_score_max=target_score_max,
    )
    return _store_last(summary)


def create_mcp_server(
    *,
    host: str,
    port: int,
    streamable_http_path: str,
    log_level: str,
) -> FastMCP:
    mcp = FastMCP(
        name="FORGE Risk Intelligence",
        instructions=(
            "FORGE server for operational, financial and cyber risk intelligence. "
            "Use these tools to optimize remediation schedules and generate D3 visual reports."
        ),
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
        log_level=log_level.upper(),
    )

    @mcp.tool(
        name="forge_load_csv_dataset",
        description=(
            "Load a risk dataset from a CSV file path, validate schema, and store it for later optimization/report calls."
        ),
    )
    def forge_load_csv_dataset(
        file_path: str,
        dataset_id: str | None = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        df = load_risk_file_csv(
            file_path,
            delimiter=delimiter,
            encoding=encoding,
        )
        return _register_dataset(
            risks=df.to_dict(orient="records"),
            source_type="csv",
            source_path=file_path,
            dataset_id=dataset_id,
        )

    @mcp.tool(
        name="forge_load_excel_dataset",
        description=(
            "Load a risk dataset from an Excel file path, validate schema, and store it for later optimization/report calls."
        ),
    )
    def forge_load_excel_dataset(
        file_path: str,
        dataset_id: str | None = None,
        sheet_name: str = "0",
    ) -> dict[str, Any]:
        parsed_sheet: str | int
        parsed_sheet = int(sheet_name) if sheet_name.isdigit() else sheet_name
        df = load_risk_file_excel(file_path, sheet_name=parsed_sheet)
        return _register_dataset(
            risks=df.to_dict(orient="records"),
            source_type="excel",
            source_path=f"{file_path}#sheet={parsed_sheet}",
            dataset_id=dataset_id,
        )

    @mcp.tool(
        name="forge_list_datasets",
        description="List loaded datasets available for FORGE analysis.",
    )
    def forge_list_datasets() -> dict[str, Any]:
        return {
            "active_dataset_id": ACTIVE_DATASET_ID,
            "datasets": [_dataset_summary(dataset_id, entry) for dataset_id, entry in DATASETS.items()],
        }

    @mcp.tool(
        name="forge_set_active_dataset",
        description="Set a loaded dataset as active so tool calls can omit dataset_id.",
    )
    def forge_set_active_dataset(dataset_id: str) -> dict[str, Any]:
        global ACTIVE_DATASET_ID
        if dataset_id not in DATASETS:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")
        ACTIVE_DATASET_ID = dataset_id
        return {
            "active_dataset_id": ACTIVE_DATASET_ID,
            "dataset": _dataset_summary(dataset_id, DATASETS[dataset_id]),
        }

    @mcp.tool(
        name="forge_remove_dataset",
        description="Remove a loaded dataset from FORGE server memory.",
    )
    def forge_remove_dataset(dataset_id: str) -> dict[str, Any]:
        global ACTIVE_DATASET_ID
        if dataset_id not in DATASETS:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")
        DATASETS.pop(dataset_id)
        if ACTIVE_DATASET_ID == dataset_id:
            ACTIVE_DATASET_ID = next(iter(DATASETS), None)
        return {
            "removed_dataset_id": dataset_id,
            "active_dataset_id": ACTIVE_DATASET_ID,
            "remaining_dataset_count": len(DATASETS),
        }

    @mcp.tool(
        name="forge_get_risk_dataset",
        description=(
            "Return FORGE sample risk records (or validated custom records), including total risk score and budget baseline."
        ),
    )
    def forge_get_risk_dataset(
        num_records: int = 15,
        dataset_id: str | None = None,
        risks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        resolved_risks, used_dataset_id = _resolve_risks_input(
            dataset_id=dataset_id,
            risks=risks,
            num_records=num_records,
        )
        df = prepare_risk_data(risks=resolved_risks, num_records=num_records)
        return {
            "dataset_id": used_dataset_id,
            "num_records": int(len(df)),
            "total_score": float(df["Score"].sum()),
            "total_residual_floor": float(df["Res_Score"].sum()),
            "total_cta_if_all_remediated": float(df["CTA"].sum()),
            "risks": df.to_dict(orient="records"),
        }

    @mcp.tool(
        name="forge_optimize_schedule",
        description=(
            "Optimize remediation plan cost while meeting target score, deadline, capacity, and predecessor constraints."
        ),
    )
    def forge_optimize_schedule(
        num_records: int = 15,
        dataset_id: str | None = None,
        deadline: int = 20,
        team_capacity: int = 4,
        target_remaining_ratio: float = 0.5,
        target_score_max: float | None = None,
        risks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        summary = _run_analysis(
            dataset_id=dataset_id,
            risks=risks,
            num_records=num_records,
            deadline=deadline,
            team_capacity=team_capacity,
            target_remaining_ratio=target_remaining_ratio,
            target_score_max=target_score_max,
        )
        result = summary_to_dict(summary)
        result.pop("solver", None)
        result["engine"] = "FORGE Optimizer"
        result["dataset_id"] = dataset_id or ACTIVE_DATASET_ID
        return result

    @mcp.tool(
        name="forge_visual_report",
        description=(
            "Generate a markdown report that includes schedule table plus D3 charts for budget timeline and remediation Gantt."
        ),
    )
    def forge_visual_report(
        num_records: int = 15,
        dataset_id: str | None = None,
        deadline: int = 20,
        team_capacity: int = 4,
        target_remaining_ratio: float = 0.5,
        target_score_max: float | None = None,
        risks: list[dict[str, Any]] | None = None,
    ) -> str:
        global LAST_VISUAL_REPORT, LAST_VISUAL_REPORT_KEY, LAST_VISUAL_REPORT_TS
        report_key = _build_visual_report_key(
            num_records=num_records,
            dataset_id=dataset_id,
            deadline=deadline,
            team_capacity=team_capacity,
            target_remaining_ratio=target_remaining_ratio,
            target_score_max=target_score_max,
            risks=risks,
        )
        now = time.time()
        if (
            LAST_VISUAL_REPORT is not None
            and LAST_VISUAL_REPORT_KEY == report_key
            and LAST_VISUAL_REPORT_TS is not None
            and (now - LAST_VISUAL_REPORT_TS) <= VISUAL_REPORT_DEDUP_WINDOW_SEC
        ):
            return (
                "A full visual report for this same scenario was already generated just now. "
                "Skipping duplicate output to avoid showing repeated charts."
            )

        summary = _run_analysis(
            dataset_id=dataset_id,
            risks=risks,
            num_records=num_records,
            deadline=deadline,
            team_capacity=team_capacity,
            target_remaining_ratio=target_remaining_ratio,
            target_score_max=target_score_max,
        )
        report = build_visual_report_markdown(summary)
        LAST_VISUAL_REPORT = report
        LAST_VISUAL_REPORT_KEY = report_key
        LAST_VISUAL_REPORT_TS = now
        return report

    @mcp.tool(
        name="forge_visual_report_from_last_run",
        description="Return the last generated FORGE visual report (including D3 charts).",
    )
    def forge_visual_report_from_last_run() -> str:
        if LAST_VISUAL_REPORT is not None:
            return LAST_VISUAL_REPORT
        if LAST_ANALYSIS is None:
            return (
                "No previous analysis found.\n\n"
                "Run `forge_visual_report` or `forge_optimize_schedule` first."
            )
        return build_visual_report_markdown(LAST_ANALYSIS)

    @mcp.tool(
        name="forge_benchmark_runtime",
        description="Benchmark FORGE optimization runtime for repeated runs on the same scenario.",
    )
    def forge_benchmark_runtime(
        iterations: int = 5,
        num_records: int = 15,
        dataset_id: str | None = None,
        deadline: int = 20,
        team_capacity: int = 4,
        target_remaining_ratio: float = 0.5,
        target_score_max: float | None = None,
        risks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if iterations <= 0:
            raise ValueError("iterations must be greater than 0.")

        durations: list[float] = []
        last_summary: AnalysisSummary | None = None
        for _ in range(iterations):
            started = time.perf_counter()
            last_summary = _run_analysis(
                dataset_id=dataset_id,
                risks=risks,
                num_records=num_records,
                deadline=deadline,
                team_capacity=team_capacity,
                target_remaining_ratio=target_remaining_ratio,
                target_score_max=target_score_max,
            )
            durations.append(time.perf_counter() - started)

        return {
            "engine": "FORGE Optimizer",
            "iterations": iterations,
            "runtime_sec": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
            },
            "last_run_feasible": bool(last_summary.feasible) if last_summary is not None else False,
        }

    return mcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FORGE Risk Intelligence MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=os.getenv("FORGE_MCP_TRANSPORT", "stdio"),
        help="MCP transport to use.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("FORGE_MCP_HOST", "127.0.0.1"),
        help="Host for streamable-http transport.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FORGE_MCP_PORT", "8765")),
        help="Port for streamable-http transport.",
    )
    parser.add_argument(
        "--streamable-http-path",
        default=os.getenv("FORGE_MCP_STREAMABLE_HTTP_PATH", "/mcp"),
        help="HTTP path for MCP streamable transport.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("FORGE_MCP_LOG_LEVEL", "INFO"),
        help="Server log level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mcp = create_mcp_server(
        host=args.host,
        port=args.port,
        streamable_http_path=args.streamable_http_path,
        log_level=args.log_level,
    )
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

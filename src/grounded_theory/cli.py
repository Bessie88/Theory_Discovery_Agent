"""Command-line entry point for the standalone Grounded Theory state machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .persistence import records_from_csv
from .project import GroundedTheoryProject


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grounded Theory front-end state machine")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init", help="create a GT project from a CSV")
    init.add_argument("--project", required=True)
    init.add_argument("--study-id", required=True)
    init.add_argument("--question", required=True)
    init.add_argument("--csv", required=True)
    init.add_argument("--text-column", default="text_review")
    init.add_argument("--id-column")
    init.add_argument("--open-batch-size", type=int, default=24)
    init.add_argument("--relational-batch-size", type=int, default=48)
    for command in ("task-packet", "status", "export"):
        child = subparsers.add_parser(command)
        child.add_argument("--project", required=True)
        if command == "export":
            child.add_argument("--output")
    submit = subparsers.add_parser("submit", help="validate and commit one agent JSON result")
    submit.add_argument("--project", required=True)
    submit.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    if args.command == "init":
        project = GroundedTheoryProject.create(
            args.project,
            study_id=args.study_id,
            research_question=args.question,
            records=records_from_csv(args.csv, text_column=args.text_column, id_column=args.id_column),
            open_coding_batch_size=args.open_batch_size,
            relational_batch_size=args.relational_batch_size,
        )
        print(json.dumps(project.status(), ensure_ascii=False, indent=2))
        return 0
    project = GroundedTheoryProject.load(args.project)
    if args.command == "task-packet":
        print(json.dumps(project.task_packet(), ensure_ascii=False, indent=2))
    elif args.command == "status":
        print(json.dumps(project.status(), ensure_ascii=False, indent=2))
    elif args.command == "submit":
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        print(json.dumps(project.submit_agent_result(payload), ensure_ascii=False, indent=2))
    else:
        print(project.export_outputs(args.output))
    return 0

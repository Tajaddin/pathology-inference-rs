"""Create the sprint backlog on GitHub from docs/agile/backlog.json.

    python scripts/agile/sync_backlog.py                 labels, milestones and issues (idempotent by title)
    python scripts/agile/sync_backlog.py --board         also a GitHub Projects board with a Sprint field
                                                         (needs `gh auth refresh -s project` once)

Issues are created in file order, so on a fresh repository their numbers match the order in the file.
Everything goes through the `gh` CLI; nothing here holds a token.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BACKLOG = Path(__file__).resolve().parents[2] / "docs" / "agile" / "backlog.json"


def gh(*args: str, input_text: str | None = None) -> str:
    out = subprocess.run(["gh", *args], check=False, input=input_text, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {out.stderr.strip()}")
    return out.stdout


def repo_name() -> str:
    return gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner").strip()


def sync_labels(labels: list[dict[str, str]]) -> None:
    for label in labels:
        gh(
            "label",
            "create",
            label["name"],
            "--color",
            label["color"],
            "--description",
            label["description"],
            "--force",
        )


def sync_milestones(repo: str, milestones: list[dict[str, Any]]) -> dict[str, int]:
    existing = {
        m["title"]: m["number"] for m in json.loads(gh("api", f"repos/{repo}/milestones?state=all&per_page=100"))
    }
    for m in milestones:
        if m["title"] in existing:
            continue
        fields = ["-f", f"title={m['title']}", "-f", f"description={m['description']}"]
        if m.get("due"):
            fields += ["-f", f"due_on={m['due']}T23:59:59Z"]
        created = json.loads(gh("api", f"repos/{repo}/milestones", "-X", "POST", *fields))
        existing[m["title"]] = created["number"]
    return existing


def sync_issues(repo: str, issues: list[dict[str, Any]]) -> list[int]:
    existing = {
        i["title"]: i["number"]
        for i in json.loads(gh("issue", "list", "--state", "all", "--limit", "500", "--json", "title,number"))
    }
    numbers = []
    for issue in issues:
        if issue["title"] in existing:
            numbers.append(existing[issue["title"]])
            continue
        args = ["issue", "create", "--title", issue["title"], "--body-file", "-", "--milestone", issue["milestone"]]
        for label in issue["labels"]:
            args += ["--label", label]
        url = gh(*args, input_text=issue["body"]).strip()
        numbers.append(int(url.rsplit("/", 1)[1]))
        print(f"#{numbers[-1]} {issue['title']}")
    return numbers


def sync_board(repo: str, backlog: dict[str, Any], numbers: list[int]) -> None:
    owner, name = repo.split("/")
    title = f"{name} sprints"
    projects = json.loads(gh("project", "list", "--owner", owner, "--format", "json"))["projects"]
    project = next((p for p in projects if p["title"] == title), None)
    if project is None:
        project = json.loads(gh("project", "create", "--owner", owner, "--title", title, "--format", "json"))
    number = str(project["number"])
    gh("project", "link", number, "--owner", owner, "--repo", repo)
    sprints = [m["title"] for m in backlog["milestones"]]
    fields = json.loads(gh("project", "field-list", number, "--owner", owner, "--format", "json"))["fields"]
    if not any(f["name"] == "Sprint" for f in fields):
        gh(
            "project",
            "field-create",
            number,
            "--owner",
            owner,
            "--name",
            "Sprint",
            "--data-type",
            "SINGLE_SELECT",
            "--single-select-options",
            ",".join(sprints),
        )
    for n in numbers:
        gh("project", "item-add", number, "--owner", owner, "--url", f"https://github.com/{repo}/issues/{n}")
    print(
        f"board: https://github.com/users/{owner}/projects/{number} (set the Sprint field per item from its milestone)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--board", action="store_true", help="also create a GitHub Projects board (needs the project scope)"
    )
    args = parser.parse_args()
    backlog = json.loads(BACKLOG.read_text(encoding="utf-8"))
    repo = repo_name()
    sync_labels(backlog["labels"])
    sync_milestones(repo, backlog["milestones"])
    numbers = sync_issues(repo, backlog["issues"])
    if args.board:
        sync_board(repo, backlog, numbers)
    return 0


if __name__ == "__main__":
    sys.exit(main())

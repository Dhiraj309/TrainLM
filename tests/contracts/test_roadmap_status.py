"""Keep duplicated roadmap progress views synchronized."""

from __future__ import annotations

from pathlib import Path
import re


ROADMAP = Path(__file__).parents[2] / "docs" / "ROADMAP.md"
STATUS = r"([ x~d])"


def _roadmap_text() -> str:
    return ROADMAP.read_text(encoding="utf-8")


def _expected_status(statuses: list[str]) -> str:
    assert statuses
    if all(status == "x" for status in statuses):
        return "x"
    if all(status == " " for status in statuses):
        return " "
    if all(status == "d" for status in statuses):
        return "d"
    return "~"


def _milestone_sections(text: str) -> dict[int, str]:
    headings = list(re.finditer(r"^## M(\d+)\b.*$", text, flags=re.MULTILINE))
    sections = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections[int(heading.group(1))] = text[heading.start():end]
    return sections


def _detailed_statuses(text: str) -> dict[int, str]:
    statuses = {}
    for milestone, section in _milestone_sections(text).items():
        match = re.search(
            rf"^\*\*Status:\*\* \[{STATUS}\]",
            section,
            flags=re.MULTILINE,
        )
        assert match, f"M{milestone} has no detailed status."
        statuses[milestone] = match.group(1)
    return statuses


def test_feature_checklists_match_detailed_milestone_statuses():
    text = _roadmap_text()

    for milestone, section in _milestone_sections(text).items():
        detailed = re.search(
            rf"^\*\*Status:\*\* \[{STATUS}\]",
            section,
            flags=re.MULTILINE,
        ).group(1)
        checklist = re.findall(
            rf"^- \[{STATUS}\]",
            section,
            flags=re.MULTILINE,
        )
        assert detailed == _expected_status(checklist), (
            f"M{milestone} detailed status does not match its checklist."
        )


def test_milestone_overview_matches_detailed_statuses():
    text = _roadmap_text()
    overview = text.split("## Milestone overview", 1)[1].split("## M0", 1)[0]
    overview_statuses = {
        int(milestone): status
        for status, milestone in re.findall(
            rf"^\| \[{STATUS}\] \| M(\d+) \|",
            overview,
            flags=re.MULTILINE,
        )
    }

    assert overview_statuses == _detailed_statuses(text)


def test_pr_overview_matches_covered_milestones():
    text = _roadmap_text()
    pr_table = text.split("### Dense-AR V1 PR sequence", 1)[1].split(
        "Future tracks:", 1
    )[0]
    milestone_statuses = _detailed_statuses(text)
    rows = re.findall(
        rf"^\| \[{STATUS}\] \| PR\d+ \|.*\| M(\d+)-M(\d+) \|",
        pr_table,
        flags=re.MULTILINE,
    )

    assert rows
    for pr_status, first, last in rows:
        covered = [
            milestone_statuses[milestone]
            for milestone in range(int(first), int(last) + 1)
        ]
        assert pr_status == _expected_status(covered)

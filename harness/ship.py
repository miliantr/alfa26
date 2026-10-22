"""What goes into the platform ZIP.

The platform copies only the root-level ``.py`` files and ``requirements.txt`` of the
ZIP into the task directory (organizers' upload instructions), so everything ships
flat: ``solution.py``, the ``agent_*.py`` modules and the provided dependency list.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The dependency list from the organizers' Drive folder (also in problem/); the agent
# only imports packages present in both variants, so either would install it.
REQUIREMENTS = "requirements_drive_folder.txt"


def code_files(root: Path = REPO) -> list[Path]:
    return [root / "solution.py", *sorted(root.glob("agent_*.py"))]


def zip_members(root: Path = REPO) -> dict[str, Path]:
    """ZIP name -> source file. Every name is at the archive root."""
    members = {p.name: p for p in code_files(root)}
    members["requirements.txt"] = root / REQUIREMENTS
    return members


def local_modules(root: Path = REPO) -> set[str]:
    return {p.stem for p in code_files(root)}

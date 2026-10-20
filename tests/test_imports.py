"""Every import in the shipped code must come from the stdlib or from packages present in
BOTH requirements variants (the platform container has nothing else)."""
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHIPPED = [REPO / "solution.py", *sorted((REPO / "agent").rglob("*.py"))]

# distribution name -> import names
DIST_TO_MODULES = {
    "scikit-learn": ["sklearn"], "pyyaml": ["yaml"], "python-dotenv": ["dotenv"],
    "category-encoders": ["category_encoders"], "feature-engine": ["feature_engine"],
    "imbalanced-learn": ["imblearn"], "autogluon.tabular": ["autogluon"],
    "flaml": ["flaml"], "lightautoml": ["lightautoml"], "langchain-openai": ["langchain_openai"],
}


def _dists(path: Path) -> set[str]:
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name = re.split(r"[\[=<>!~ ]", line, maxsplit=1)[0].strip().lower()
        out.add(name)
    return out


def allowed_modules() -> set[str]:
    common = _dists(REPO / "requirements.txt") & _dists(REPO / "requirements_drive_folder.txt")
    mods = set()
    for d in common:
        mods.update(DIST_TO_MODULES.get(d, [d.replace("-", "_")]))
    return mods


def imported_top_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def test_requirements_intersection_is_sane():
    allowed = allowed_modules()
    for m in ("numpy", "pandas", "sklearn", "lightgbm", "catboost", "xgboost", "openai", "psutil"):
        assert m in allowed
    for m in ("requests", "aiohttp", "langchain_openai", "mcp"):
        assert m not in allowed


def test_shipped_imports_are_allowed():
    allowed = allowed_modules() | set(sys.stdlib_module_names) | {"agent"}
    bad = {}
    for p in SHIPPED:
        extra = imported_top_modules(p) - allowed
        if extra:
            bad[str(p.relative_to(REPO))] = sorted(extra)
    assert not bad, f"imports outside requirements intersection: {bad}"


def test_no_network_libraries_besides_llm_client():
    """Only agent/llm_client.py may talk to the network (the organizers' LLM endpoint)."""
    net = {"socket", "urllib", "http", "httpx", "openai", "requests", "aiohttp", "ftplib", "smtplib"}
    offenders = {}
    for p in SHIPPED:
        if p.name in ("llm_client.py", "sandbox.py"):
            continue
        used = imported_top_modules(p) & net
        if used:
            offenders[str(p.relative_to(REPO))] = sorted(used)
    assert not offenders, offenders

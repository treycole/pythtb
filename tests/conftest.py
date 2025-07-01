# tests/conftest.py
import pytest
from pathlib import Path
from collections import defaultdict
from datetime import datetime

results = defaultdict(list)

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()

    if rep.when == "call":
        test_path = Path(item.fspath).relative_to(item.config.rootdir)
        test_name = item.name
        passed = rep.passed
        results[str(test_path.parent)].append((test_name, passed))

@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    output = ["# ✅ Test Status Checklist", ""]
    for folder in sorted(results):
        output.append(f"- **{folder}/**")
        for test_name, passed in sorted(results[folder]):
            check = "[x]" if passed else "[ ]"
            timestamp = f" _(at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})_" if passed else ""
            output.append(f"  - {check} `{test_name}`{timestamp}")
        output.append("")

    out_path = Path("PASSING.md")

    out_path.write_text("\n".join(output))
    print(f"\n✅ Wrote test results to {out_path}")
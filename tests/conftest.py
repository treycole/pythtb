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

def build_nested_tree(results):
    tree = {}
    for path, tests in results.items():
        parts = Path(path).parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
        node["_tests"] = sorted(tests)
    return tree

def format_tree(node, indent=0):
    lines = []
    for key, value in sorted(node.items()):
        if key == "_tests":
            for test_name, passed in value:
                check = "✅" if passed else "❌"
                lines.append("  " * indent + f"- {check} `{test_name}`")
        else:
            lines.append("  " * indent + f"- **{key}/**")
            lines.extend(format_tree(value, indent + 1))
    return lines

@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    output = [
        "# 📋 Test Status Report",
        "",
        f"Generated on **{datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}**",
        "",
        "---",
        ""
    ]

    tree = build_nested_tree(results)
    output.extend(format_tree(tree))


    # for folder in sorted(results):
    #     output.append(f"- **{folder}/**")
    #     for test_name, passed in sorted(results[folder]):
    #         check = "✅" if passed else "❌"
    #         output.append(f"    - {check} `{test_name}`")

    #     output.append("")

    out_path = Path(session.config.rootdir) / "tests" / "README.md"
    print(f"\n✅ Wrote nested test results to {out_path}")

    out_path.write_text("\n".join(output), encoding="utf-8")
    print(f"\n✅ Wrote test results to {out_path}")
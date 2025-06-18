import os
import json
from pathlib import Path

base_dir = Path(".")

def print_report():
    print(f"{'Group':<15} | {'Example':<15} | {'Status':<8} | {'Last Pass'}")
    print("-" * 65)

    for group in sorted(base_dir.iterdir()):
        if not group.is_dir():
            continue
        status_file = group / "status.json"
        if not status_file.exists():
            continue
        with open(status_file) as f:
            status = json.load(f)
        for example, entry in status.items():
            print(f"{group.name:<15} | {example:<15} | {entry.get('status', '???'):<8} | {entry.get('last_pass', 'n/a')}")

if __name__ == "__main__":
    print_report()
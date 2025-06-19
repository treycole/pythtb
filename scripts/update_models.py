#!/usr/bin/env python3

import os
import ast
from pathlib import Path

def generate_docstring(func_name, args):
    params_doc = "\n".join([f"    {arg} : TYPE\n        Description." for arg in args])
    return f'"""\n    {func_name} tight-binding model.\n\n' \
           f'    Parameters\n    ----------\n{params_doc}\n\n' \
           f'    Returns\n    -------\n    TBModel\n        An instance of the model.\n    """\n'

def enhance_model_docstrings(models_dir):
    model_files = sorted(f for f in os.listdir(models_dir)
                         if f.endswith(".py") and f != "__init__.py")
    for filename in model_files:
        file_path = models_dir / filename
        model_name = filename[:-3]

        with open(file_path, "r") as f:
            lines = f.readlines()

        tree = ast.parse("".join(lines))
        modified = False

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == model_name:
                if ast.get_docstring(node) is None:
                    args = [arg.arg for arg in node.args.args]
                    docstring = generate_docstring(model_name, args)
                    def_line = node.lineno - 1
                    indent = len(lines[def_line]) - len(lines[def_line].lstrip())
                    lines.insert(def_line + 1, " " * (indent + 4) + docstring + "\n")
                    modified = True
                break

        if modified:
            with open(file_path, "w") as f:
                f.writelines(lines)
            print(f"✅ Added docstring to: {filename}")

def regenerate_init_file(models_dir):
    model_files = sorted(f for f in os.listdir(models_dir)
                         if f.endswith(".py") and f != "__init__.py")
    imports = [f"from .{f[:-3]} import {f[:-3]}" for f in model_files]
    all_exports = [f'"{f[:-3]}"' for f in model_files]
    init_content = "\n".join(imports) + "\n\n__all__ = [\n    " + ",\n    ".join(all_exports) + "\n]\n"

    init_path = models_dir / "__init__.py"
    with open(init_path, "w") as f:
        f.write(init_content)
    print(f"🔁 Regenerated __init__.py with {len(model_files)} exports.")

def main():
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "pythtb" / "models"

    if not models_dir.exists():
        raise FileNotFoundError(f"Cannot find models directory: {models_dir}")

    enhance_model_docstrings(models_dir)
    regenerate_init_file(models_dir)

if __name__ == "__main__":
    main()
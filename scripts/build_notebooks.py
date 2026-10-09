"""Собирает .ipynb с выводами из исходников notebooks/src/*.py (формат «percent»: # %% и # %% [markdown]).

    python scripts/build_notebooks.py            # все ноутбуки
    python scripts/build_notebooks.py 02         # только 02_*

Зачем свой исполнитель: исходники ноутбуков лежат как обычный Python (удобно ревьюить в git),
а Jupyter для сборки не нужен — достаточно numpy/pandas/matplotlib. Если Jupyter есть, итоговые
.ipynb можно перезапустить как обычно.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "src"
OUT = ROOT / "notebooks"
FIG = ROOT / "reports" / "figures"


def parse_cells(text: str) -> list[tuple[str, str]]:
    cells, kind, buf = [], "code", []
    for line in text.splitlines():
        m = re.match(r"^# %%(.*)$", line)
        if m:
            if any(l.strip() for l in buf):
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
            buf = []
            continue
        if kind == "markdown":
            line = re.sub(r"^# ?", "", line)
        buf.append(line)
    if any(l.strip() for l in buf):
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def display_value(v) -> dict | None:
    if v is None:
        return None
    data = {"text/plain": repr(v)}
    if hasattr(v, "to_html"):
        data["text/html"] = v.to_html(float_format=lambda x: f"{x:,.4f}", border=0)
        data["text/plain"] = v.to_string() if hasattr(v, "to_string") else repr(v)
    return {"output_type": "execute_result", "data": data, "metadata": {}, "execution_count": None}


def run_code(code: str, ns: dict, fig_prefix: str = "", fig_counter: list | None = None) -> list[dict]:
    outputs, stdout = [], io.StringIO()
    tree = ast.parse(code)
    last_expr = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    result = None
    with contextlib.redirect_stdout(stdout):
        exec(compile(tree, "<cell>", "exec"), ns)
        if last_expr is not None:
            result = eval(compile(ast.Expression(last_expr.value), "<cell>", "eval"), ns)
    if stdout.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue()})
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        if fig_counter is not None:  # копия графика для README
            fig_counter[0] += 1
            FIG.mkdir(parents=True, exist_ok=True)
            (FIG / f"{fig_prefix}_fig{fig_counter[0]}.png").write_bytes(buf.getvalue())
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": "<Figure>"}})
        plt.close(fig)
    # служебные объекты matplotlib (Legend, Axes, Text, списки линий) не показываем
    is_mpl = type(result).__module__.startswith("matplotlib") or (
        isinstance(result, (list, tuple)) and result and type(result[0]).__module__.startswith("matplotlib"))
    disp = None if is_mpl else display_value(result)
    if disp:
        outputs.append(disp)
    return outputs


def build(src: Path) -> Path:
    cells = parse_cells(src.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    os.chdir(ROOT / "notebooks")  # как в Jupyter: ноутбук сам добавляет пути к src (без помощи сборщика)
    nb_cells, n = [], 0
    fig_counter = [0]
    t0 = time.time()
    for kind, text in cells:
        lines = [l + "\n" for l in text.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "id": f"{src.stem[:2]}-{len(nb_cells):02d}", "metadata": {}, "source": lines})
            continue
        n += 1
        try:
            outs = run_code(text, ns, src.stem[:2], fig_counter)
        except Exception:
            print(f"Ошибка в {src.name}, ячейка {n}:\n{text}\n"); traceback.print_exc(); raise
        for o in outs:
            if o["output_type"] == "execute_result":
                o["execution_count"] = n
        nb_cells.append({"cell_type": "code", "id": f"{src.stem[:2]}-{len(nb_cells):02d}", "metadata": {},
                         "execution_count": n, "source": lines, "outputs": outs})
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    dst = OUT / (src.stem + ".ipynb")
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dst.name}: {n} ячеек кода, {time.time() - t0:.0f} с")
    return dst


if __name__ == "__main__":
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    for f in sorted(p for p in SRC.glob(f"{pick}*.py") if not p.name.startswith("_")):
        build(f)

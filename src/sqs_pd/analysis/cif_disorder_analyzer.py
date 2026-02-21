"""CIF 无序分析（SD / PD / SPD）

本模块实现面向 CIF 文件的无序识别：
- 以 `_atom_site_label` 聚合位点
- 依据本项目规范将每个位点分类为 Ordered / SD / PD / SPD
- 汇总得到 CIF 级别的无序类型集合（可多标签）

设计目标：
- 不依赖 pymatgen 的 Structure 合并语义，直接从 CIF 的 `_atom_site_*` 表读取
- 尽量兼容常见的 CIF loop 写法（字段顺序可变；数值可带引号；行可分割）

注意：这是“判定”模块，不负责后续 SQS 优化。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
import shlex

from ..foundation.constants import OCCUPANCY_TOLERANCE
from .analysis_utils import (
    DISORDER_ORDER,
    classify_disorder_types_from_sites,
    classify_site_sd_pd_spd,
)


def _tokenize_cif_line(line: str) -> List[str]:
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""  # CIF 中 # 不一定表示注释
    return list(lexer)


def _normalize_type_symbol(type_symbol: str) -> str:
    """将 CIF 的 _atom_site_type_symbol 规范化为元素符号。

    示例：
    - "W5.677+" -> "W"
    - "Cs+" -> "Cs"
    - "O2-" -> "O"
    """
    s = type_symbol.strip()
    if not s:
        return s

    # 常见形式：元素符号开头，后面跟数字/电荷/括号等
    m = re.match(r"^([A-Za-z]{1,2})", s)
    if not m:
        return s

    sym = m.group(1)
    return sym[0].upper() + sym[1:].lower()


def _parse_float(value: str, default: Optional[float] = None) -> Optional[float]:
    v = value.strip()
    if v in {".", "?", ""}:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _read_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def _iter_cif_loops(lines: List[str]) -> Iterable[Tuple[List[str], List[List[str]]]]:
    """遍历 CIF 中的 loop_ 表，产出 (headers, rows_tokens)。"""

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        if line.lower().startswith("loop_"):
            i += 1
            headers: List[str] = []

            # 读 headers
            while i < n:
                l = lines[i].strip()
                if not l or l.startswith("#"):
                    i += 1
                    continue
                if l.startswith("_"):
                    headers.append(_tokenize_cif_line(l)[0])
                    i += 1
                    continue
                break

            if not headers:
                continue

            # 读 rows：按 header 数量聚合 token
            rows: List[List[str]] = []
            buf: List[str] = []

            def flush_if_ready() -> None:
                nonlocal buf
                while len(buf) >= len(headers):
                    rows.append(buf[: len(headers)])
                    buf = buf[len(headers) :]

            while i < n:
                l_raw = lines[i]
                l = l_raw.strip()

                if not l or l.startswith("#"):
                    i += 1
                    continue

                lower = l.lower()
                # loop 结束条件：遇到新的 loop_ / data_ / 顶层 item，且当前没有未完成的行缓冲
                if not buf and (
                    lower.startswith("loop_")
                    or lower.startswith("data_")
                    or l.startswith("_")
                ):
                    break

                buf.extend(_tokenize_cif_line(l_raw))
                flush_if_ready()
                i += 1

            yield headers, rows
            continue

        i += 1


def read_atom_site_table(cif_file: str | Path) -> List[Dict[str, str]]:
    """读取 CIF 的 `_atom_site_*` 表。

    返回：每一行对应一个 dict（header -> value）。
    """
    path = Path(cif_file)
    lines = _read_lines(path)

    required_any = {
        "_atom_site_label",
    }

    type_keys = ("_atom_site_type_symbol", "_atom_site_symbol")
    occ_key = "_atom_site_occupancy"

    for headers, rows in _iter_cif_loops(lines):
        header_set = set(h.lower() for h in headers)
        if not required_any.issubset(header_set):
            continue

        # 找 type key
        type_key: Optional[str] = None
        for k in type_keys:
            if k in header_set:
                type_key = k
                break

        if type_key is None:
            continue

        # 构建 row dicts
        out: List[Dict[str, str]] = []
        for r in rows:
            d = {headers[j].lower(): r[j] for j in range(len(headers))}
            # 规范化关键字段名到 lower
            if "_atom_site_label" not in d:
                continue
            if type_key not in d:
                continue
            # occupancy 缺省时不填，后续当 1.0
            out.append(d)

        if out:
            return out

    return []


def analyze_cif_disorder(
    cif_file: str | Path,
    *,
    group_by: str = "coords",
    tol: float = OCCUPANCY_TOLERANCE,
    coord_tol: float = 1e-4,
) -> Dict[str, Any]:
    """分析 CIF 的无序类型（不进行任何优化）。

    Args:
        cif_file: CIF 文件路径
        group_by: 位点聚合方式，"coords" (默认，按坐标) 或 "label" (兼容旧版)
        tol: 判断 W≈1 的容差
        coord_tol: 坐标相等判断容差（仅用于 group_by="coords"）

    Returns:
        dict，包含：
        - success: bool
        - disorder_types: List[str]  # 例如 ["PD", "SD"]
        - disorder_type: str         # 兼容展示用，例如 "PD+SD" 或 "ordered"
        - site_results: List[dict]   # 每个位点的分类与占据信息
        - warnings: List[str]
        - error: Optional[str]
    """

    if group_by not in ("coords", "label"):
        return {
            "success": False,
            "error": f"Unsupported group_by={group_by!r} (only 'coords' or 'label' supported)",
            "warnings": [],
            "site_results": [],
            "disorder_types": [],
            "disorder_type": "error",
        }

    rows = read_atom_site_table(cif_file)
    if not rows:
        return {
            "success": False,
            "error": "No _atom_site loop with label/type found in CIF",
            "warnings": [],
            "site_results": [],
            "disorder_types": [],
            "disorder_type": "error",
        }

    warnings: List[str] = []

    # 选择 type key（read_atom_site_table 已确保存在）
    sample_keys = set(rows[0].keys())
    type_key = (
        "_atom_site_type_symbol"
        if "_atom_site_type_symbol" in sample_keys
        else "_atom_site_symbol"
    )
    occ_key = "_atom_site_occupancy"

    # Group by coords or label
    if group_by == "coords":
        # Extract coordinates for grouping
        by_site: Dict[Tuple[float, float, float], Dict[str, float]] = {}
        coord_keys = ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")

        for r in rows:
            label = str(r.get("_atom_site_label", "")).strip()

            # Parse coordinates
            try:
                x = _parse_float(str(r.get(coord_keys[0], "0.0")), default=0.0)
                y = _parse_float(str(r.get(coord_keys[1], "0.0")), default=0.0)
                z = _parse_float(str(r.get(coord_keys[2], "0.0")), default=0.0)
                if x is None or y is None or z is None:
                    warnings.append(f"Site {label}: invalid coordinates, skipped")
                    continue
                coords = (float(x), float(y), float(z))
            except Exception:
                warnings.append(f"Site {label}: failed to parse coordinates, skipped")
                continue

            raw_type = str(r.get(type_key, "")).strip()
            if not raw_type:
                warnings.append(f"Site {label}: missing {type_key}")
                continue

            elem = _normalize_type_symbol(raw_type)
            occ_raw = r.get(occ_key, "1.0")
            occ = _parse_float(str(occ_raw), default=1.0)
            if occ is None:
                warnings.append(
                    f"Site {label}: invalid occupancy {occ_raw!r}, defaulted to 1.0"
                )
                occ = 1.0

            # Find existing site with same coords (within tolerance)
            site_key = None
            for existing_coords in by_site.keys():
                if all(
                    abs(coords[i] - existing_coords[i]) < coord_tol for i in range(3)
                ):
                    site_key = existing_coords
                    break

            if site_key is None:
                site_key = coords
                by_site[site_key] = {}

            by_site[site_key][elem] = by_site[site_key].get(elem, 0.0) + float(occ)
    else:
        # Original label-based grouping (legacy)
        by_label: Dict[str, Dict[str, float]] = {}

        for r in rows:
            label = str(r.get("_atom_site_label", "")).strip()
            if not label:
                continue

            raw_type = str(r.get(type_key, "")).strip()
            if not raw_type:
                warnings.append(f"Site {label}: missing {type_key}")
                continue

            elem = _normalize_type_symbol(raw_type)
            occ_raw = r.get(occ_key, "1.0")
            occ = _parse_float(str(occ_raw), default=1.0)
            if occ is None:
                warnings.append(
                    f"Site {label}: invalid occupancy {occ_raw!r}, defaulted to 1.0"
                )
                occ = 1.0

            if label not in by_label:
                by_label[label] = {}

            by_label[label][elem] = by_label[label].get(elem, 0.0) + float(occ)

        # Convert to unified format
        by_site = {k: v for k, v in by_label.items()}

    site_results: List[Dict[str, Any]] = []
    disorder_types_set: set[str] = set()

    for site_key in sorted(by_site.keys(), key=str):
        species = by_site[site_key]
        # Generate label for display
        if group_by == "coords":
            label = f"site_{len(site_results)}"
        else:
            label = str(site_key)
        # 过滤 0 或负值
        species = {k: v for k, v in species.items() if v > 0.0}

        num_species = len(species)
        total_occupancy = sum(species.values())
        cls = classify_site_sd_pd_spd(num_species, total_occupancy, tol=tol)

        if cls in DISORDER_ORDER:
            disorder_types_set.add(cls)

        if cls == "Invalid":
            warnings.append(
                f"Site {label}: total occupancy {total_occupancy:.6g} > 1 (check CIF data)"
            )

        site_results.append(
            {
                "label": label,
                "species": dict(sorted(species.items())),
                "num_species": num_species,
                "total_occupancy": total_occupancy,
                "site_type": cls,
            }
        )

    disorder_types = classify_disorder_types_from_sites(list(disorder_types_set))

    return {
        "success": True,
        "disorder_types": disorder_types,
        "num_sites": len(site_results),
        "num_disordered_sites": sum(
            1 for s in site_results if s["site_type"] in DISORDER_ORDER
        ),
        "site_results": site_results,
        "warnings": warnings,
        "error": None,
    }

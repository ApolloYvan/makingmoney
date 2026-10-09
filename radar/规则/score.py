#!/usr/bin/env python3
"""机会雷达第一阶段的打分脚本。

读 规则/打分规则.toml、规则/来源清单.csv 和 数据/ 下的四个 CSV，
检查数据、算分，生成 周报/YYYY-Www.md。

分数不存进任何文件，每次都按当前规则从头算，所以规则一改，历史方向会全部按新规则重算。
大模型不参与这里的任何计算。

只用标准库，Python 3.9 以上都能跑（macOS 自带的就是 3.9）。

用法（在仓库根目录）：
    python3 radar/规则/score.py                    以今天为截止日生成周报
    python3 radar/规则/score.py --date 2026-10-18  指定截止日
    python3 radar/规则/score.py --since 2026-09-18 指定周报统计区间的起点（默认是截止日前 7 天）
    python3 radar/规则/score.py --check            只检查数据，有问题时返回非 0
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    import tomllib  # Python 3.11 起自带
except ModuleNotFoundError:  # 更早的版本用下面的简易解析器
    tomllib = None  # type: ignore[assignment]

RULES_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = RULES_DIR.parent

POSITIVE_RESULTS = {"确认痛点", "有人付定金"}
EXCERPT_LEN = 60


# ---------------------------------------------------------------- 读文件

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
        return rows


def parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_number(s: str) -> float | None:
    s = s.replace(",", "").replace("，", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def split_multi(s: str) -> list[str]:
    for sep in ("、", "/", "，", ","):
        s = s.replace(sep, "|")
    return [p.strip() for p in s.split("|") if p.strip()]


def tier(value: float, bands: list[list[float]]) -> int:
    """bands 形如 [[至少多少, 得分], ...]，从高往低匹配。"""
    for floor, points in sorted(bands, key=lambda b: b[0], reverse=True):
        if value >= floor:
            return int(points)
    return int(min(b[1] for b in bands))


def fmt_num(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


# ---------------------------------------------------------------- 数据结构

@dataclass
class Signal:
    id: str
    day: date
    source: str
    line: str
    stage: str
    link: str
    excerpt: str
    direction: str
    pay: str
    amount: float | None


@dataclass
class Scored:
    row: dict[str, str]
    kind: str
    signals: list[Signal] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)   # 计算过程，给人看
    missing: list[str] = field(default_factory=list)
    total: float | None = None
    passed: bool = False
    gate_note: str = ""
    filtered: str = ""                                # 非空表示被过滤，写原因
    recent: int = 0

    @property
    def id(self) -> str:
        return self.row.get("编号", "")

    @property
    def name(self) -> str:
        return self.row.get("名称", "")


@dataclass
class Result:
    window_start: date
    window_end: date
    signals: list[Signal]
    ignored: list[str]
    problems: list[str]
    long: list[Scored]
    fast: list[Scored]
    policies: list[dict[str, str]]
    policy_notes: list[str]
    validations: list[dict[str, str]]
    source_names: list[str]
    phase1_sources: list[str]


# ---------------------------------------------------------------- 核心逻辑

def load_rules(root: Path) -> dict:
    path = root / "规则" / "打分规则.toml"
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f)
    return parse_simple_toml(path.read_text(encoding="utf-8"))


def parse_simple_toml(text: str) -> dict:
    """给 Python 3.9、3.10 用的简易 TOML 解析器。

    只支持打分规则用到的写法：[表头]（可带引号、可用点分层）、键 = 值、
    字符串、整数、小数、true/false、可嵌套且可跨行的数组、# 注释。
    """
    root: dict = {}
    table = root
    pending = ""
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw).strip()
        if pending:
            line = pending + " " + line
            pending = ""
        if not line:
            continue
        if line.startswith("[") and "=" not in line:
            table = root
            for part in _split_key(line[1:-1].strip(), lineno):
                table = table.setdefault(part, {})
            continue
        if "=" not in line:
            raise ValueError(f"打分规则.toml 第 {lineno} 行看不懂：{raw.strip()}")
        key_text, value_text = _split_assignment(line, lineno)
        if value_text.count("[") > value_text.count("]"):
            pending = line  # 数组跨行，接着读下一行
            continue
        value, end = _parse_value(value_text, 0, lineno)
        if value_text[end:].strip():
            raise ValueError(f"打分规则.toml 第 {lineno} 行值的后面有多余内容")
        keys = _split_key(key_text, lineno)
        target = table
        for part in keys[:-1]:
            target = target.setdefault(part, {})
        target[keys[-1]] = value
    if pending:
        raise ValueError("打分规则.toml 结尾有没闭合的数组")
    return root


def _strip_comment(line: str) -> str:
    in_str = False
    for i, ch in enumerate(line):
        if ch == '"' and (i == 0 or line[i - 1] != "\\"):
            in_str = not in_str
        elif ch == "#" and not in_str:
            return line[:i]
    return line


def _split_assignment(line: str, lineno: int) -> tuple[str, str]:
    in_str = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_str = not in_str
        elif ch == "=" and not in_str:
            return line[:i].strip(), line[i + 1:].strip()
    raise ValueError(f"打分规则.toml 第 {lineno} 行缺少等号")


def _split_key(text: str, lineno: int) -> list[str]:
    parts, buf, in_str, quoted = [], "", False, False
    for ch in text:
        if ch == '"':
            in_str = not in_str
            quoted = True
        elif ch == "." and not in_str:
            parts.append(buf if quoted else buf.strip())
            buf, quoted = "", False
        elif in_str or not ch.isspace():
            buf += ch
    parts.append(buf if quoted else buf.strip())
    if in_str or any(p == "" for p in parts):
        raise ValueError(f"打分规则.toml 第 {lineno} 行的键写法不对：{text}")
    return parts


def _parse_value(s: str, i: int, lineno: int):
    while i < len(s) and s[i] in " ,":
        i += 1
    if i >= len(s):
        raise ValueError(f"打分规则.toml 第 {lineno} 行缺少值")
    ch = s[i]
    if ch == '"':
        j, out = i + 1, ""
        while j < len(s) and s[j] != '"':
            if s[j] == "\\" and j + 1 < len(s):
                j += 1
            out += s[j]
            j += 1
        if j >= len(s):
            raise ValueError(f"打分规则.toml 第 {lineno} 行字符串没有闭合")
        return out, j + 1
    if ch == "[":
        items, j = [], i + 1
        while True:
            while j < len(s) and s[j] in " ,":
                j += 1
            if j >= len(s):
                raise ValueError(f"打分规则.toml 第 {lineno} 行数组没有闭合")
            if s[j] == "]":
                return items, j + 1
            item, j = _parse_value(s, j, lineno)
            items.append(item)
    j = i
    while j < len(s) and s[j] not in ",]":
        j += 1
    token = s[i:j].strip()
    if token in ("true", "false"):
        return token == "true", j
    try:
        return (float(token) if any(c in token for c in ".eE") else int(token)), j
    except ValueError:
        raise ValueError(f"打分规则.toml 第 {lineno} 行的值看不懂：{token}") from None


def load_signals(root: Path, rules: dict, sources: set[str], direction_ids: set[str],
                 until: date) -> tuple[list[Signal], list[str]]:
    lines = rules["枚举"]["所属线"]
    stages = rules["阶段"]
    pay_opts = rules["快钱"]["付费证据"]
    out: list[Signal] = []
    ignored: list[str] = []
    seen_ids: set[str] = set()
    seen_excerpts: set[tuple[str, str]] = set()

    for i, r in enumerate(read_csv(root / "数据" / "信号.csv"), start=2):
        sid = r.get("编号") or f"第{i}行"

        def bad(why: str) -> None:
            ignored.append(f"{sid}：{why}")

        d = parse_date(r.get("日期", ""))
        if d is None:
            bad(f"日期格式不对（{r.get('日期', '')!r}），应为 YYYY-MM-DD")
            continue
        if d > until:
            continue
        if not r.get("编号"):
            bad("没有编号")
            continue
        if sid in seen_ids:
            bad("编号重复")
            continue
        if not r.get("链接") or not r.get("原文摘录"):
            bad("缺链接或原文摘录")
            continue
        if r.get("来源") not in sources:
            bad(f"来源 {r.get('来源')!r} 不在来源清单里")
            continue
        line = r.get("所属线", "")
        if line not in lines:
            bad(f"所属线 {line!r} 无效")
            continue
        if r.get("阶段") not in stages.get(line, []):
            bad(f"阶段 {r.get('阶段')!r} 不是 {line} 线的有效阶段")
            continue
        pay = r.get("付费证据") or "无"
        if pay not in pay_opts:
            bad(f"付费证据 {pay!r} 无效")
            continue
        amount_raw = r.get("金额元", "")
        amount = parse_number(amount_raw)
        if amount_raw and amount is None:
            bad(f"金额元 {amount_raw!r} 不是数字")
            continue
        direction = r.get("方向编号", "")
        if direction and direction not in direction_ids:
            bad(f"方向编号 {direction!r} 在方向.csv 里不存在")
            continue
        key = (r["链接"], r["原文摘录"])
        if key in seen_excerpts:
            bad("同一链接、同一段摘录已经记过，重复")
            continue
        seen_ids.add(sid)
        seen_excerpts.add(key)
        out.append(Signal(sid, d, r["来源"], line, r["阶段"], r["链接"], r["原文摘录"],
                          direction, pay, amount))
    return out, ignored


def crossing(rules: dict, lines: set[str]) -> float:
    names = {1: "一条线", 2: "两条线", 3: "三条线"}
    return float(rules["交汇"].get(names.get(len(lines), "一条线"), 1.0)) if lines else 1.0


def score_direction(row: dict[str, str], sigs: list[Signal], rules: dict, until: date) -> Scored:
    g = rules["通用"]
    kind = row.get("类型", "")
    s = Scored(row=row, kind=kind, signals=sorted(sigs, key=lambda x: x.day, reverse=True))

    cross_from = until - timedelta(days=int(g["交汇回看天数"]) - 1)
    recent_lines = {x.line for x in sigs if x.day >= cross_from}
    s.lines = [ln for ln in rules["枚举"]["所属线"] if ln in recent_lines]
    mult = crossing(rules, recent_lines)
    diff_from = until - timedelta(days=int(g["扩散回看天数"]) - 1)
    s.recent = sum(1 for x in sigs if x.day >= diff_from)

    hit = row.get("命中不碰", "")
    if hit and hit != "否":
        s.filtered = f"命中不碰的事：{hit}"

    access_opt = row.get("准入成本", "")
    access = rules["准入成本"].get(access_opt)

    if kind == "快钱":
        q = rules["快钱"]
        diffusion = tier(s.recent, q["扩散速度"]["档位"])
        pay = max((q["付费证据"][x.pay] for x in sigs), default=q["付费证据"]["无"])
        # 窗口剩余由信号决定：方向下没有"大厂进场"的信号，就算无大厂进场；
        # 有的话，才需要在方向表里选是"同赛道融资"还是"云厂商一键方案"。
        entered = [x for x in sigs if x.stage == "大厂进场"]
        if entered:
            window_opt = row.get("窗口剩余", "")
            window = q["窗口剩余"].get(window_opt)
            if window is None or window_opt == "无大厂进场":
                window = None
                ids = "、".join(x.id for x in entered[:3])
                s.missing.append(f"窗口剩余（已有大厂进场信号 {ids}，需要选同赛道融资或云厂商一键方案）")
        else:
            window_opt = "无大厂进场"
            window = q["窗口剩余"][window_opt]
        s.parts = [
            f"扩散速度 {diffusion}（近 {g['扩散回看天数']} 天新增 {s.recent} 条）",
            f"付费证据 {pay}",
            f"窗口剩余 {window if window is not None else '?'}（{window_opt or '未填'}）",
        ]
        if window is not None:
            s.total = diffusion * pay * window * mult
        stage_count = len({x.stage for x in sigs if x.line == "技术普及"})
        pay_count = sum(1 for x in sigs if x.pay != "无")
        gate = q["门槛"]
        s.passed = stage_count >= gate["最少阶段数"] and pay_count >= gate["最少付费信号"]
        s.gate_note = f"覆盖 {stage_count} 个阶段，付费信号 {pay_count} 条"
    elif kind == "长钱":
        q = rules["长钱"]
        amount = parse_number(row.get("每月金额元", ""))
        items: list[tuple[str, float | None, str]] = []
        if amount is None:
            items.append(("金额", None, "未填"))
        else:
            items.append(("金额", tier(amount, q["金额"]["档位"]), f"每月约 {fmt_num(amount)} 元"))
        for col in ("频次", "持续性", "技术门槛", "决策链"):
            opt = row.get(col, "")
            items.append((col, q[col].get(opt), opt or "未填"))
        for col, val, why in items:
            if val is None:
                s.missing.append(f"{col}（{why}）")
            s.parts.append(f"{col} {val if val is not None else '?'}（{why}）")
        if access is None:
            s.missing.append(f"准入成本（{access_opt or '未填'}）")
        s.parts.append(f"准入 {access if access is not None else '?'}（{access_opt or '未填'}）")
        if not s.missing:
            product = 1.0
            for _, val, _ in items:
                product *= float(val)  # type: ignore[arg-type]
            s.total = product * float(access) * mult  # type: ignore[arg-type]
        offline = sum(1 for x in sigs if x.source == g["线下来源"])
        with_amount = sum(1 for x in sigs if x.source != g["线下来源"] and x.amount is not None)
        gate = q["门槛"]
        s.passed = offline >= gate["最少线下信号"] or with_amount >= gate["或最少带金额公开信号"]
        s.gate_note = f"线下信号 {offline} 条，带金额的公开信号 {with_amount} 条"
    else:
        s.missing.append(f"类型（{kind or '未填'}）")

    if access == 0 and not s.filtered:
        s.filtered = "准入成本为“个人拿不到”"
    s.parts.append(f"交汇 {fmt_num(mult)}（{'、'.join(s.lines) or '近期无信号'}）")
    return s


def check_policies(rows: list[dict[str, str]], rules: dict) -> list[str]:
    stages = rules["枚举"]["政策阶段"]
    types = rules["枚举"]["政策类型"]
    notes: list[str] = []
    by_id: dict[str, list[tuple[date, int, int]]] = {}
    for i, r in enumerate(rows, start=2):
        pid = r.get("编号") or f"第{i}行"
        if r.get("阶段") not in stages:
            notes.append(f"政策 {pid}：阶段 {r.get('阶段')!r} 无效")
            continue
        for t in split_multi(r.get("类型", "")):
            if t not in types:
                notes.append(f"政策 {pid}：类型 {t!r} 无效")
        if r.get("施行日期") and parse_date(r["施行日期"]) is None:
            notes.append(f"政策 {pid}：施行日期格式不对")
        upd = parse_date(r.get("更新日期", ""))
        if upd is None:
            notes.append(f"政策 {pid}：更新日期格式不对")
            continue
        by_id.setdefault(pid, []).append((upd, i, stages.index(r["阶段"])))
    for pid, hist in by_id.items():
        hist.sort()
        for (_, _, a), (_, row_no, b) in zip(hist, hist[1:]):
            if b < a:
                notes.append(f"政策 {pid}：第 {row_no} 行的阶段比之前的记录倒退了，阶段只能往前走")
    return notes


def run(root: Path, until: date, since: date | None = None) -> Result:
    rules = load_rules(root)
    g = rules["通用"]
    source_rows = read_csv(root / "规则" / "来源清单.csv")
    source_names = [r["名称"] for r in source_rows if r.get("名称")]
    phase1 = [r["名称"] for r in source_rows if r.get("一期") == "是"]

    problems: list[str] = []
    directions = read_csv(root / "数据" / "方向.csv")
    direction_ids: set[str] = set()
    for i, d in enumerate(directions, start=2):
        did = d.get("编号", "")
        if not did:
            problems.append(f"方向.csv 第 {i} 行没有编号")
        elif did in direction_ids:
            problems.append(f"方向 {did}：编号重复")
        direction_ids.add(did)
        if d.get("状态") and d["状态"] not in rules["枚举"]["方向状态"]:
            problems.append(f"方向 {did}：状态 {d['状态']!r} 无效")

    signals, ignored = load_signals(root, rules, set(source_names), direction_ids, until)

    by_dir: dict[str, list[Signal]] = {}
    for x in signals:
        if x.direction:
            by_dir.setdefault(x.direction, []).append(x)

    long_: list[Scored] = []
    fast: list[Scored] = []
    seen: set[str] = set()
    for d in directions:
        did = d.get("编号", "")
        if not did or did in seen:
            continue
        seen.add(did)
        sc = score_direction(d, by_dir.get(did, []), rules, until)
        if sc.kind == "长钱":
            long_.append(sc)
        elif sc.kind == "快钱":
            fast.append(sc)
        else:
            problems.append(f"方向 {did}：类型 {sc.kind!r} 无效，应为 快钱 或 长钱")

    policies = read_csv(root / "数据" / "政策.csv")
    policy_notes = check_policies(policies, rules)

    validations = read_csv(root / "数据" / "验证.csv")
    for i, v in enumerate(validations, start=2):
        if v.get("方向编号") not in direction_ids:
            problems.append(f"验证.csv 第 {i} 行：方向编号 {v.get('方向编号')!r} 不存在")
        if v.get("方式") not in rules["枚举"]["验证方式"]:
            problems.append(f"验证.csv 第 {i} 行：方式 {v.get('方式')!r} 无效")
        if v.get("结果") not in rules["枚举"]["验证结果"]:
            problems.append(f"验证.csv 第 {i} 行：结果 {v.get('结果')!r} 无效")
        if parse_date(v.get("日期", "")) is None:
            problems.append(f"验证.csv 第 {i} 行：日期格式不对")

    start = since or until - timedelta(days=int(g["扩散回看天数"]) - 1)
    return Result(start, until, signals, ignored, problems, long_, fast, policies,
                  policy_notes, validations, source_names, phase1)


# ---------------------------------------------------------------- 周报

def rank(items: list[Scored]) -> list[Scored]:
    return sorted(items, key=lambda s: (-(s.total or 0), -s.recent, s.id))


def is_active(s: Scored) -> bool:
    return s.row.get("状态", "") in ("", "观察中", "候选")


def card(n: int, s: Scored) -> list[str]:
    lines = [f"### {n}. {s.id} {s.name}（{fmt_num(s.total or 0)} 分）", ""]
    lines.append("计算：" + " × ".join(s.parts) + f" = {fmt_num(s.total or 0)}")
    lines.append("")
    lines.append(f"门槛：{s.gate_note}。")
    for col in ("金额依据", "频次依据", "持续性依据", "技术门槛依据", "决策链依据", "窗口依据",
                "准入成本依据"):
        if s.row.get(col):
            lines.append(f"- {col}：{s.row[col]}")
    lines.append("")
    lines.append("最近的证据：")
    for x in s.signals[:6]:
        ex = x.excerpt if len(x.excerpt) <= EXCERPT_LEN else x.excerpt[:EXCERPT_LEN] + "…"
        lines.append(f"- {x.day} · {x.source} · {x.line}/{x.stage} · [{ex}]({x.link})")
    if s.row.get("备注"):
        lines += ["", f"备注：{s.row['备注']}"]
    lines.append("")
    return lines


def rules_dirty(root: Path) -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", str(root / "规则")],
                             cwd=root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def render(res: Result, rules: dict, root: Path) -> str:
    g = rules["通用"]
    y, w, _ = res.window_end.isocalendar()
    new = [x for x in res.signals if x.day >= res.window_start]
    out = [f"# 机会雷达周报 {y}-W{w:02d}", ""]
    out.append(f"统计区间 {res.window_start} 至 {res.window_end}。有效信号累计 {len(res.signals)} 条，"
               f"本期新增 {len(new)} 条；被忽略 {len(res.ignored)} 条；"
               f"方向 {len(res.long) + len(res.fast)} 个。由 score.py 生成，不要手改分数。")
    out.append("")
    if rules_dirty(root):
        out += ["> 注意：规则/ 目录有未提交的改动。确认是我自己改的再用这期结果。", ""]

    sections = [("长钱候选", res.long, int(g["长钱最多推送"])),
                ("快钱候选", res.fast, int(g["快钱最多推送"]))]
    watch: list[Scored] = []
    for title, items, limit in sections:
        out += [f"## {title}", ""]
        ok = [s for s in rank(items) if is_active(s) and not s.filtered and s.total is not None
              and s.passed]
        if not ok:
            out += ["本期没有过门槛的方向。", ""]
        for n, s in enumerate(ok[:limit], start=1):
            out += card(n, s)
        watch += ok[limit:]
        watch += [s for s in items if is_active(s) and not s.filtered and s.total is not None
                  and not s.passed]

    out += ["## 观察列表", ""]
    watch = rank(watch)[: int(g["观察列表展示"])]
    if watch:
        for s in watch:
            note = "过了门槛但名额已满" if s.passed else f"未过门槛：{s.gate_note}"
            out.append(f"- {s.id} {s.name}（{s.kind}，{fmt_num(s.total or 0)} 分）：{note}")
    else:
        out.append("暂无。")
    out.append("")

    pending = [s for s in res.long + res.fast if is_active(s) and not s.filtered and s.missing]
    out += ["## 缺分项、暂时算不了分的方向", ""]
    if pending:
        for s in pending:
            out.append(f"- {s.id} {s.name}：缺 {'、'.join(s.missing)}")
    else:
        out.append("暂无。")
    out.append("")

    out += ["## 验证中和已通过", ""]
    tracked = [s for s in res.long + res.fast if s.row.get("状态") in ("验证中", "通过")]
    if tracked:
        for s in tracked:
            score = fmt_num(s.total) if s.total is not None else "算不了"
            out.append(f"- {s.id} {s.name}（{s.row['状态']}，{score} 分）")
    else:
        out.append("暂无。")
    out.append("")

    filtered = [s for s in res.long + res.fast if s.filtered]
    if filtered:
        out += ["## 被过滤的方向", ""]
        out += [f"- {s.id} {s.name}：{s.filtered}" for s in filtered]
        out.append("")

    out += ["## 政策动态", ""]
    soon_until = res.window_end + timedelta(days=int(g["政策临近施行天数"]))
    stages = rules["枚举"]["政策阶段"]
    updated, soon = [], []
    for p in res.policies:
        upd = parse_date(p.get("更新日期", ""))
        eff = parse_date(p.get("施行日期", ""))
        label = f"{p.get('编号', '')} {p.get('标题', '')}（{p.get('发文机关', '')}，{p.get('阶段', '')}，{p.get('类型', '')}）"
        if p.get("链接"):
            label = f"[{label}]({p['链接']})"
        if upd and res.window_start <= upd <= res.window_end:
            updated.append(f"- {label}")
        if (eff and res.window_end < eff <= soon_until and p.get("阶段") in stages
                and stages.index(p["阶段"]) < stages.index("施行")):
            soon.append(f"- {eff} 施行：{label}")
    out.append("本期新增或更新：")
    out += updated or ["- 无"]
    out.append("")
    out.append(f"未来 {g['政策临近施行天数']} 天内施行：")
    out += soon or ["- 无"]
    out.append("")

    out += ["## 本期验证记录", ""]
    recent_v = [v for v in res.validations
                if (d := parse_date(v.get("日期", ""))) and res.window_start <= d <= res.window_end]
    if recent_v:
        out += [f"- {v['日期']} {v.get('方向编号', '')} {v.get('方式', '')}：{v.get('结果', '')}"
                f"（{v.get('证据', '')}）" for v in recent_v]
    else:
        out.append("本期没有验证记录。每周至少 2 次访谈。")
    out.append("")

    out += ["## 来源统计", ""]
    positive_dirs = {v.get("方向编号") for v in res.validations if v.get("结果") in POSITIVE_RESULTS}
    out.append("| 来源 | 本期新增 | 累计 | 其中属于验证通过的方向 |")
    out.append("|---|---|---|---|")
    for name in res.source_names:
        total = [x for x in res.signals if x.source == name]
        if not total and name not in res.phase1_sources:
            continue
        fresh = sum(1 for x in total if x.day >= res.window_start)
        hit = sum(1 for x in total if x.direction in positive_dirs)
        out.append(f"| {name} | {fresh} | {len(total)} | {hit} |")
    quiet = [n for n in res.phase1_sources
             if not any(x.source == n and x.day >= res.window_start for x in res.signals)]
    if quiet:
        out += ["", "本期没有产出的一期来源：" + "、".join(quiet) + "。连续 4 周没产出的，复盘时考虑删掉。"]
    out.append("")

    issues = res.ignored + res.problems + res.policy_notes
    out += ["## 数据问题", ""]
    out += [f"- {m}" for m in issues] or ["没有。"]
    out.append("")

    out += ["## GPT 复核意见", "", "（把 GPT 的复核结论贴在这里。）", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- 入口

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="机会雷达第一阶段打分脚本")
    ap.add_argument("--date", help="截止日，YYYY-MM-DD，默认今天")
    ap.add_argument("--since", help="周报统计区间的起点，YYYY-MM-DD，默认是截止日前 7 天")
    ap.add_argument("--root", help="radar 目录，默认是本脚本的上一级目录")
    ap.add_argument("--check", action="store_true", help="只检查数据，不生成周报")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else DEFAULT_ROOT
    until = parse_date(args.date) if args.date else date.today()
    if until is None:
        print(f"日期格式不对：{args.date}，应为 YYYY-MM-DD", file=sys.stderr)
        return 2

    rules = load_rules(root)
    since = parse_date(args.since) if args.since else None
    if args.since and since is None:
        print(f"日期格式不对：{args.since}，应为 YYYY-MM-DD", file=sys.stderr)
        return 2
    if since and since > until:
        print("--since 不能晚于截止日", file=sys.stderr)
        return 2
    res = run(root, until, since)
    issues = res.ignored + res.problems + res.policy_notes

    if args.check:
        for m in issues:
            print(m)
        print(f"有效信号 {len(res.signals)} 条，问题 {len(issues)} 处。")
        return 1 if issues else 0

    y, w, _ = until.isocalendar()
    path = root / "周报" / f"{y}-W{w:02d}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(res, rules, root), encoding="utf-8")
    print(f"已生成 {path.relative_to(root.parent) if root.parent in path.parents else path}")
    if issues:
        print(f"数据有 {len(issues)} 处问题，详见周报最后的“数据问题”一节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""score.py 的测试，用手工编的样例数据。

运行：python3 -m unittest discover -s radar/tests
"""
import csv
import importlib.util
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

RADAR = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("score", RADAR / "规则" / "score.py")
score = importlib.util.module_from_spec(spec)
sys.modules["score"] = score  # dataclass 需要能从 sys.modules 找到模块
spec.loader.exec_module(score)

DIR_COLS = ["编号", "名称", "类型", "状态", "命中不碰", "准入成本", "准入成本依据", "窗口剩余", "窗口依据",
            "每月金额元", "金额依据", "频次", "频次依据", "持续性", "持续性依据", "技术门槛", "技术门槛依据",
            "决策链", "决策链依据", "备注"]
SIG_COLS = ["编号", "日期", "来源", "所属线", "阶段", "链接", "原文摘录", "方向编号", "付费证据", "金额元", "备注"]
POL_COLS = ["编号", "标题", "发文机关", "链接", "阶段", "施行日期", "管哪些企业", "类型", "涉及金额",
            "新增许可要求", "更新日期", "关联方向", "备注"]
VAL_COLS = ["日期", "方向编号", "方式", "结果", "证据", "备注"]

TODAY = date(2026, 10, 18)

LONG_OK = {
    "编号": "D001", "名称": "小微制造企业回款管理", "类型": "长钱", "状态": "候选", "命中不碰": "否",
    "准入成本": "只需营业执照", "每月金额元": "20000", "金额依据": "访谈 1 原话",
    "频次": "每月", "持续性": "三年以上", "技术门槛": "需要高可靠或权限审计", "决策链": "老板一人拍板",
}
FAST_OK = {
    "编号": "D002", "名称": "个人 AI 助手代部署", "类型": "快钱", "状态": "观察中", "命中不碰": "否",
    "准入成本": "只需营业执照", "窗口剩余": "无大厂进场",
}


def sig(i, day, source, line, stage, direction, pay="无", amount="", link=None, excerpt="原文"):
    return {"编号": f"S{i:04d}", "日期": day, "来源": source, "所属线": line, "阶段": stage,
            "链接": link if link is not None else f"https://example.com/{i}", "原文摘录": excerpt,
            "方向编号": direction, "付费证据": pay, "金额元": amount}


def write(path, cols, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "规则").mkdir()
        (self.tmp / "数据").mkdir()
        for name in ("打分规则.toml", "来源清单.csv"):
            shutil.copy(RADAR / "规则" / name, self.tmp / "规则" / name)
        self.rules = score.load_rules(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def build(self, directions, signals, policies=(), validations=()):
        write(self.tmp / "数据" / "方向.csv", DIR_COLS, directions)
        write(self.tmp / "数据" / "信号.csv", SIG_COLS, signals)
        write(self.tmp / "数据" / "政策.csv", POL_COLS, policies)
        write(self.tmp / "数据" / "验证.csv", VAL_COLS, validations)
        return score.run(self.tmp, TODAY)

    def find(self, res, did):
        return next(s for s in res.long + res.fast if s.id == did)

    def test_long_money_arithmetic_and_crossing(self):
        res = self.build([LONG_OK], [
            sig(1, "2026-10-15", "线下访谈", "经营压力", "开始花钱", "D001", link="访谈/a.md"),
            sig(2, "2026-09-01", "国务院政策文件库", "政策", "撑腰", "D001"),
        ])
        s = self.find(res, "D001")
        # 金额 2 万 → 4；每月 3；三年以上 5；高可靠 5；老板拍板 5；准入 1.0；两条线 1.3
        self.assertAlmostEqual(s.total, 4 * 3 * 5 * 5 * 5 * 1.0 * 1.3)
        self.assertTrue(s.passed)
        self.assertEqual(s.lines, ["经营压力", "政策"])

    def test_crossing_ignores_old_signals(self):
        res = self.build([LONG_OK], [
            sig(1, "2026-10-15", "线下访谈", "经营压力", "开始花钱", "D001"),
            sig(2, "2026-01-01", "国务院政策文件库", "政策", "撑腰", "D001"),  # 超过 90 天
        ])
        self.assertAlmostEqual(self.find(res, "D001").total, 4 * 3 * 5 * 5 * 5)

    def test_long_money_gate_needs_offline_or_two_amounts(self):
        res = self.build([LONG_OK], [sig(1, "2026-10-15", "国家统计局", "经营压力", "压力出现", "D001",
                                         amount="5000")])
        self.assertFalse(self.find(res, "D001").passed)
        res = self.build([LONG_OK], [
            sig(1, "2026-10-15", "国家统计局", "经营压力", "压力出现", "D001", amount="5000"),
            sig(2, "2026-10-16", "互动易和e互动", "经营压力", "压力持续", "D001", amount="8000"),
        ])
        self.assertTrue(self.find(res, "D001").passed)

    def test_missing_subscore_is_not_guessed(self):
        row = dict(LONG_OK, 频次="", 每月金额元="")
        res = self.build([row], [sig(1, "2026-10-15", "线下访谈", "经营压力", "开始花钱", "D001")])
        s = self.find(res, "D001")
        self.assertIsNone(s.total)
        self.assertEqual(len(s.missing), 2)

    def test_fast_money_score_and_gate(self):
        signals = [
            sig(1, "2026-10-13", "GitHub", "技术普及", "新能力出现", "D002"),
            sig(2, "2026-10-14", "linux.do", "技术普及", "出现摩擦", "D002"),
            sig(3, "2026-10-15", "闲鱼淘宝和知识付费平台", "技术普及", "有人付钱", "D002", pay="服务或课程上架"),
            sig(4, "2026-10-16", "V2EX", "技术普及", "出现摩擦", "D002"),
        ]
        res = self.build([FAST_OK], signals)
        s = self.find(res, "D002")
        # 近 7 天 4 条 → 4；付费 3；无大厂 5；一条线 1.0
        self.assertAlmostEqual(s.total, 4 * 3 * 5)
        self.assertTrue(s.passed)
        res = self.build([FAST_OK], signals[:2])  # 没有付费信号
        self.assertFalse(self.find(res, "D002").passed)

    def test_signal_without_link_or_unknown_source_is_ignored(self):
        res = self.build([LONG_OK], [
            sig(1, "2026-10-15", "线下访谈", "经营压力", "开始花钱", "D001", link=""),
            sig(2, "2026-10-15", "某个没登记的网站", "经营压力", "开始花钱", "D001"),
            sig(3, "2026-10-15", "国家统计局", "经营压力", "不存在的阶段", "D001"),
            sig(4, "2026-10-15", "国家统计局", "经营压力", "压力出现", "D999"),
        ])
        self.assertEqual(len(res.signals), 0)
        self.assertEqual(len(res.ignored), 4)

    def test_future_signals_are_excluded(self):
        res = self.build([LONG_OK], [sig(1, "2026-10-25", "线下访谈", "经营压力", "开始花钱", "D001")])
        self.assertEqual(len(res.signals), 0)
        self.assertEqual(res.ignored, [])

    def test_redline_and_unobtainable_licence_are_filtered(self):
        a = dict(LONG_OK, 编号="D003", 命中不碰="第一条：帮人绕过访问限制")
        b = dict(LONG_OK, 编号="D004", 准入成本="个人拿不到")
        res = self.build([a, b], [])
        self.assertTrue(self.find(res, "D003").filtered)
        self.assertTrue(self.find(res, "D004").filtered)

    def test_policy_stage_cannot_go_backwards(self):
        base = {"编号": "P001", "标题": "某条例", "发文机关": "国务院", "类型": "加义务"}
        res = self.build([], [], policies=[
            dict(base, 阶段="发布", 更新日期="2026-09-01"),
            dict(base, 阶段="征求意见", 更新日期="2026-10-01"),
        ])
        self.assertTrue(any("倒退" in n for n in res.policy_notes))

    def test_report_is_written(self):
        self.build([LONG_OK, FAST_OK], [
            sig(1, "2026-10-15", "线下访谈", "经营压力", "开始花钱", "D001", link="访谈/a.md"),
        ], validations=[{"日期": "2026-10-16", "方向编号": "D001", "方式": "访谈", "结果": "确认痛点",
                         "证据": "访谈/a.md"}])
        code = score.main(["--root", str(self.tmp), "--date", TODAY.isoformat()])
        self.assertEqual(code, 0)
        text = (self.tmp / "周报" / "2026-W42.md").read_text(encoding="utf-8")
        self.assertIn("## 长钱候选", text)
        self.assertIn("D001 小微制造企业回款管理", text)
        self.assertIn("确认痛点", text)

    def test_check_mode_returns_nonzero_on_problems(self):
        self.build([LONG_OK], [sig(1, "2026-10-15", "线下访谈", "经营压力", "开始花钱", "D001", link="")])
        self.assertEqual(score.main(["--root", str(self.tmp), "--date", "2026-10-18", "--check"]), 1)


if __name__ == "__main__":
    unittest.main()

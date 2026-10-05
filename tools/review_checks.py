"""审核用校验脚本:番型公式/结算/听任意判定/弃胡阈值。"""
import random
from functools import lru_cache

# ---------- 1. 番型公式 vs 平台实测样例 ----------
def fan(branch=1, chain=0, four_white=False, baotou=False):
    return branch * 2 ** chain * (2 if four_white else 1) * (2 if baotou else 1)

def settle(F, base=1):
    dealer_win = 3 * base * F * 8
    nondealer_win = base * F * 8 + 2 * base * F * 1
    return dealer_win, nondealer_win

samples = [
    ("平胡+爆头", fan(1, 0, False, True), 48, 20),
    ("平胡+杠开+爆头", fan(1, 1, False, True), 96, 40),
    ("平胡+连杠x2+爆头", fan(1, 2, False, True), 192, 80),
    ("平胡+连杠x3+爆头", fan(1, 3, False, True), 384, 160),
    ("平胡+三财飘+4白+爆头", fan(1, 3, True, True), 768, 320),
]
print("== 样例校验 ==")
for name, F, d, n in samples:
    dw, nw = settle(F)
    print(f"{name:<18} 番={F:<4} 庄胡={dw:<5}({'OK' if dw == d else 'X'}) 闲胡={nw:<4}({'OK' if nw == n else 'X'})")
print("平胡上限", fan(1, 6, True, True), " 全局上限", fan(16, 3, True, True))

# ---------- 2. 席位方差 ----------
# 每家每局胡牌概率 p 相同、番 F 相同时,单局得分二阶矩
dealer_m2 = 1 * 24 ** 2 + 3 * 8 ** 2          # 自己胡 +24F;任一闲胡 -8F
nondealer_m2 = 10 ** 2 + 8 ** 2 + 2 * 1 ** 2  # 自己胡 +10F;庄胡 -8F;另两闲胡 各 -1F
print("\n== 方差 ==  庄/闲 二阶矩比 =", dealer_m2 / nondealer_m2, " 标准差比 =", (dealer_m2 / nondealer_m2) ** 0.5)

# ---------- 3. 听任意判定:穷举 vs 结构化刻画 ----------
# 33 种真牌(27 数牌 + 东南西北中发),白板为百搭 J
REAL = 33

def is_suit(k):
    return k < 27

@lru_cache(maxsize=None)
def sets_only(counts, j, need_sets):
    """counts(tuple) + j 张百搭能否恰好分解成 need_sets 个面子(无对子)。"""
    i = next((x for x, c in enumerate(counts) if c), None)
    if i is None:
        return j == 3 * need_sets
    if need_sets == 0:
        return False
    c = list(counts)
    # 刻子(用 0~2 张百搭)
    for use in range(0, 3):
        real = 3 - use
        if c[i] >= real and j >= use and real >= 1:
            c2 = c[:]; c2[i] -= real
            if sets_only(tuple(c2), j - use, need_sets - 1):
                return True
    # 顺子:i 作为最小牌,i+1/i+2 缺的用百搭补
    if is_suit(i) and i % 9 <= 6:
        miss = 0; c2 = c[:]; c2[i] -= 1
        for k in (i + 1, i + 2):
            if c2[k] > 0:
                c2[k] -= 1
            else:
                miss += 1
        if miss <= j and sets_only(tuple(c2), j - miss, need_sets - 1):
            return True
    # i 作为顺子中间/末尾且左侧由百搭补(i 为该花色 1 或 2 时)
    if is_suit(i):
        r = i % 9
        for start in (i - 2, i - 1):
            if start < 0 or start // 9 != i // 9 or start % 9 > 6:
                continue
            c2 = c[:]; miss = 0; ok = True
            for k in (start, start + 1, start + 2):
                if k < i:
                    miss += 1          # 比 i 小的位置必为百搭(i 是最小真牌)
                elif c2[k] > 0:
                    c2[k] -= 1
                else:
                    miss += 1
            if miss <= j and miss < 3 and sets_only(tuple(c2), j - miss, need_sets - 1):
                return True
    return False

def can_win_normal(counts, j, need_sets=4):
    c = list(counts)
    for p in range(REAL):  # 真对子 / 单张+百搭
        for use in (0, 1):
            if c[p] >= 2 - use and j >= use:
                c2 = c[:]; c2[p] -= 2 - use
                if sets_only(tuple(c2), j - use, need_sets):
                    return True
    if j >= 2 and sets_only(tuple(c), j - 2, need_sets):  # 百搭对
        return True
    return False

def can_win_qidui(counts, j):
    odd = sum(x % 2 for x in counts)
    return sum(counts) + j == 14 and odd <= j

def ting_any_bruteforce(counts, j, qidui=False):
    for x in range(REAL + 1):
        c = list(counts)
        jj = j
        if x == REAL:
            jj += 1
        else:
            c[x] += 1
        ok = can_win_qidui(tuple(c), jj) if qidui else can_win_normal(tuple(c), jj)
        if not ok:
            return False
    return True

def ting_any_structural(counts, j):
    # A: 去 1 白 → 4 面子;B: 去 2 白 → 3 面子 + 1 对
    if j >= 1 and sets_only(tuple(counts), j - 1, 4):
        return True
    if j >= 2:
        c = list(counts)
        for p in range(REAL):
            for use in (0, 1):
                if c[p] >= 2 - use and j - 2 >= use:
                    c2 = c[:]; c2[p] -= 2 - use
                    if sets_only(tuple(c2), j - 2 - use, 3):
                        return True
        if j >= 4 and sets_only(tuple(c), j - 4, 3):
            return True
    return False

def random_hand(nj, rng):
    wall = [k for k in range(REAL) for _ in range(4)]
    rng.shuffle(wall)
    c = [0] * REAL
    for k in wall[: 13 - nj]:
        c[k] += 1
    return tuple(c)

def near_ting_hand(nj, rng):
    """构造接近听任意的手牌以提高命中率:随机 4 面子(含百搭)+ 1 张,再扰动。"""
    while True:
        c = [0] * REAL; jleft = nj; tiles = 0
        ok = True
        for _ in range(4):
            if rng.random() < 0.5:
                k = rng.randrange(REAL); c[k] += 3
            else:
                s = rng.randrange(3); r = rng.randrange(7)
                for d in range(3):
                    c[s * 9 + r + d] += 1
        # 用百搭替换若干真牌
        for _ in range(nj):
            ks = [k for k in range(REAL) if c[k]]
            k = rng.choice(ks); c[k] -= 1
        k = rng.randrange(REAL); c[k] += 1
        # 随机扰动一张
        if rng.random() < 0.5:
            ks = [k for k in range(REAL) if c[k]]
            k = rng.choice(ks); c[k] -= 1
            c[rng.randrange(REAL)] += 1
        if max(c) <= 4 and sum(c) + nj == 13:
            return tuple(c)


NAMES = [f"{i}w" for i in range(1, 10)] + [f"{i}t" for i in range(1, 10)] + [f"{i}s" for i in range(1, 10)] + ["东", "南", "西", "北", "中", "发"]
IDX = {n: i for i, n in enumerate(NAMES)}

def parse(lst):
    c = [0] * REAL; j = 0
    for t in lst:
        if t == "白":
            j += 1
        else:
            c[IDX[t]] += 1
    return tuple(c), j

def check(name, hand, draw):
    c, j = parse(hand)
    c2, j2 = parse(hand + [draw])
    print(f"{name}: 听任意(普通)={ting_any_bruteforce(c, j)} 听任意(七对)={ting_any_bruteforce(c, j, qidui=True)}"
          f" | 摸{draw}: 普通胡={can_win_normal(c2, j2)} 七对胡={can_win_qidui(c2, j2)}")

if __name__ == "__main__":
    rng = random.Random(7)

    print("\n== 听任意:穷举 vs 结构刻画(普通胡) ==")
    for nj in (0, 1, 2, 3, 4):
        mismatch = 0; pos = 0; n = 0
        for t in range(4000):
            c = near_ting_hand(nj, rng) if t % 2 else random_hand(nj, rng)
            a = ting_any_bruteforce(c, nj)
            b = ting_any_structural(c, nj)
            n += 1; pos += a
            if a != b:
                mismatch += 1
        print(f"白={nj}: 样本 {n}, 听任意 {pos}, 不一致 {mismatch}")

    print("\n== 七对听任意:穷举 vs '单张数 ≤ 白数-1' ==")
    mm = 0; pos = 0
    for t in range(20000):
        nj = rng.randrange(0, 5)
        c = random_hand(nj, rng)
        if t % 2:  # 偏向对子多的手
            c = [0] * REAL
            for _ in range((13 - nj) // 2):
                c[rng.randrange(REAL)] += 2
            if (13 - nj) % 2:
                c[rng.randrange(REAL)] += 1
            if max(c) > 4:
                continue
            c = tuple(c)
        a = ting_any_bruteforce(c, nj, qidui=True)
        odd = sum(x % 2 for x in c)
        b = nj >= 1 and odd <= nj - 1
        pos += a; mm += (a != b)
    print("七对样本不一致:", mm, " 听任意数:", pos)

    # ---------- 4. 弃胡加倍的盈亏平衡成功率 ----------
    # 一步加倍:EV(续) = s*2kN + b*k*Nb - o*L ;EV(胡) = kN
    # 无回落(b=0)、失败即对手胡(o=1-s):s* = (1+λ)/(2+λ),λ = L/(kN)
    def s_star(lam):
        return (1 + lam) / (2 + lam)

    print("\n== 加倍盈亏平衡 s*(失败=对手先胡) ==")
    cases = [
        ("闲家,威胁来自另一闲家", lambda Fo, N: (1 * Fo) / (10 * N)),
        ("庄家,威胁来自任一闲家", lambda Fo, N: (8 * Fo) / (24 * N)),
        ("闲家,威胁来自庄家", lambda Fo, N: (8 * Fo) / (10 * N)),
    ]
    for name, lamf in cases:
        row = []
        for N, Fo in ((2, 2), (2, 4), (4, 2), (8, 4)):
            lam = lamf(Fo, N)
            row.append(f"N={N},F对={Fo}: λ={lam:.2f} s*={s_star(lam):.1%}")
        print(name, " | ".join(row))

    # 弃胡转爆头:(1-o-d)*2 - o*λ ≥ 1  →  o ≤ (1-2d)/(2+λ)
    print("\n== 普通听摸到胡张 → 弃胡改单吊白(×1→×2) 可承受的一圈内对手自摸率 o 上限(d=0) ==")
    for name, lamf in cases:
        print(name, " | ".join(f"F对={Fo}: o≤{1 / (2 + lamf(Fo, 1)):.1%}" for Fo in (2, 4, 8)))

    # 链=0 爆头态"钓鱼"(弃胡打摸到的杂牌,等白/杠):
    # 收益 ≈ k*2B*[(1-o)(1+u) - 1] - o*L ;需 u > o(1+u) + o*L/(k*2B)
    print("\n== 链0爆头态钓鱼:每圈需要的 u(下一摸可飘/可杠概率)下限 ==")
    for name, lamf in cases:
        out = []
        for o in (0.03, 0.08, 0.15):
            lam = lamf(2.5, 2)  # 对手期望番 2.5,当前 N=2(平胡爆头)
            # u > o(1+u) + o*lam  →  u(1-o) > o(1+lam)  →  u > o(1+lam)/(1-o)
            out.append(f"o={o:.0%}: u>{o * (1 + lam) / (1 - o):.1%}")
        print(name, " | ".join(out))

    # ---------- 5. 建议提交给平台 fan-calc 的歧义测试手牌(本地口径) ----------
    print("\n== 歧义测试手牌(本地口径) ==")
    check("T1 4白非听任意", ["6w","7w","8w","南","南","北","中","中","发","白","白","白","白"], "1w")
    check("T1b", ["6w","7w","8w","南","南","北","中","中","发","白","白","白","白"], "北")
    check("T2 白补四张?", ["1w","1w","4w","4w","7w","7w","东","东","南","南","西","西","白"], "东")
    check("T3 白白白白算豪华?", ["1w","1w","4w","4w","7w","7w","东","东","南","白","白","白","白"], "南")
    check("T4 形态B听任意", ["1w","2w","3w","4w","5w","6w","7w","8w","9w","东","东","白","白"], "北")
    check("T5 七对非爆头", ["1w","1w","4w","4w","7w","7w","东","东","南","南","西","北","白"], "西")

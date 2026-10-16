"""Overcooked 的像素画渲染 —— 手绘精灵 + 逐像素画布。

================================================================================
为什么是像素画
================================================================================
前两版是矢量形状（matplotlib 的圆/圆角矩形）。做高保真做不了，而"做得像"
又做不到 —— 半吊子的矢量图只会显得廉价。像素画是**能做到好看**的那一档：
形状简单、靠配色和轮廓取胜，而且天然不要求被放大得精细（放大后还是像素）。

实现上不引入任何外部素材：精灵用字符网格手写，颜色查表，最终**烘到一张
RGB 画布上一次性 `imshow(interpolation="nearest")`** —— 最近邻保证放大后是
硬边像素，不会被插值糊掉。

================================================================================
精灵记号
================================================================================
  .  透明        #  深色轮廓     F/f 地板明/暗    B/b 台面身/顶
  Y/y 洋葱明/暗   M/m 番茄明/暗   G  绿色（芽/叶）  D/d 盘子/盘沿
  P/p 锅身/锅沿   S/s 上菜口/暗纹  H  白色（厨师帽）  K  肤色
  R  队伍色（按 agent 换）

`R` 是**唯一**随 agent 变的记号，别拿它当普通颜色用：`_to_rgb` 在没传
`agent_color` 时把 `R` 当透明。番茄一开始就误用了 `R`，于是整颗番茄是空的。
"""

from __future__ import annotations

import numpy as np

TILE = 16          # 每格 16x16 像素

PALETTE = {
    ".": None,
    "#": (40, 32, 28),
    "F": (238, 222, 192),
    "f": (226, 206, 172),
    "B": (138, 96, 57),
    "b": (185, 141, 94),
    "Y": (233, 195, 74),
    "y": (201, 160, 46),
    "M": (206, 66, 43),
    "m": (163, 46, 31),
    "G": (90, 158, 74),
    "D": (251, 251, 251),
    "d": (201, 201, 201),
    "P": (59, 59, 59),
    "p": (99, 99, 99),
    "S": (207, 212, 216),
    "s": (154, 162, 168),
    "H": (253, 253, 253),
    "K": (242, 201, 160),
}


def _grid(rows: list[str]) -> np.ndarray:
    a = np.array([list(r) for r in rows], dtype="<U1")
    assert a.shape == (TILE, TILE), f"精灵必须是 {TILE}x{TILE}，得到 {a.shape}"
    return a


# ---------------------------------------------------------------------------
# 地形：程序生成（它们是重复的底子，手绘反而更难看）
# ---------------------------------------------------------------------------
def _floor(light: bool) -> np.ndarray:
    a = np.full((TILE, TILE), "F" if light else "f", dtype="<U1")
    # 每块砖加一圈略暗的缝，放大后能看出砌缝
    a[0, :] = a[-1, :] = "f" if light else "F"
    a[:, 0] = a[:, -1] = "f" if light else "F"
    return a


def _counter() -> np.ndarray:
    a = np.full((TILE, TILE), "B", dtype="<U1")
    a[0:5, :] = "b"                 # 顶面亮
    a[5, :] = "#"                   # 顶面与侧面的分界
    a[0, :] = "#"
    a[:, 0] = a[:, -1] = "#"
    a[-1, :] = "#"
    for x in (4, 11):               # 侧面的木纹：淡色刻痕，太密会读成栅栏
        a[8:14, x] = "b"
    return a


def _serve() -> np.ndarray:
    a = _counter()
    a[6:12, 2:14] = "S"
    for y in range(7, 11, 2):
        a[y, 3:13] = "s"
    return a


# ---------------------------------------------------------------------------
# 物品与角色：手绘
# ---------------------------------------------------------------------------
ONION = _grid([
    "................",
    "................",
    ".......GG.......",
    "......G..G......",
    ".....y.YY.y.....",
    "....yYYYYYYy....",
    "...yYYYYYYYYy...",
    "..yYYYYYYYYYYy..",
    "..yYYYYYYYYYYy..",
    "..yYYYYYYYYYYy..",
    "...yYYYYYYYYy...",
    "....yyyyyyyy....",
    ".....yyyyyy.....",
    "......yyyy......",
    "................",
    "................",
])

TOMATO = _grid([
    "................",
    "................",
    ".......GG.......",
    "......G..G......",
    ".....m.MM.m.....",
    "....mMMMMMMm....",
    "...mMMMMMMMMm...",
    "..mMMMMMMMMMMm..",
    "..mMMMMMMMMMMm..",
    "..mMMMMMMMMMMm..",
    "...mMMMMMMMMm...",
    "....mmmmmmmm....",
    ".....mmmmmm.....",
    "......mmmm......",
    "................",
    "................",
])

DISH = _grid([
    "................",
    "................",
    "................",
    "....dddddddd....",
    "...dDDDDDDDDd...",
    "..dDDDDDDDDDDd..",
    "..dDDddddddDDd..",
    "..dDDd....dDDd..",
    "..dDDd....dDDd..",
    "..dDDddddddDDd..",
    "..dDDDDDDDDDDd..",
    "...dDDDDDDDDd...",
    "....dddddddd....",
    "................",
    "................",
    "................",
])

POT_EMPTY = _grid([
    "................",
    "................",
    "..############..",
    "..#pppppppppp#..",
    ".##PPPPPPPPPP##.",
    ".#PPPPPPPPPPPP#.",
    ".#P..........P#.",
    ".#P..........P#.",
    ".#P..........P#.",
    ".#P..........P#.",
    ".#PPPPPPPPPPPP#.",
    ".#PPPPPPPPPPPP#.",
    ".##PPPPPPPPPP##.",
    "..############..",
    "................",
    "................",
])


def pot_with_soup(ready: bool, frac: float) -> np.ndarray:
    """锅里有汤：按进度填充高度，煮好加气泡。"""
    a = POT_EMPTY.copy()
    col = "G" if ready else "Y"
    top = 12 - int(round(max(0.0, min(1.0, frac)) * 5))   # 12 -> 7
    a[top:12, 3:13] = col
    a[top, 3:13] = "#"
    if ready:
        a[8, 5] = a[10, 11] = "H"
        a[6, 9] = "H"
    return a


CHEF = _grid([
    "................",
    "......HHHH......",
    ".....HHHHHH.....",
    "....HHHHHHHH....",
    "....HHHHHHHH....",
    "....########....",
    "....KKKKKKKK....",
    "....K#KKKK#K....",     # 两只眼睛。原来是居中的一道黑杠，读成了嘴
    "....KKKKKKKK....",
    "...RRRRRRRRRR...",
    "..RRRRRRRRRRRR..",
    "..RRRRRRRRRRRR..",
    "..RRRRRRRRRRRR..",
    "...RRRRRRRRRR...",
    "................",
    "................",
])

SOUP_BOWL = _grid([
    "................",
    "................",
    "................",
    "................",
    "....########....",
    "...##GGGGGG##...",
    "..##GGGGGGGG##..",
    "..#GGGGGGGGGG#..",
    "..#GGG####GGG#..",
    "..#GGGGGGGGGG#..",
    "...#GGGGGGGG#...",
    "....##GGGG##....",
    ".....######.....",
    "................",
    "................",
    "................",
])

AGENT_RGB = [(232, 85, 45), (47, 127, 216), (47, 174, 90), (164, 79, 196), (232, 160, 45)]


def _to_rgb(sprite: np.ndarray, agent_color: tuple | None = None) -> np.ndarray:
    """字符网格 -> RGBA 画布。`R` 用队伍色，`.` 透明。

    ⚠️ RGB 是 **0..255**，alpha 是 **0..1** —— 两套刻度故意不统一（RGB 直接对齐
    调色板，读代码时不用换算；alpha 要参与 `_blit` 的线性混合，必须是 0..1）。
    一开始把 alpha 也写成 255.0，混合式就成了 `-254*canvas + 255*sub`，
    大负数被裁成 0，整张图全黑。
    """
    h, w = sprite.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    for ch in np.unique(sprite):
        rgb = agent_color if ch == "R" else PALETTE.get(ch)
        if ch != "R" and ch not in PALETTE:
            raise KeyError(f"精灵里出现未定义的记号 {ch!r}")
        if rgb is None:
            continue
        out[sprite == ch] = (*rgb, 1.0)
    return out


def _blit(canvas: np.ndarray, sprite: np.ndarray, px: int, py: int,
          agent_color=None, scale: int = 1) -> None:
    """把精灵贴到像素画布的 (px, py)（左上角，单位：像素）。"""
    rgba = _to_rgb(sprite, agent_color)
    if scale != 1:
        rgba = np.repeat(np.repeat(rgba, scale, 0), scale, 1)
    h, w = rgba.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(px, 0), max(py, 0)
    x1, y1 = min(px + w, W), min(py + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    sub = rgba[y0 - py:y1 - py, x0 - px:x1 - px]
    a = sub[..., 3:4]
    canvas[y0:y1, x0:x1, :3] = (1 - a) * canvas[y0:y1, x0:x1, :3] + a * sub[..., :3]
    canvas[y0:y1, x0:x1, 3] = np.maximum(canvas[y0:y1, x0:x1, 3], sub[..., 3])


def canvas_from_state(env, show_view: bool = True) -> np.ndarray:
    """按当前状态烘一张 (H*TILE, W*TILE, 4) 的像素画布。"""
    mdp, state = env._mdp, env._state
    grid = mdp.terrain_mtx
    rows, cols = len(grid), len(grid[0])
    # 画布全程 float32、RGB 走 0..255、alpha 走 0..1，最后一步才裁成 uint8。
    # `imshow` 对 **float** RGBA 期望 0..1，所以这里必须转 uint8 再交出去 ——
    # 忘了转的话每个通道都被裁到 1，整张图全白。
    canvas = np.zeros((rows * TILE, cols * TILE, 4), dtype=np.float32)
    canvas[..., :3] = np.array([247, 242, 231], dtype=np.float32)
    canvas[..., 3] = 1.0

    def cell(x, y):
        return x * TILE, y * TILE

    view_boxes = []
    if show_view:
        s = env._sight
        for i, p in enumerate(state.players):
            x, y = p.position
            col = np.array(AGENT_RGB[i % len(AGENT_RGB)], dtype=np.float32)
            view_boxes.append((
                max(x - s, 0) * TILE, max(y - s, 0) * TILE,
                min(x + s + 1, cols) * TILE, min(y + s + 1, rows) * TILE,
                col,
            ))

    # 两趟画地形：**先地板、后其它**。中间这一趟的间隙就是给视野上色留的 ——
    # 台面被染上色会发灰发紫，整张图像贴了色片而不像厨房；只染地板，厨房保持
    # 本来的配色，而地板占内部面积的大头，"谁看得见哪片区域"照样一眼可读。
    for y in range(rows):
        for x in range(cols):
            if grid[y][x] == " " or grid[y][x].isdigit():
                _blit(canvas, _floor((x + y) % 2 == 0), *cell(x, y))

    # 每个 agent 的视野：铺一层淡色。
    #
    # 这层是**演示的重点**，不是装饰：它让"谁看得见什么"直接可见，而那正是通信要
    # 解决的问题。所以它必须看得清 —— 0.12 的浓度在米色地板上几乎读不出来，实测
    # 要 0.20 才够，再配一圈同色描边，重叠处才知道边界在哪。
    #
    # ⚠️ 位置很关键：夹在**地板之后、台面之前**。地板精灵是不透明的，铺在它之前
    # 会被整个盖掉（第一版就是这样，只剩描边看得见）；台面之后又会把台面染色；
    # 角色之后则会把队伍色糊掉，谁是谁就分不清了。
    for x0, y0, x1, y1, col in view_boxes:
        canvas[y0:y1, x0:x1, :3] = 0.80 * canvas[y0:y1, x0:x1, :3] + 0.20 * col

    for y in range(rows):
        for x in range(cols):
            ch = grid[y][x]
            px, py = cell(x, y)
            if ch == "S":
                _blit(canvas, _serve(), px, py)
            elif ch == "X":
                _blit(canvas, _counter(), px, py)
            elif ch == "O":
                _blit(canvas, _counter(), px, py)
                _blit(canvas, ONION, px, py - 1)
            elif ch == "T":
                _blit(canvas, _counter(), px, py)
                _blit(canvas, TOMATO, px, py - 1)
            elif ch == "D":
                _blit(canvas, _counter(), px, py)
                _blit(canvas, DISH, px, py - 1)
            elif ch == "P":
                _blit(canvas, _counter(), px, py)
                soup = state.objects.get((x, y))
                if soup is None:
                    _blit(canvas, POT_EMPTY, px, py)
                else:
                    ct = soup._cooking_tick
                    total = max(soup.cook_time, 1) if ct >= 0 else 1
                    frac = 0.0 if ct < 0 else min(1.0, (ct + 1) / total)
                    _blit(canvas, pot_with_soup(ct >= 0 and soup.is_ready, frac), px, py)

    # 台面上散落的物品
    for (x, y), obj in state.objects.items():
        if grid[y][x] in "OPDS":
            continue
        px, py = cell(x, y)
        n = obj.name
        if n == "onion":
            _blit(canvas, ONION, px, py)
        elif n == "tomato":
            _blit(canvas, TOMATO, px, py)
        elif n == "dish":
            _blit(canvas, DISH, px, py)
        elif n.startswith("soup"):
            _blit(canvas, SOUP_BOWL, px, py)

    # 角色
    for i, p in enumerate(state.players):
        x, y = p.position
        px, py = cell(x, y)
        col = AGENT_RGB[i % len(AGENT_RGB)]
        _blit(canvas, CHEF, px, py, agent_color=col)
        if p.held_object is not None:
            dx, dy = p.orientation
            hx, hy = px + TILE // 2 + dx * 10 - 4, py + TILE // 2 + dy * 10 - 4
            n = p.held_object.name
            spr = {"onion": ONION, "tomato": TOMATO, "dish": DISH}.get(n, SOUP_BOWL)
            _blit(canvas, spr[4:12, 4:12], hx, hy, scale=1)

    # 视野描边放在最后：它要压在台面和角色上面才读得出边界
    for x0, y0, x1, y1, col in view_boxes:
        edge = 0.45 * canvas[y0:y1, x0:x1, :3] + 0.55 * col
        canvas[y0:y0 + 1, x0:x1, :3] = edge[:1]
        canvas[y1 - 1:y1, x0:x1, :3] = edge[-1:]
        canvas[y0:y1, x0:x0 + 1, :3] = edge[:, :1]
        canvas[y0:y1, x1 - 1:x1, :3] = edge[:, -1:]

    # 出口处两套刻度都要归一：RGB 裁到 0..255，alpha 从 0..1 放大回 0..255
    # （`imshow` 收 uint8 RGBA 时 alpha 按 0..255 读，留成 1 就是几乎全透明）
    canvas = np.clip(canvas, 0, 255)
    canvas[..., 3] *= 255.0
    return canvas.astype(np.uint8)

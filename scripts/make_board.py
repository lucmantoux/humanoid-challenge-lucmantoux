"""Phase 1: the A4 print pack — ArUco table board (page 1) and checkerboard (page 2).

Both pages are vector drawings at exact physical size, so printing at 100% ("Actual
size", scaling off) gives the millimetre dimensions printed below. Measure them anyway.

Usage:
    uv run python scripts/make_board.py
    uv run python scripts/make_board.py overwrite=true
"""

from __future__ import annotations

import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

import _bootstrap  # noqa: E402,F401  puts src/ on sys.path

from omegaconf import OmegaConf  # noqa: E402

from palm_prior.perception.aruco import build_board  # noqa: E402
from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_seed, should_skip  # noqa: E402

A4_W, A4_H = 210.0, 297.0  # mm, portrait
M = 1000.0  # metres -> mm
# Where the table-frame origin sits on page 1, in mm from the page's bottom-left
# when the title is at the top. Shared with the setup schematic so the drawn sheet
# matches the PDF.
BOARD_ORIGIN_PAGE_Y_MM = 118.0


def new_page() -> tuple[plt.Figure, plt.Axes]:
    """An A4 portrait figure whose axes are in millimetres."""
    fig = plt.figure(figsize=(A4_W / 25.4, A4_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W)
    ax.set_ylim(0, A4_H)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def draw_ruler(ax: plt.Axes, x: float, y: float, length: float = 100.0) -> None:
    """A length-mm scale bar with 10 mm ticks, to check the printer scale."""
    ax.plot([x, x + length], [y, y], color="black", lw=0.8)
    for i in range(int(length // 10) + 1):
        h = 3.0 if i % 5 == 0 else 1.5
        ax.plot([x + 10 * i, x + 10 * i], [y, y + h], color="black", lw=0.8)
    ax.text(x + length / 2, y - 4.5, f"{length:.0f} mm — measure this", ha="center", va="top", fontsize=8)


def draw_markers(ax: plt.Axes, tb, x0: float, y0: float) -> None:
    """Draw the board's markers as vector squares, placed by their table-frame object points.

    (x0, y0) is where the table-frame origin lands on the page, in mm. Marker corner 0 is
    the top-left of the printed marker, so image row 0 goes at the cell's maximum y.
    """
    obj = np.array(tb.board.getObjPoints()) * M
    ids = np.array(tb.board.getIds()).ravel()
    n_border = 1
    n_modules = 4 + 2 * n_border  # DICT_4X4: 4 data modules plus a 1-module black border
    for marker_id, quad in zip(ids, obj):
        bits = cv2.aruco.generateImageMarker(tb.dictionary, int(marker_id), n_modules, borderBits=n_border)
        assert bits.shape == (n_modules, n_modules), bits.shape
        left, top = quad[:, 0].min(), quad[:, 1].max()
        step = (quad[:, 0].max() - left) / n_modules
        for r in range(n_modules):
            for c in range(n_modules):
                if bits[r, c] == 0:
                    ax.add_patch(
                        Rectangle(
                            (x0 + left + c * step, y0 + top - (r + 1) * step),
                            step,
                            step,
                            facecolor="black",
                            edgecolor="none",
                        )
                    )


def page_aruco(tb, cfg) -> plt.Figure:
    w, h = tb.size[0] * M, tb.size[1] * M
    x0, y0 = (A4_W - w) / 2.0, BOARD_ORIGIN_PAGE_Y_MM  # table-frame origin on the page

    fig, ax = new_page()
    ax.text(A4_W / 2, 280, "palm-prior — ArUco table board", ha="center", fontsize=14)
    ax.text(
        A4_W / 2,
        272,
        f"{cfg.aruco.dict}, GridBoard {cfg.aruco.grid[0]}x{cfg.aruco.grid[1]}, "
        f"marker {cfg.aruco.marker_len * M:.0f} mm, gap {cfg.aruco.marker_gap * M:.0f} mm",
        ha="center",
        fontsize=9,
    )
    ax.text(A4_W / 2, 266, "Print at 100% (Actual size). Do not use 'fit to page'.", ha="center", fontsize=9)
    ax.text(
        A4_W / 2,
        260,
        "Table frame T: origin = ORIGIN corner, x right along the bottom edge, "
        "y up the left edge, z out of the page.",
        ha="center",
        fontsize=7.5,
    )

    # nothing is drawn on the board itself: the detector needs a clean white quiet zone
    draw_markers(ax, tb, x0, y0)

    arrow_len = 55.0
    ax.add_patch(FancyArrow(x0, y0 - 8, arrow_len, 0, width=0.4, head_width=3, head_length=5, color="black"))
    ax.text(x0 + arrow_len + 3, y0 - 8, "x", va="center", fontsize=12, style="italic")
    ax.add_patch(FancyArrow(x0 - 8, y0, 0, arrow_len, width=0.4, head_width=3, head_length=5, color="black"))
    ax.text(x0 - 8, y0 + arrow_len + 4, "y", ha="center", fontsize=12, style="italic")

    ax.plot([x0 - 11, x0 - 2.5], [y0 - 11, y0 - 2.5], color="black", lw=0.6)
    ax.text(x0 - 12, y0 - 11, "ORIGIN", ha="right", va="top", fontsize=9)
    ax.text(x0 - 12, y0 - 16, "(0, 0)", ha="right", va="top", fontsize=9)

    ax.text(x0 + w, y0 - 14, f"board {w:.0f} x {h:.0f} mm", ha="right", va="top", fontsize=8)
    draw_ruler(ax, x0, 95.0)
    return fig


def page_checkerboard(cfg) -> tuple[plt.Figure, float, float]:
    n_long, n_short = int(cfg.calib.checker_inner[0]) + 1, int(cfg.calib.checker_inner[1]) + 1
    sq = float(cfg.calib.checker_square) * M
    # the long side runs up the portrait page so the pattern fits on A4
    w, h = n_short * sq, n_long * sq
    assert w < A4_W and h < A4_H, (w, h)
    x0, y0 = (A4_W - w) / 2.0, 10.0

    fig, ax = new_page()
    for r in range(n_long):
        for c in range(n_short):
            if (r + c) % 2 == 0:
                ax.add_patch(
                    Rectangle((x0 + c * sq, y0 + r * sq), sq, sq, facecolor="black", edgecolor="none")
                )
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor="black", lw=0.4))
    ax.text(A4_W / 2, 288, "palm-prior — calibration checkerboard", ha="center", fontsize=14)
    ax.text(
        A4_W / 2,
        281,
        f"{cfg.calib.checker_inner[0]}x{cfg.calib.checker_inner[1]} inner corners, "
        f"{sq:.0f} mm squares, printed pattern {w:.0f} x {h:.0f} mm. Tape it flat on cardboard.",
        ha="center",
        fontsize=9,
    )
    return fig, w, h


def draw_table_layout(path) -> None:
    """Top-down of the taped table, as seen from the chair, plus which printed sheet is which."""
    fig = plt.figure(figsize=(11, 8.2))
    sheets = fig.add_gridspec(1, 2, left=0.04, right=0.96, top=0.90, bottom=0.58, wspace=0.08)
    ax_qr = fig.add_subplot(sheets[0, 0])
    ax_cb = fig.add_subplot(sheets[0, 1])
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.46])

    fig.suptitle("Which sheet is which, and where the dots go", fontsize=14)

    def _marker(axis, x, y, s):
        axis.add_patch(Rectangle((x, y), s, s, facecolor="black"))
        axis.add_patch(Rectangle((x + 0.18 * s, y + 0.18 * s), 0.64 * s, 0.64 * s, facecolor="white"))
        axis.add_patch(Rectangle((x + 0.34 * s, y + 0.34 * s), 0.32 * s, 0.32 * s, facecolor="black"))

    ax_qr.set_xlim(0, 10)
    ax_qr.set_ylim(0, 10)
    ax_qr.set_aspect("equal")
    ax_qr.axis("off")
    for col, row in ((1, 5.2), (5.2, 5.2), (1, 1), (5.2, 1)):
        _marker(ax_qr, col, row, 3.6)
    ax_qr.set_title(
        "PAGE 1 — tape this on the table\n"
        "One sheet, four QR-like squares. Do not cut them apart.\n"
        "If you only have three, one was cut off: reprint the page.",
        fontsize=9,
    )

    ax_cb.set_xlim(0, 7)
    ax_cb.set_ylim(0, 10)
    ax_cb.set_aspect("equal")
    ax_cb.axis("off")
    for r in range(8):
        for c in range(6):
            if (r + c) % 2 == 0:
                ax_cb.add_patch(Rectangle((0.4 + c * 1.0, 0.4 + r * 1.15), 1.0, 1.15, facecolor="black"))
    ax_cb.set_title(
        "PAGE 2 — the checkerboard\n"
        "Tape it flat on cardboard. Hold it in your hand for calib.mp4.\n"
        "It does not get taped to the table.",
        fontsize=9,
    )

    ax.set_xlim(-8, 78)
    ax.set_ylim(-14, 50)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-4, -6), 76, 52, facecolor="#f4f1ea", edgecolor="#888", lw=1))
    ax.add_patch(Rectangle((0, 0), 13.5, 13.5, facecolor="white", edgecolor="black", lw=1))
    for col, row in ((0.6, 7.2), (7.2, 7.2), (0.6, 0.6), (7.2, 0.6)):
        _marker(ax, col, row, 5.6)
    ax.text(0, -3.2, "ORIGIN corner", fontsize=8, ha="left")
    dots = {
        1: (22, 40), 2: (37, 40), 3: (52, 40),
        4: (22, 25), 5: (37, 25), 6: (52, 25),
        7: (22, 10), 8: (37, 10), 9: (52, 10),
    }
    for number, (x, y) in dots.items():
        ax.scatter([x], [y], s=80, c="black", zorder=3)
        ax.text(x, y + 1.6, str(number), ha="center", va="bottom", fontsize=9)
    ax.plot([64, 72], [40, 40], color="black", lw=4, solid_capstyle="butt")
    ax.text(68, 42.2, "REST", ha="center", fontsize=8)
    ax.annotate("", xy=(70, 2), xytext=(16, 2), arrowprops={"arrowstyle": "->", "color": "#333"})
    ax.text(43, 3.4, "x  (to your right)", ha="center", fontsize=8, color="#333")
    ax.annotate("", xy=(16, 36), xytext=(16, 8), arrowprops={"arrowstyle": "->", "color": "#333"})
    ax.text(16.8, 22, "y  (toward you)", fontsize=8, color="#333", rotation=90, va="center")
    ax.text(34, 47.5, "YOU, sitting here", ha="center", fontsize=11)
    ax.text(34, -12.5, "PHONE on the tripod, landscape, looking at you", ha="center", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _dim(ax, p, q, text, dx=0.0, dy=0.0):
    """A double-headed dimension from p to q, labelled at the midpoint plus (dx, dy)."""
    ax.annotate(
        "",
        xy=q,
        xytext=p,
        arrowprops={"arrowstyle": "<->", "color": "#1d4e89", "lw": 0.9, "shrinkA": 0, "shrinkB": 0},
        zorder=4,
    )
    ax.text(
        (p[0] + q[0]) / 2 + dx,
        (p[1] + q[1]) / 2 + dy,
        text,
        ha="center",
        va="center",
        fontsize=8,
        color="#1d4e89",
        zorder=5,
    )


def draw_setup_schematic(cfg, path) -> None:
    """Dimensioned top view and side view of the taped table, in centimetres.

    Dot positions come from data/raw/dots.yaml. The board size comes from the ArUco
    config. Phone height, setback and the rest-mark length are the filming-guide numbers.
    """
    dots_file = OmegaConf.load(resolve(cfg.paths.dots))
    dots = {int(k): (float(v[0]), float(v[1])) for k, v in dots_file.dots.items()}
    n_x, n_y = int(cfg.aruco.grid[0]), int(cfg.aruco.grid[1])
    side = float(cfg.aruco.marker_len) * 100.0  # m -> cm
    gap = float(cfg.aruco.marker_gap) * 100.0
    board = n_x * side + (n_x - 1) * gap
    board_h = n_y * side + (n_y - 1) * gap
    assert abs(board - board_h) < 1e-9

    # Page 1 is A4 portrait. ORIGIN is the marker corner, not the paper corner.
    page_x0 = -((A4_W - board * 10.0) / 2.0) / 10.0  # cm
    page_y0 = -BOARD_ORIGIN_PAGE_Y_MM / 10.0
    page_w, page_h = A4_W / 10.0, A4_H / 10.0

    rest_x, rest_y, rest_len = 64.0, 40.0, 8.0  # filming guide Step 2
    lens_x = dots[8][0]                          # centred on dot 8
    setback = 40.0                               # cm back from the y = 10 row
    lens_z = 55.0                                # cm, lens glass above the table
    lens_y = dots[7][1] - setback
    aim = dots[5]
    tilt = float(np.degrees(np.arctan2(lens_z, aim[1] - lens_y)))

    fig, (top, side_ax) = plt.subplots(1, 2, figsize=(14.5, 7.6))
    fig.suptitle(
        "Table setup, in centimetres, measured from the ORIGIN corner\n"
        "Top view is what you see sitting in the chair: you are at the top, the phone is at the bottom",
        fontsize=12,
    )

    top.set_aspect("equal")
    top.set_xlim(-28, 84)
    top.set_ylim(-48, 50)
    top.axis("off")
    top.set_title("Top view", fontsize=11)

    top.add_patch(Rectangle((page_x0, page_y0), page_w, page_h, facecolor="#f7f7f7", edgecolor="#888", lw=0.8, zorder=0))
    top.text(page_x0 + page_w / 2, page_y0 - 1.6, "A4 sheet (page 1)", ha="center", va="top", fontsize=7, color="#555")
    top.add_patch(Rectangle((0, 0), board, board, facecolor="white", edgecolor="black", lw=1.2, zorder=1))
    for ix in range(n_x):
        for iy in range(n_y):
            top.add_patch(
                Rectangle(
                    (ix * (side + gap), iy * (side + gap)),
                    side,
                    side,
                    facecolor="black",
                    zorder=2,
                )
            )
    top.scatter([0], [0], s=28, c="black", zorder=3)
    top.text(0.4, -1.5, "ORIGIN\nmeasure from here", fontsize=7.5, ha="left", va="top")
    top.annotate("", xy=(18, -0.2), xytext=(6, -0.2), arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8})
    top.text(18.4, -0.2, "x", fontsize=8, va="center")
    top.annotate("", xy=(-0.2, 16), xytext=(-0.2, 6), arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8})
    top.text(-0.2, 16.5, "y", fontsize=8, ha="center")

    for number, (x, y) in dots.items():
        top.scatter([x], [y], s=55, c="black", zorder=3)
        top.text(x, y + 1.5, str(number), ha="center", va="bottom", fontsize=9)
    top.plot([rest_x, rest_x + rest_len], [rest_y, rest_y], color="black", lw=5, solid_capstyle="butt", zorder=3)
    top.text(rest_x + rest_len / 2, rest_y + 1.8, "REST", ha="center", fontsize=8)
    top.scatter([lens_x], [lens_y], s=70, c="#1d4e89", marker="s", zorder=3)
    top.text(lens_x - 2.2, lens_y, "LENS", ha="right", va="center", fontsize=8, color="#1d4e89")

    # Spacings along x, below the phone, and along y, left of the sheet.
    x_marks = [0.0, dots[7][0], dots[8][0], dots[9][0], rest_x]
    for a, b in zip(x_marks, x_marks[1:]):
        _dim(top, (a, -40), (b, -40), f"{b - a:.0f}", dy=-1.6)
    y_marks = [0.0, dots[7][1], dots[5][1], dots[1][1]]
    for a, b in zip(y_marks, y_marks[1:]):
        _dim(top, (-16, a), (-16, b), f"{b - a:.0f}", dx=-2.2)
    _dim(top, (lens_x + 14, lens_y), (lens_x + 14, dots[7][1]), f"{setback:.0f}", dx=2.4)
    _dim(top, (board + 0.4, -3.2), (board + 0.4, board), f"{board:.1f}", dx=3.2)
    top.text(36, 46, "YOU", ha="center", fontsize=11)
    top.text(8, -46, "every dot is measured from ORIGIN, not from the edge of the paper", fontsize=8, color="#333")

    side_ax.set_aspect("equal")
    side_ax.set_xlim(-42, 50)
    side_ax.set_ylim(-8, 68)
    side_ax.set_xlabel("toward you  →   (cm)")
    side_ax.set_title("Side view, through dot 8", fontsize=11)
    side_ax.axhline(0, color="black", lw=1)
    side_ax.text(46, 1.2, "table", ha="right", fontsize=8)
    side_ax.plot([lens_y, lens_y], [0, lens_z], color="#1d4e89", lw=1.2)
    side_ax.scatter([lens_y], [lens_z], s=70, c="#1d4e89", marker="s", zorder=3)
    side_ax.text(lens_y - 1.5, lens_z + 1.5, "LENS", ha="right", fontsize=8, color="#1d4e89")
    side_ax.plot([lens_y, aim[1]], [lens_z, 0], color="#1d4e89", lw=0.8, ls="--")
    side_ax.scatter([aim[1]], [0], s=40, c="black", zorder=3)
    side_ax.text(aim[1], 2.2, "dot 5", ha="center", fontsize=8)
    for y, name in ((dots[8][1], "dots 7–9"), (dots[1][1], "dots 1–3")):
        side_ax.plot([y, y], [0, 1.2], color="black", lw=1)
        side_ax.text(y, -2.4, name, ha="center", fontsize=7)
    _dim(side_ax, (lens_y, 8), (dots[7][1], 8), f"{setback:.0f}", dy=2.2)
    _dim(side_ax, (lens_y - 8, 0), (lens_y - 8, lens_z), f"{lens_z:.0f}", dx=-2.6)
    side_ax.text(
        -6,
        62,
        f"tilt {tilt:.0f}° down, aimed at dot 5\nmeasure 55 cm to the lens glass",
        fontsize=8,
        color="#1d4e89",
    )
    side_ax.spines["top"].set_visible(False)
    side_ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    out_pdf = resolve(cfg.paths.results) / "print" / "board_a4.pdf"
    preview = out_pdf.with_name("board_preview.png")
    layout_png = out_pdf.with_name("table_layout.png")
    schematic_png = out_pdf.with_name("setup_schematic.png")
    ensure_dir(out_pdf.parent)
    draw_table_layout(layout_png)
    print(f"wrote {layout_png}")
    draw_setup_schematic(cfg, schematic_png)
    print(f"wrote {schematic_png}")
    if should_skip(out_pdf, bool(cfg.get("overwrite", False))):
        return

    with Timer("make_board"):
        tb = build_board(cfg)
        fig_board = page_aruco(tb, cfg)
        fig_check, check_w, check_h = page_checkerboard(cfg)
        with PdfPages(out_pdf) as pdf:
            pdf.savefig(fig_board)
            pdf.savefig(fig_check)
        fig_board.savefig(preview.with_name("board_page1.png"), dpi=110)
        fig_check.savefig(preview.with_name("board_page2.png"), dpi=110)
        plt.close(fig_board)
        plt.close(fig_check)

        page1 = plt.imread(preview.with_name("board_page1.png"))
        page2 = plt.imread(preview.with_name("board_page2.png"))
        fig, axes = plt.subplots(1, 2, figsize=(9, 6.8))
        for ax, page in zip(axes, (page1, page2)):
            ax.imshow(page)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(preview, dpi=120)
        plt.close(fig)
        preview.with_name("board_page1.png").unlink()
        preview.with_name("board_page2.png").unlink()

    w, h = tb.size[0] * M, tb.size[1] * M
    print(f"wrote {out_pdf}")
    print(f"wrote {preview}")
    print("Measure after printing (both pages A4 portrait, 100% scale):")
    print(f"  page 1  marker side         {cfg.aruco.marker_len * M:.0f} mm")
    print(f"  page 1  gap between markers {cfg.aruco.marker_gap * M:.0f} mm")
    print(f"  page 1  whole board         {w:.0f} x {h:.0f} mm")
    print("  page 1  scale bar           100 mm")
    print(f"  page 2  square side         {cfg.calib.checker_square * M:.0f} mm")
    print(
        f"  page 2  whole pattern       {check_w:.0f} x {check_h:.0f} mm "
        f"({cfg.calib.checker_inner[1] + 1} x {cfg.calib.checker_inner[0] + 1} squares)"
    )
    print("If a measurement is off, reprint with scaling disabled; do not edit the config.")


if __name__ == "__main__":
    main(sys.argv[1:])

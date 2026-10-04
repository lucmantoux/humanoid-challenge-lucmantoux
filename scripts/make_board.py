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

from palm_prior.perception.aruco import build_board  # noqa: E402
from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_seed, should_skip  # noqa: E402

A4_W, A4_H = 210.0, 297.0  # mm, portrait
M = 1000.0  # metres -> mm


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
    x0, y0 = (A4_W - w) / 2.0, 118.0  # where the table-frame origin lands on the page

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


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    out_pdf = resolve(cfg.paths.results) / "print" / "board_a4.pdf"
    preview = out_pdf.with_name("board_preview.png")
    ensure_dir(out_pdf.parent)
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

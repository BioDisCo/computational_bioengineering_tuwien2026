from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


HERE = Path(__file__).resolve().parent
BOX_LEFT = 0.05
BOX_RIGHT = 0.95
BOX_BOUNDS = (BOX_LEFT, BOX_RIGHT, BOX_LEFT, BOX_RIGHT)
BALL_RADIUS = 0.06
NM_SCALE_BAR = (10.0 / 3.4) * (2 * BALL_RADIUS)
FRAMES = 900
FRAME_DT = 0.01
FPS = 50
FIGSIZE = (3.0, 3.0)
POS0 = np.array([[0.30, 0.30], [0.65, 0.62]])
VEL0 = 1.25 * np.array([[0.40, 0.30], [-0.30, -0.20]])


def save_mp4(anim: Any, output: Path, frames: int, fps: int = 25, dpi: int = 200) -> None:
    fig = anim._fig
    fig.set_dpi(dpi)
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")

    for frame in range(frames):
        anim._draw_next_frame(frame, blit=False)
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        writer.write(bgr)

    writer.release()
    plt.close(fig)
    print(output)


def boxed_two_ball_animation(
    pos_traj: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    radius: float,
    title: str,
    figsize: tuple[float, float],
) -> animation.FuncAnimation:
    x_min, x_max, y_min, y_max = bounds
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()
    ax.add_patch(
        patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1.5,
            edgecolor="black",
            facecolor="white",
        )
    )
    scale_x = x_min + 0.07 * (x_max - x_min)
    scale_y = y_min + 0.08 * (y_max - y_min)
    ax.plot(
        [scale_x, scale_x + NM_SCALE_BAR],
        [scale_y, scale_y],
        "k-",
        lw=2,
        solid_capstyle="butt",
    )
    ax.plot([scale_x, scale_x], [scale_y - 0.015, scale_y + 0.015], "k-", lw=1.5)
    ax.plot(
        [scale_x + NM_SCALE_BAR, scale_x + NM_SCALE_BAR],
        [scale_y - 0.015, scale_y + 0.015],
        "k-",
        lw=1.5,
    )
    ax.text(
        scale_x + NM_SCALE_BAR / 2,
        scale_y - 0.03,
        "1 nm",
        ha="center",
        va="top",
        fontsize=7,
    )

    colors = ["C0", "C1"]
    circles = [
        ax.add_patch(
            patches.Circle(
                pos_traj[0, i],
                radius,
                lw=1.2,
                edgecolor=colors[i],
                facecolor=colors[i],
                alpha=0.55,
            )
        )
        for i in range(2)
    ]
    plt.tight_layout()

    def update(frame: int) -> list:
        for i, circle in enumerate(circles):
            circle.center = pos_traj[frame, i]
        return circles

    return animation.FuncAnimation(
        fig, update, frames=len(pos_traj), interval=40, blit=False
    )


def simulate_argon() -> np.ndarray:
    # Match the notebook's reduced-unit visual scale:
    # notebook draw radius = 0.42 while LJ sigma = 1.0.
    sigma = BALL_RADIUS / 0.42
    eps = 0.06
    mass = 1.0
    dt = 0.0005
    substeps = int(FRAME_DT / dt)

    pos = POS0.copy()
    vel = VEL0.copy()
    pos_traj = np.zeros((FRAMES, 2, 2))

    def lj_force(r_vec: np.ndarray) -> np.ndarray:
        r = max(float(np.linalg.norm(r_vec)), 0.5 * sigma)
        sr6 = (sigma / r) ** 6
        return (4 * eps * (12 * sr6**2 - 6 * sr6) / r**2) * r_vec

    for frame in range(FRAMES):
        for _ in range(substeps):
            force = lj_force(pos[0] - pos[1])
            vel[0] += (force / mass) * dt
            vel[1] -= (force / mass) * dt
            pos += vel * dt

            for i in range(2):
                for dim in range(2):
                    if pos[i, dim] - BALL_RADIUS < BOX_LEFT:
                        pos[i, dim] = BOX_LEFT + BALL_RADIUS
                        vel[i, dim] = abs(vel[i, dim])
                    elif pos[i, dim] + BALL_RADIUS > BOX_RIGHT:
                        pos[i, dim] = BOX_RIGHT - BALL_RADIUS
                        vel[i, dim] = -abs(vel[i, dim])

        pos_traj[frame] = pos

    return pos_traj


def simulate_billiard(frames: int = FRAMES) -> np.ndarray:
    pos = POS0.copy()
    vel = VEL0.copy()
    pos_traj = np.zeros((frames, 2, 2))

    for frame in range(frames):
        pos += vel * FRAME_DT

        for i in range(2):
            for dim in range(2):
                if pos[i, dim] - BALL_RADIUS < BOX_LEFT:
                    pos[i, dim] = BOX_LEFT + BALL_RADIUS
                    vel[i, dim] = abs(vel[i, dim])
                elif pos[i, dim] + BALL_RADIUS > BOX_RIGHT:
                    pos[i, dim] = BOX_RIGHT - BALL_RADIUS
                    vel[i, dim] = -abs(vel[i, dim])

        delta = pos[0] - pos[1]
        dist = float(np.linalg.norm(delta))
        if dist < 2 * BALL_RADIUS and dist > 1e-10:
            normal = delta / dist
            rel_normal_speed = float(np.dot(vel[0] - vel[1], normal))
            if rel_normal_speed < 0:
                vel[0] -= rel_normal_speed * normal
                vel[1] += rel_normal_speed * normal
            pos[0] += (2 * BALL_RADIUS - dist) * 0.5 * normal
            pos[1] -= (2 * BALL_RADIUS - dist) * 0.5 * normal

        pos_traj[frame] = pos

    return pos_traj


def export_argon() -> None:
    pos_traj = simulate_argon()
    anim = boxed_two_ball_animation(
        pos_traj,
        bounds=BOX_BOUNDS,
        radius=BALL_RADIUS,
        title="Ar + Ar colliding (LJ force)",
        figsize=FIGSIZE,
    )
    save_mp4(anim, HERE / "argon_collision.mp4", frames=len(pos_traj), fps=FPS)


def export_billiard() -> None:
    pos_traj = simulate_billiard()
    anim = boxed_two_ball_animation(
        pos_traj,
        bounds=BOX_BOUNDS,
        radius=BALL_RADIUS,
        title="Billiard-ball collision",
        figsize=FIGSIZE,
    )
    save_mp4(anim, HERE / "billiard_collision.mp4", frames=len(pos_traj), fps=FPS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export collision animations as MP4 videos."
    )
    parser.add_argument(
        "kind",
        choices=["argon", "billiard", "all"],
        nargs="?",
        default="all",
    )
    args = parser.parse_args()

    if args.kind in {"argon", "all"}:
        export_argon()
    if args.kind in {"billiard", "all"}:
        export_billiard()


if __name__ == "__main__":
    main()

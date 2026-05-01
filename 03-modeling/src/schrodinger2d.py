"""
2D time-dependent Schrödinger equation simulator.

Uses the Visscher leapfrog scheme: real and imaginary parts of ψ are
staggered by dt/2 in time, giving a conditionally stable explicit method
with no linear algebra.  The update is simply:

    ψ_R(t + dt) = ψ_R(t) + dt · H ψ_I(t)
    ψ_I(t + dt) = ψ_I(t) - dt · H ψ_R(t + dt)

where H = -(1/2)∇² (atomic units, ℏ = mₑ = 1) and ∇² is the standard
5-point finite-difference Laplacian.  Hard walls are enforced by setting
ψ = 0 on the barrier grid points after every half-step; the barrier
geometry is supplied as a callable barrier(X, Y) → bool array.

Stability requires  dt ≤ dx² / 2  (kinetic term in 2D).
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _laplacian(f: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4 * f[1:-1, 1:-1]
    ) / dx**2
    return lap


def gaussian_packet(
    nx: int,
    ny: int,
    dx: float,
    x0: float,
    y0: float,
    kx: float,
    ky: float,
    sigma: float,
) -> NDArray[np.complex128]:
    """Normalised Gaussian wave packet centred at (x0, y0) with momentum (kx, ky)."""
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    psi = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)) * np.exp(
        1j * (kx * X + ky * Y)
    )
    return psi / np.sqrt((np.abs(psi) ** 2).sum() * dx**2)


def simulate(
    nx: int = 128,
    ny: int = 128,
    dx: float = 0.5,
    dt: float = 0.04,
    n_steps: int = 500,
    n_frames: int = 80,
    x0: float | None = None,
    y0: float | None = None,
    kx: float = 0.0,
    ky: float = 3.0,
    sigma: float = 3.0,
    barrier: Callable[[NDArray, NDArray], NDArray[np.bool_]] | None = None,
) -> tuple[list[NDArray[np.float64]], NDArray, NDArray, NDArray[np.bool_]]:
    """Run the simulation and return (frames, x, y, barrier_mask).

    barrier(X, Y) receives the meshgrid arrays (shape nx×ny, indexing='ij')
    and must return a boolean array of the same shape; True marks hard-wall
    grid points where ψ is forced to zero.  Defaults to a double-slit wall
    at 55 % of the domain height with two symmetric gaps.

    Each frame in the returned list is a 2-D array of |ψ|² at equally
    spaced time points.
    """
    assert dt <= dx**2 / 2, f"unstable: dt={dt} > dx²/2={dx**2 / 2:.4f}"

    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")

    if x0 is None:
        x0 = nx * dx / 2
    if y0 is None:
        y0 = ny * dx * 0.25

    if barrier is None:
        cx, wall_y = nx * dx / 2, ny * dx * 0.55
        barrier = lambda Xg, Yg: (  # noqa: E731
            (Yg >= wall_y)
            & (Yg < wall_y + dx)
            & ~((np.abs(Xg - (cx - 3.5)) < 1.25) | (np.abs(Xg - (cx + 3.5)) < 1.25))
        )

    mask: NDArray[np.bool_] = barrier(X, Y)

    psi = gaussian_packet(nx, ny, dx, x0, y0, kx=kx, ky=ky, sigma=sigma)
    psi_R = psi.real.copy()
    psi_I = psi.imag.copy()
    # Stagger imaginary part by −dt/2 (Visscher initialisation)
    psi_I -= (dt / 2) * (-0.5 * _laplacian(psi_R, dx))
    psi_I[mask] = 0.0

    stride = max(1, n_steps // n_frames)
    frames: list[NDArray[np.float64]] = []

    for step in range(n_steps):
        psi_R += dt * (-0.5 * _laplacian(psi_I, dx))
        psi_R[mask] = 0.0

        psi_I -= dt * (-0.5 * _laplacian(psi_R, dx))
        psi_I[mask] = 0.0

        if step % stride == 0:
            frames.append(psi_R**2 + psi_I**2)

    return frames, x, y, mask

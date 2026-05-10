import numpy as np
import scipy.stats as st
import numba
from typing import TypedDict


class CRN(TypedDict):
    species: list[str]
    r_vec: np.ndarray
    state_update_vec: np.ndarray
    rate_constants: np.ndarray


def update_propensity(
        propensities: np.ndarray,
        state: np.ndarray,
        r_vec: np.ndarray,
        rate_constants: np.ndarray) -> None:
    """Update the propensities for each reaction based on the current state.
    
    Attention: assumes that r_vec only contains 0, 1, and 2

    Parameters
    ----------
    propensities : np.ndarray
        Array to store the propensities for each reaction.
    state : np.ndarray
        Current state of the system (number of molecules of each species).
    r_vec : np.ndarray
        Reactant vector indicating how many molecules of each species are consumed by each reaction.
    rate_constants : np.ndarray
        Rate constants for each reaction.
    """
    propensities[:] = rate_constants * (
        np.where(r_vec >= 1, state, 1) * np.where(r_vec == 2, state - 1, 1)
    ).prod(axis=1)


def sample_discrete(probs: np.ndarray) -> int:
    """Randomly sample an index with probability given by probs."""
    # Generate random number
    q: float = np.random.rand()
    
    # Find index
    i: int = 0
    p_sum: float = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1


def gillespie_draw(
        state: np.ndarray,
        propensities: np.ndarray,
        crn: CRN,
        ) -> tuple[int, float]:
    update_propensity(propensities, state, crn["r_vec"], crn["rate_constants"])
    props_sum = propensities.sum()
    delta = np.random.exponential(1.0 / props_sum)
    rxn = sample_discrete(propensities / props_sum)
    return rxn, delta


def gillespie_ssa(
        crn: CRN,
        initial_state: np.ndarray,
        output_times: np.ndarray,
        ) -> np.ndarray:
    update = crn["state_update_vec"]
    
    # allocate arrays
    output_states = np.empty((len(output_times), update.shape[1]), dtype=int)
    propensities = np.zeros(update.shape[0])

    i_time: int = 1
    i: int = 0

    # initial time, state, and output
    t = output_times[0]
    state = initial_state.copy()
    state_prev = state.copy()
    output_states[0, :] = state

    # event loop
    while i < len(output_times):
        while t < output_times[i_time]:
            # 2 random draws
            event, dt = gillespie_draw(state, propensities, crn)

            # update
            state_prev = state.copy()
            state += update[event, :]
            t += dt

        # passed an output time
        i = np.sum(output_times <= t)
        output_states[i_time : min(i, len(output_times))] = state_prev
        i_time = i
    return output_states

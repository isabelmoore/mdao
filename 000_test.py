Fz_err   = self.mass*9.81 - F_lift*np.cos(bank_meas)
alpha_cmd = alpha_prev + k_F * Fz_err + alpha_ff    # k_F  [rad / N]
alpha_cmd = np.clip(alpha_cmd, α_min, α_max)

# trajectory_dqn_training.py
"""End‑to‑end DQN example for a trajectory‑planning task with **discrete**
(alpha, bank) commands.

* Action space → 25 integer actions, each maps to a (deg) pair from a small grid.
* Observation → 1‑D float32 vector (concatenate current state + history).
* Compatible with Stable‑Baselines3 `DQN`.
* Includes a **MockProblem/Scenario** so the script is runnable; replace these
  with your actual OpenMDAO problem + scenario objects.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# -----------------------------------------------------------------------------
# ╭──────────────────────────── Action grid ───────────────────────────────╮
# -----------------------------------------------------------------------------
# 5 × 5 grid → 25 discrete actions
ALPHA_GRID_DEG = [-6, -3, 0, 3, 6]        # angle‑of‑attack in degrees
BANK_GRID_DEG  = [-15, -7.5, 0, 7.5, 15]  # bank angle in degrees
ACTION_LOOKUP: list[Tuple[float, float]] = [
    (np.deg2rad(a), np.deg2rad(b))
    for a, b in itertools.product(ALPHA_GRID_DEG, BANK_GRID_DEG)
]  # len == 25

# -----------------------------------------------------------------------------
# ╭─────────────────────────── Mock back‑end  ─────────────────────────────╮
# -----------------------------------------------------------------------------
class MockProblem:
    """Bare‑bones stand‑in for your OpenMDAO `Problem`."""

    def __init__(self):
        self._vals: Dict[str, list[float]] = {
            'traj.phase0.timeseries.heading': [0.0],
            'traj.phase0.timeseries.height':  [100.0],
            'traj.phase0.timeseries.alpha_rate': [0.0],
            'traj.phase0.timeseries.bank_rate':  [0.0],
            'traj.phase0.timeseries.states:x':   [0.0],
            'traj.phase0.timeseries.states:y':   [0.0],
            'traj.phase0.timeseries.states:theta': [0.0],
            'traj.phase0.timeseries.states:wL':    [0.0],
            'traj.phase0.timeseries.states:wR':    [0.0],
        }

    # --- API shim --------------------------------------------------------
    def initial_state(self):
        return np.zeros(6, dtype=np.float32)

    def get_val(self, key):
        return np.asarray(self._vals[key])

    def set_val(self, key, val):
        # Update last sample only for mock
        if key not in self._vals:
            self._vals[key] = [val]
        else:
            self._vals[key][-1] = val


class MockScenario:
    def __init_conditions(self, problem, initial_conditions: Dict[str, float]):
        # In real code you would re‑initialise phase0 IC’s here
        for k, v in initial_conditions.items():
            problem.set_val(k, v)

# -----------------------------------------------------------------------------
# ╭─────────────────────── Trajectory environment ─────────────────────────╮
# -----------------------------------------------------------------------------
class TrajectoryEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, problem=None, scenario=None):
        super().__init__()
        self.problem = problem or MockProblem()
        self.scenario = scenario or MockScenario()

        # --- Discrete action space ---------------------------------------
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))

        # --- Observation space (flattened) -------------------------------
        self.history_length = 10
        obs_dim = 6 + self.history_length * (6 + 2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- Buffers ------------------------------------------------------
        self.pose = np.zeros(6, dtype=np.float32)
        self.actions = np.zeros(2, dtype=np.float32)
        self.prev_states = np.zeros((self.history_length, 6), dtype=np.float32)
        self.prev_inputs = np.zeros((self.history_length, 2), dtype=np.float32)

        # Targets
        self.desired_heading = 0.0
        self.desired_height = 0.0

        self.initial_conditions = {
            'traj.phase0.timeseries.alpha':   0.0,
            'traj.phase0.timeseries.bank':    0.0,
            'traj.phase0.timeseries.height':  100.0,
            'traj.phase0.timeseries.heading': 0.0,
            'traj.phase0.timeseries.roll':    0.0,
            'traj.phase0.timeseries.pitch':   0.0,
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _flatten_obs(self):
        return np.concatenate([
            self.pose,
            self.prev_states.flatten(),
            self.prev_inputs.flatten()
        ])

    def _update_history(self):
        self.prev_states = np.roll(self.prev_states, 1, axis=0)
        self.prev_inputs = np.roll(self.prev_inputs, 1, axis=0)
        self.prev_states[0] = self.pose
        self.prev_inputs[0] = self.actions

import numpy as np

# ------------------------------------------------------------------
# helper: scale [-1,1] → [lo, hi]
def denorm(val_norm, lo, hi):
    return lo + 0.5 * (val_norm + 1.0) * (hi - lo)


# ------------------------------------------------------------------
def compute_inputs(self, action):
    """
    Agent's action = [α_ff_norm, ϕ_ff_norm, k_p_h_norm, k_p_ψ_norm]
    Returns (alpha_cmd, bank_cmd) in *radians*.
    """
    # --------------- 1. decode RL action --------------------------
    alpha_ff = denorm(action[0], self.alpha_min, self.alpha_max)
    bank_ff  = denorm(action[1], self.bank_min,  self.bank_max)

    kp_h   = denorm(action[2], 0.0, self.kp_h_max)
    kp_psi = denorm(action[3], 0.0, self.kp_psi_max)

    # --------------- 2. grab current state ------------------------
    V     = self.problem.get_val('traj.phase0.timeseries.vel')[-1]          # m/s
    h     = self.problem.get_val('traj.phase0.timeseries.height')[-1]       # m
    hdot  = self.problem.get_val('traj.phase0.timeseries.hdot')[-1]         # m/s
    psi   = self.problem.get_val('traj.phase0.timeseries.heading')[-1]      # rad
    r_meas = self.problem.get_val('traj.phase0.timeseries.r')[-1]           # yaw-rate rad/s

    # --------------- 3. outer altitude loop -----------------------
    h_err      = self.desired_height - h
    gamma_cmd  = kp_h * h_err - self.ki_h * hdot
    gamma_cmd  = np.clip(gamma_cmd, -self.gamma_max, self.gamma_max)

    # --------------- 4. heading → desired yaw-rate ----------------
    psi_err     = np.arctan2(np.sin(self.desired_heading - psi),
                             np.cos(self.desired_heading - psi))
    psi_dot_cmd = kp_psi * psi_err
    psi_dot_cmd = np.clip(psi_dot_cmd, -self.psi_dot_max, self.psi_dot_max)

    # --------------- 5. map γ_cmd → α_cmd (lift inversion) --------
    rho   = self.rho                      # kg/m³  (constant or ISA lookup)
    W     = self.mass * 9.81
    CL_req = W / (0.5 * rho * V**2 * self.S)          # no cosϕ; body-lift
    alpha_trim = self.alpha_0 + (CL_req - self.CL_0) / self.CL_alpha

    gamma_meas = np.arcsin(np.clip(hdot / max(V, 1e-3), -1.0, 1.0))
    alpha_cmd  = alpha_trim + self.kp_gamma * (gamma_cmd - gamma_meas) + alpha_ff
    alpha_cmd  = np.clip(alpha_cmd, self.alpha_min, self.alpha_max)

    # --------------- 6. yaw-rate → rudder / fin deflection --------
    bank_cmd = self.rudder_gain * (psi_dot_cmd - r_meas) + bank_ff
    bank_cmd = np.clip(bank_cmd, self.bank_min, self.bank_max)

    # --------------- 7. done --------------------------------------
    return float(alpha_cmd), float(bank_cmd)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.pose[:] = 0.0
        self.prev_states[:] = 0.0
        self.prev_inputs[:] = 0.0
        self.scenario.__init_conditions(self.problem, self.initial_conditions)
        return self._flatten_obs(), {}

    def step(self, action_idx: int):
        alpha_cmd, bank_cmd = self._compute_inputs(action_idx)
        self.actions[:] = [alpha_cmd, bank_cmd]

        # Push controls to (mock) backend
        self.problem.set_val('traj.phase0.controls:alpha', alpha_cmd)
        self.problem.set_val('traj.phase0.controls:bank',  bank_cmd)

        # --- Update pose from mock backend (random walk for demo) ---------
        self.pose += np.random.randn(6).astype(np.float32) * 0.01  # stub only

        self._update_history()
        obs = self._flatten_obs()

        # --- Reward -------------------------------------------------------
        head   = self.problem.get_val('traj.phase0.timeseries.heading')[-1]
        height = self.problem.get_val('traj.phase0.timeseries.height')[-1]
        reward = - (abs(height - self.desired_height) + abs(head - self.desired_heading))

        terminated = False; truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass  # no on‑screen visualisation for now

# -----------------------------------------------------------------------------
# ╭──────────────────────────── Training script ───────────────────────────╮
# -----------------------------------------------------------------------------

def make_env():
    return TrajectoryEnv()


def main(total_timesteps: int = 200_000):
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.995,
        tau=0.005,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./tb_traj_dqn/",
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("traj_dqn")
    env.save("traj_vecnorm.pkl")
    print("✅ Training complete – model + VecNormalize stats saved.")


if __name__ == "__main__":
    main()

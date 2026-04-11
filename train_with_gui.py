import os
import sys
import argparse
import traceback
import time

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

print("Initializing training script...", flush=True)

try:
    import torch
    import numpy as np
    print("Libraries loaded successfully", flush=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.envs.wsn_env import WSNEnv
    from src.networks.wsn_policy import WSNActorCritic
    from src.agents.maml_agent import MAMLAgent
    print("Modules imported successfully", flush=True)
except Exception as e:
    print(f"ERROR importing modules: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)


def create_tasks(num_tasks, env_config):
    tasks = []
    for _ in range(num_tasks):
        config = env_config.copy()
        base = env_config['comm_range']
        config['comm_range'] = np.random.uniform(
            max(0.05, base - 0.05),
            min(0.95, base + 0.05)
        )
        env = WSNEnv(config)
        tasks.append(env)
    return tasks


def read_force_death_pct():
    """قراءة نسبة الموت الإجباري من الملف المشترك مع الواجهة."""
    try:
        fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.force_death_pct')
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                return int(f.read().strip())
    except Exception:
        pass
    return 0


def collect_rollout(env, policy, max_steps=None, deterministic=True, fixed_init=None):
    if max_steps is None:
        max_steps = 200

    N = env.num_nodes
    states_pos = np.empty((max_steps, N, 2), dtype=np.float32)
    states_bat = np.empty((max_steps, N), dtype=np.float32)
    states_con = np.empty((max_steps, N, N), dtype=np.float32)
    actions_tp = np.empty((max_steps, N), dtype=np.float32)
    actions_ss = np.empty((max_steps, N), dtype=np.float32)
    rewards_arr = np.empty(max_steps, dtype=np.float32)
    dones_arr = np.empty(max_steps, dtype=bool)

    state = env.reset()
    if fixed_init is not None:
        # Override with fixed topology (for consistent evaluation across rounds)
        env.node_positions = fixed_init['positions'].copy()
        env.battery_levels  = fixed_init['batteries'].copy()
        diff = env.node_positions[:, np.newaxis, :] - env.node_positions[np.newaxis, :, :]
        env._dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        env.update_connectivity()
        state = env._get_observation()
    done = False
    step = 0
    total_energy = 0.0
    total_delay = 0.0
    inv_pairs = 1.0 / (N * (N - 1))
    range_energy = env.energy_consumption * env.comm_range

    while not done and step < max_steps:
        action = policy.get_action(state, deterministic=deterministic)
        next_state, reward, done, _ = env.step(action)

        # حساب الطاقة: العقد المستيقظة تستهلك بناءً على قوة البث (transmit_power)، والنائمة تستهلك 0.01
        # استخدام is_sleeping لضمان تطابق الحساب مع ما تفعله البيئة
        is_sleeping = (action['sleep_schedule'] > 0.5).astype(np.float32)
        active_mask = 1.0 - is_sleeping
        
        # استهلاك الطاقة يعتمد على الإجراء الفعلي لقوة الإرسال
        energy = range_energy * np.mean(active_mask * action['transmit_power'] + (1.0 - active_mask) * 0.01)
        total_energy += energy

        # حساب التأخير (Delay): 
        # base_latency = 5ms (تأخير الشبكة الأساسي)
        # routing_penalty = نسبة العقد المعزولة × 100ms (مسارات طويلة وإعادة إرسال)
        has_link = np.any(next_state['connectivity'] > 0, axis=1)
        num_active = max(float(np.sum(active_mask)), 1.0)
        connected_active = float(np.sum(has_link & (active_mask > 0.5)))
        disconnection_ratio = 1.0 - (connected_active / num_active)
        
        delay_noise = 0.0 if deterministic else np.random.uniform(0.0, 2.0)
        delay = 5.0 + disconnection_ratio * 100.0 + delay_noise
            
        total_delay += delay

        # تخزين مباشرة في المصفوفات
        states_pos[step] = state['node_positions']
        states_bat[step] = state['battery_levels']
        states_con[step] = state['connectivity']
        actions_tp[step] = action['transmit_power']
        actions_ss[step] = action['sleep_schedule']
        rewards_arr[step] = reward
        dones_arr[step] = done

        state = next_state
        step += 1

    avg_energy = total_energy / max(step, 1)
    avg_delay = total_delay / max(step, 1)

    return {
        'states': {
            'node_positions': states_pos[:step],
            'battery_levels': states_bat[:step],
            'connectivity': states_con[:step],
        },
        'actions': {
            'transmit_power': actions_tp[:step],
            'sleep_schedule': actions_ss[:step],
        },
        'rewards': rewards_arr[:step],
        'dones': dones_arr[:step],
        'avg_energy': avg_energy,
        'avg_delay': avg_delay,
        'last_connectivity': np.sum(state['connectivity']) * inv_pairs * 100.0,
    }


class FuzzyLogicPolicy:
    """
    Deterministic fuzzy controller for the WSN comparison path.

    The controller maps each node state into fuzzy memberships, applies a small
    Sugeno-style rule base, and then derives:
      1. a relay-importance score,
      2. a sleep-desirability score,
      3. a transmit-power level.
    """

    def __init__(
        self,
        min_awake_ratio=0.68,
        max_sleep_ratio=0.22,
        sleep_threshold=0.58,
        exploration_noise=0.02,
        seed=42,
    ):
        self.min_awake_ratio = min_awake_ratio
        self.max_sleep_ratio = max_sleep_ratio
        self.sleep_threshold = sleep_threshold
        self.exploration_noise = exploration_noise
        self.rng = np.random.RandomState(seed)
        self.power_bias = 0.0
        self.sleep_bias = 0.0
        self.energy_reference = None
        self.delay_reference = None
        self.density_gain = 1.0
        self.degree_gain = 1.0
        self.centrality_gain = 1.0
        self.power_scale = 1.0
        self.sleep_scale = 1.0
        self._baseline_score = None
        self._trial_param = None
        self._trial_delta = 0.0
        self._search_cursor = 0
        self._tuning_params = {
            'min_awake_ratio': {'step': 0.012, 'min_step': 0.004, 'max_step': 0.030, 'min': 0.55, 'max': 0.92, 'direction': 1.0},
            'max_sleep_ratio': {'step': 0.010, 'min_step': 0.004, 'max_step': 0.025, 'min': 0.08, 'max': 0.30, 'direction': -1.0},
            'sleep_threshold': {'step': 0.012, 'min_step': 0.004, 'max_step': 0.030, 'min': 0.42, 'max': 0.78, 'direction': 1.0},
            'power_bias': {'step': 0.018, 'min_step': 0.006, 'max_step': 0.045, 'min': -0.08, 'max': 0.22, 'direction': 1.0},
            'sleep_bias': {'step': 0.018, 'min_step': 0.006, 'max_step': 0.045, 'min': -0.18, 'max': 0.18, 'direction': -1.0},
            'density_gain': {'step': 0.040, 'min_step': 0.015, 'max_step': 0.080, 'min': 0.80, 'max': 1.25, 'direction': 1.0},
            'degree_gain': {'step': 0.040, 'min_step': 0.015, 'max_step': 0.080, 'min': 0.80, 'max': 1.25, 'direction': 1.0},
            'centrality_gain': {'step': 0.040, 'min_step': 0.015, 'max_step': 0.080, 'min': 0.80, 'max': 1.25, 'direction': -1.0},
            'power_scale': {'step': 0.035, 'min_step': 0.015, 'max_step': 0.070, 'min': 0.85, 'max': 1.20, 'direction': 1.0},
            'sleep_scale': {'step': 0.035, 'min_step': 0.015, 'max_step': 0.070, 'min': 0.85, 'max': 1.20, 'direction': -1.0},
        }
        self._tuning_order = list(self._tuning_params.keys())
        self._topology_cache = {}

    @staticmethod
    def _normalize(values):
        values = values.astype(np.float32, copy=False)
        if values.size == 0:
            return values
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmax - vmin < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - vmin) / (vmax - vmin)).astype(np.float32)

    @staticmethod
    def _triangular_vec(x, left, center, right):
        x = np.asarray(x, dtype=np.float32)
        result = np.zeros_like(x, dtype=np.float32)
        left_mask = (x > left) & (x < center)
        if center - left > 1e-8:
            result[left_mask] = (x[left_mask] - left) / (center - left)
        right_mask = (x > center) & (x < right)
        if right - center > 1e-8:
            result[right_mask] = (right - x[right_mask]) / (right - center)
        result[np.isclose(x, center)] = 1.0
        return result

    @staticmethod
    def _trapezoidal_vec(x, a, b, c, d):
        x = np.asarray(x, dtype=np.float32)
        result = np.zeros_like(x, dtype=np.float32)
        plateau = (x >= b) & (x <= c)
        result[plateau] = 1.0
        rise = (x > a) & (x < b)
        if b - a > 1e-8:
            result[rise] = (x[rise] - a) / (b - a)
        fall = (x > c) & (x < d)
        if d - c > 1e-8:
            result[fall] = (d - x[fall]) / (d - c)
        return result

    @staticmethod
    def _triangular(x, left, center, right):
        if x <= left or x >= right:
            return 0.0
        if x == center:
            return 1.0
        if x < center:
            return float((x - left) / max(center - left, 1e-8))
        return float((right - x) / max(right - center, 1e-8))

    @staticmethod
    def _trapezoidal(x, a, b, c, d):
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if x < b:
            return float((x - a) / max(b - a, 1e-8))
        return float((d - x) / max(d - c, 1e-8))

    @staticmethod
    def _sugeno(rules, default_value):
        active = [(strength, output) for strength, output in rules if strength > 1e-6]
        if not active:
            return float(default_value)
        weights = np.array([strength for strength, _ in active], dtype=np.float32)
        outputs = np.array([output for _, output in active], dtype=np.float32)
        return float(np.dot(weights, outputs) / max(np.sum(weights), 1e-6))

    @staticmethod
    def _sugeno_vec(strengths, outputs, default_value):
        strengths = np.asarray(strengths, dtype=np.float32)
        outputs = np.asarray(outputs, dtype=np.float32)[:, np.newaxis]
        weight_sum = np.sum(strengths, axis=0)
        weighted = np.sum(strengths * outputs, axis=0)
        return np.where(weight_sum > 1e-6, weighted / np.maximum(weight_sum, 1e-6), default_value).astype(np.float32)

    @staticmethod
    def _fuzzy_and(*values):
        return float(np.minimum.reduce(np.asarray(values, dtype=np.float32)))

    @staticmethod
    def _fuzzy_and_vec(*values):
        return np.minimum.reduce(np.asarray(values, dtype=np.float32))

    @staticmethod
    def _fuzzy_or(*values):
        return float(np.maximum.reduce(np.asarray(values, dtype=np.float32)))

    @staticmethod
    def _fuzzy_or_vec(*values):
        return np.maximum.reduce(np.asarray(values, dtype=np.float32))

    def _memberships(self, battery, density, degree, centrality):
        battery_low = self._trapezoidal(battery, 0.00, 0.00, 0.20, 0.45)
        battery_medium = self._triangular(battery, 0.25, 0.50, 0.75)
        battery_high = self._trapezoidal(battery, 0.55, 0.75, 1.00, 1.00)

        density_sparse = self._trapezoidal(density, 0.00, 0.00, 0.18, 0.42)
        density_balanced = self._triangular(density, 0.22, 0.50, 0.78)
        density_dense = self._trapezoidal(density, 0.58, 0.78, 1.00, 1.00)

        degree_weak = self._trapezoidal(degree, 0.00, 0.00, 0.18, 0.42)
        degree_good = self._triangular(degree, 0.22, 0.50, 0.78)
        degree_strong = self._trapezoidal(degree, 0.58, 0.78, 1.00, 1.00)

        centrality_edge = self._trapezoidal(centrality, 0.00, 0.00, 0.20, 0.45)
        centrality_mid = self._triangular(centrality, 0.25, 0.50, 0.75)
        centrality_core = self._trapezoidal(centrality, 0.55, 0.75, 1.00, 1.00)

        return {
            'battery_low': battery_low,
            'battery_medium': battery_medium,
            'battery_high': battery_high,
            'density_sparse': density_sparse,
            'density_balanced': density_balanced,
            'density_dense': density_dense,
            'degree_weak': degree_weak,
            'degree_good': degree_good,
            'degree_strong': degree_strong,
            'centrality_edge': centrality_edge,
            'centrality_mid': centrality_mid,
            'centrality_core': centrality_core,
        }

    def _memberships_vec(self, battery, density, degree, centrality):
        return {
            'battery_low': self._trapezoidal_vec(battery, 0.00, 0.00, 0.20, 0.45),
            'battery_medium': self._triangular_vec(battery, 0.25, 0.50, 0.75),
            'battery_high': self._trapezoidal_vec(battery, 0.55, 0.75, 1.00, 1.00),
            'density_sparse': self._trapezoidal_vec(density, 0.00, 0.00, 0.18, 0.42),
            'density_balanced': self._triangular_vec(density, 0.22, 0.50, 0.78),
            'density_dense': self._trapezoidal_vec(density, 0.58, 0.78, 1.00, 1.00),
            'degree_weak': self._trapezoidal_vec(degree, 0.00, 0.00, 0.18, 0.42),
            'degree_good': self._triangular_vec(degree, 0.22, 0.50, 0.78),
            'degree_strong': self._trapezoidal_vec(degree, 0.58, 0.78, 1.00, 1.00),
            'centrality_edge': self._trapezoidal_vec(centrality, 0.00, 0.00, 0.20, 0.45),
            'centrality_mid': self._triangular_vec(centrality, 0.25, 0.50, 0.75),
            'centrality_core': self._trapezoidal_vec(centrality, 0.55, 0.75, 1.00, 1.00),
        }

    def _evaluate_nodes(self, battery, density, degree, centrality):
        density = np.clip(np.asarray(density, dtype=np.float32) * self.density_gain, 0.0, 1.0)
        degree = np.clip(np.asarray(degree, dtype=np.float32) * self.degree_gain, 0.0, 1.0)
        centrality = np.clip(np.asarray(centrality, dtype=np.float32) * self.centrality_gain, 0.0, 1.0)
        battery = np.asarray(battery, dtype=np.float32)

        m = self._memberships_vec(battery, density, degree, centrality)

        relay_importance = self._sugeno_vec(np.stack([
            self._fuzzy_and_vec(m['battery_high'], m['degree_strong']),
            self._fuzzy_and_vec(m['battery_high'], m['centrality_core']),
            self._fuzzy_and_vec(m['degree_good'], m['centrality_core']),
            self._fuzzy_and_vec(m['density_sparse'], m['battery_high']),
            self._fuzzy_and_vec(m['density_sparse'], m['degree_good']),
            self._fuzzy_and_vec(m['battery_medium'], m['degree_good']),
            self._fuzzy_and_vec(m['density_dense'], m['centrality_edge']),
            self._fuzzy_and_vec(m['battery_low'], m['degree_weak']),
        ], axis=0), np.array([0.98, 0.90, 0.82, 0.86, 0.78, 0.62, 0.20, 0.10], dtype=np.float32), 0.48)

        sleep_desirability = self._sugeno_vec(np.stack([
            self._fuzzy_and_vec(m['battery_low'], m['density_dense'], m['degree_strong']),
            self._fuzzy_and_vec(m['battery_low'], m['centrality_edge']),
            self._fuzzy_and_vec(m['density_dense'], m['centrality_edge']),
            self._fuzzy_and_vec(m['battery_medium'], m['density_dense'], m['degree_good']),
            self._fuzzy_and_vec(m['battery_low'], m['density_balanced']),
            self._fuzzy_and_vec(m['battery_high'], m['density_sparse']),
            self._fuzzy_or_vec(m['centrality_core'], m['degree_weak']),
            self._fuzzy_and_vec(m['battery_high'], m['centrality_core']),
        ], axis=0), np.array([0.96, 0.88, 0.82, 0.72, 0.74, 0.05, 0.08, 0.03], dtype=np.float32), 0.40)

        transmit_power = self._sugeno_vec(np.stack([
            self._fuzzy_and_vec(m['degree_weak'], m['density_sparse']),
            self._fuzzy_and_vec(m['degree_weak'], m['battery_high']),
            self._fuzzy_and_vec(m['degree_weak'], m['battery_medium']),
            self._fuzzy_and_vec(m['centrality_edge'], m['density_sparse']),
            self._fuzzy_and_vec(m['centrality_core'], m['degree_good']),
            self._fuzzy_and_vec(m['battery_low'], m['degree_strong']),
            self._fuzzy_and_vec(m['degree_strong'], m['density_dense']),
            self._fuzzy_and_vec(m['battery_medium'], m['degree_good']),
            self._fuzzy_and_vec(m['battery_high'], m['density_dense']),
        ], axis=0), np.array([1.00, 0.92, 0.84, 0.88, 0.58, 0.42, 0.36, 0.62, 0.52], dtype=np.float32), 0.64)

        sleep_desirability = np.clip(sleep_desirability * self.sleep_scale + self.sleep_bias, 0.0, 1.0)
        transmit_power = np.clip(transmit_power * self.power_scale + self.power_bias, 0.35, 1.0)

        return relay_importance.astype(np.float32), sleep_desirability.astype(np.float32), transmit_power.astype(np.float32)

    def _get_topology_features(self, positions):
        topo_key = positions.astype(np.float32, copy=False).tobytes()
        cached = self._topology_cache.get(topo_key)
        if cached is not None:
            return cached

        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_full = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)
        np.fill_diagonal(dist_full, np.inf)
        cached = {'dist_full': dist_full}
        self._topology_cache = {topo_key: cached}
        return cached

    def _evaluate_node(self, battery, density, degree, centrality):
        density = float(np.clip(density * self.density_gain, 0.0, 1.0))
        degree = float(np.clip(degree * self.degree_gain, 0.0, 1.0))
        centrality = float(np.clip(centrality * self.centrality_gain, 0.0, 1.0))
        m = self._memberships(battery, density, degree, centrality)

        relay_importance = self._sugeno([
            (self._fuzzy_and(m['battery_high'], m['degree_strong']), 0.98),
            (self._fuzzy_and(m['battery_high'], m['centrality_core']), 0.90),
            (self._fuzzy_and(m['degree_good'], m['centrality_core']), 0.82),
            (self._fuzzy_and(m['density_sparse'], m['battery_high']), 0.86),
            (self._fuzzy_and(m['density_sparse'], m['degree_good']), 0.78),
            (self._fuzzy_and(m['battery_medium'], m['degree_good']), 0.62),
            (self._fuzzy_and(m['density_dense'], m['centrality_edge']), 0.20),
            (self._fuzzy_and(m['battery_low'], m['degree_weak']), 0.10),
        ], default_value=0.48)

        sleep_desirability = self._sugeno([
            (self._fuzzy_and(m['battery_low'], m['density_dense'], m['degree_strong']), 0.96),
            (self._fuzzy_and(m['battery_low'], m['centrality_edge']), 0.88),
            (self._fuzzy_and(m['density_dense'], m['centrality_edge']), 0.82),
            (self._fuzzy_and(m['battery_medium'], m['density_dense'], m['degree_good']), 0.72),
            (self._fuzzy_and(m['battery_low'], m['density_balanced']), 0.74),
            (self._fuzzy_and(m['battery_high'], m['density_sparse']), 0.05),
            (self._fuzzy_or(m['centrality_core'], m['degree_weak']), 0.08),
            (self._fuzzy_and(m['battery_high'], m['centrality_core']), 0.03),
        ], default_value=0.40)

        transmit_power = self._sugeno([
            (self._fuzzy_and(m['degree_weak'], m['density_sparse']), 1.00),
            (self._fuzzy_and(m['degree_weak'], m['battery_high']), 0.92),
            (self._fuzzy_and(m['degree_weak'], m['battery_medium']), 0.84),
            (self._fuzzy_and(m['centrality_edge'], m['density_sparse']), 0.88),
            (self._fuzzy_and(m['centrality_core'], m['degree_good']), 0.58),
            (self._fuzzy_and(m['battery_low'], m['degree_strong']), 0.42),
            (self._fuzzy_and(m['degree_strong'], m['density_dense']), 0.36),
            (self._fuzzy_and(m['battery_medium'], m['degree_good']), 0.62),
            (self._fuzzy_and(m['battery_high'], m['density_dense']), 0.52),
        ], default_value=0.64)

        sleep_desirability = float(np.clip(sleep_desirability * self.sleep_scale + self.sleep_bias, 0.0, 1.0))
        transmit_power = float(np.clip(transmit_power * self.power_scale + self.power_bias, 0.35, 1.0))

        return relay_importance, sleep_desirability, transmit_power

    def _round_score(self, avg_energy, avg_delay, connectivity_pct, sleep_ratio, avg_reward):
        if self.energy_reference is None:
            self.energy_reference = float(avg_energy)
        if self.delay_reference is None:
            self.delay_reference = float(avg_delay)

        connectivity = float(connectivity_pct) / 100.0
        sleep_ratio = float(np.clip(sleep_ratio, 0.0, 1.0))
        energy_gain = float(np.clip(self.energy_reference / max(float(avg_energy), 1e-8), 0.6, 1.6))
        delay_gain = float(np.clip(self.delay_reference / max(float(avg_delay), 1e-8), 0.6, 1.6))
        sleep_target = 0.18
        sleep_balance = max(0.0, 1.0 - abs(sleep_ratio - sleep_target) / sleep_target)

        return (
            0.58 * float(avg_reward) +
            0.16 * connectivity +
            0.10 * (energy_gain / 1.6) +
            0.10 * (delay_gain / 1.6) +
            0.06 * sleep_balance
        )

    def _apply_tuning_delta(self, name, delta):
        spec = self._tuning_params[name]
        current = float(getattr(self, name))
        updated = float(np.clip(current + delta, spec['min'], spec['max']))
        setattr(self, name, updated)
        return updated - current

    def _schedule_next_trial(self):
        for _ in range(len(self._tuning_order)):
            name = self._tuning_order[self._search_cursor]
            self._search_cursor = (self._search_cursor + 1) % len(self._tuning_order)
            spec = self._tuning_params[name]
            desired_delta = spec['direction'] * spec['step']
            actual_delta = self._apply_tuning_delta(name, desired_delta)
            if abs(actual_delta) > 1e-6:
                self._trial_param = name
                self._trial_delta = actual_delta
                return
            spec['direction'] *= -1.0

        self._trial_param = None
        self._trial_delta = 0.0

    def adapt_controller(self, avg_energy, avg_delay, connectivity_pct, sleep_ratio, avg_reward):
        score = self._round_score(avg_energy, avg_delay, connectivity_pct, sleep_ratio, avg_reward)

        if self._baseline_score is None:
            self._baseline_score = score
            self._schedule_next_trial()
            return

        if self._trial_param is not None:
            spec = self._tuning_params[self._trial_param]
            if score >= self._baseline_score + 1e-4:
                self._baseline_score = score
                spec['step'] = min(spec['step'] * 1.04, spec['max_step'])
            elif score <= self._baseline_score - 1e-4:
                spec['direction'] *= -1.0
                spec['step'] = max(spec['step'] * 0.92, spec['min_step'])
                self._baseline_score = score
            else:
                spec['step'] = min(spec['step'] * 1.01, spec['max_step'])
                self._baseline_score = score

        self._schedule_next_trial()

    def get_action(self, state, deterministic=True):
        positions = state['node_positions']
        batteries = state['battery_levels'].astype(np.float32, copy=False)
        connectivity = state['connectivity']
        num_nodes = batteries.shape[0]

        transmit_power = np.zeros(num_nodes, dtype=np.float32)
        sleep_schedule = np.ones(num_nodes, dtype=np.float32)

        alive_mask = batteries > 0
        alive_indices = np.flatnonzero(alive_mask)
        alive_count = int(alive_indices.size)

        if alive_count == 0:
            return {'transmit_power': transmit_power, 'sleep_schedule': sleep_schedule}

        alive_positions = positions[alive_mask]
        alive_batteries = batteries[alive_mask]
        topo_features = self._get_topology_features(positions)

        if alive_count > 1:
            dist = topo_features['dist_full'][np.ix_(alive_mask, alive_mask)]

            k = min(4, alive_count - 1)
            nearest = np.partition(dist, kth=k - 1, axis=1)[:, :k]
            mean_nearest = nearest.mean(axis=1)
            density = 1.0 - self._normalize(mean_nearest)

            centroid = np.mean(alive_positions, axis=0)
            dist_to_centroid = np.sqrt(np.sum((alive_positions - centroid) ** 2, axis=1))
            centrality = 1.0 - self._normalize(dist_to_centroid)

            alive_connectivity = connectivity[np.ix_(alive_mask, alive_mask)]
            degree = alive_connectivity.sum(axis=1).astype(np.float32)
            degree_norm = degree / max(float(np.max(degree)), 1.0)
        else:
            density = np.ones(1, dtype=np.float32)
            centrality = np.ones(1, dtype=np.float32)
            degree_norm = np.ones(1, dtype=np.float32)

        relay_scores, sleep_scores, power_levels = self._evaluate_nodes(
            battery=alive_batteries,
            density=density,
            degree=degree_norm,
            centrality=centrality,
        )

        min_awake = min(alive_count, max(3, int(np.ceil(self.min_awake_ratio * alive_count))))
        max_sleep_budget = min(
            max(0, alive_count - min_awake),
            int(np.floor(self.max_sleep_ratio * alive_count)),
        )

        sleep_alive = np.zeros(alive_count, dtype=np.float32)
        if max_sleep_budget > 0:
            sleep_margin = sleep_scores - relay_scores
            candidate_order = np.argsort(-sleep_margin)
            slept = 0

            for idx in candidate_order:
                if slept >= max_sleep_budget:
                    break
                if sleep_scores[idx] < self.sleep_threshold:
                    continue
                if sleep_margin[idx] < 0.12:
                    continue
                if relay_scores[idx] > 0.55:
                    continue
                if degree_norm[idx] < 0.22 and density[idx] < 0.35:
                    continue
                sleep_alive[idx] = 1.0
                slept += 1

        if not deterministic:
            power_levels = np.clip(
                power_levels + self.rng.uniform(-self.exploration_noise, self.exploration_noise, size=alive_count),
                0.35,
                1.0,
            )

        power_levels[sleep_alive > 0.5] = 0.0

        transmit_power[alive_indices] = power_levels.astype(np.float32, copy=False)
        sleep_schedule[alive_indices] = sleep_alive

        return {
            'transmit_power': transmit_power,
            'sleep_schedule': sleep_schedule,
        }


def run_fuzzy_logic_comparison(args, env_config, eval_tasks):
    print("=== Running Fuzzy Logic Comparison ===", flush=True)
    fuzzy_policy = FuzzyLogicPolicy(min_awake_ratio=0.68, max_sleep_ratio=0.22, seed=42)

    for meta_iter in range(args.meta_iterations):
        force_death_pct = read_force_death_pct()

        _eval_energies, _eval_delays = [], []
        _eval_rollouts = []

        for _idx, _et in enumerate(eval_tasks):
            eval_init = dict(_et)
            if force_death_pct > 0:
                batteries = _et['batteries'].copy()
                n_kill = int(len(batteries) * force_death_pct / 100)
                if n_kill > 0:
                    _rng = np.random.RandomState(42 + _idx)
                    kill_idx = _rng.choice(len(batteries), size=n_kill, replace=False)
                    batteries[kill_idx] = 0.0
                eval_init['batteries'] = batteries

            _er = collect_rollout(
                _et['env'],
                fuzzy_policy,
                env_config['max_steps'],
                deterministic=True,
                fixed_init=eval_init
            )
            _eval_energies.append(_er['avg_energy'])
            _eval_delays.append(_er['avg_delay'])
            _eval_rollouts.append(_er)

        eval_energy = float(np.mean(_eval_energies))
        eval_delay = float(np.mean(_eval_delays))
        avg_reward = float(np.mean([np.mean(_rollout['rewards']) for _rollout in _eval_rollouts]))

        last_rollout = _eval_rollouts[0]
        last_bat = last_rollout['states']['battery_levels'][-1]
        last_con = last_rollout['states']['connectivity'][-1]
        last_pos = last_rollout['states']['node_positions'][-1]
        all_ss = last_rollout['actions']['sleep_schedule']
        last_ss_final = all_ss[-1]
        awake_mask = (last_ss_final < 0.5)
        num_awake_eval = np.sum(awake_mask)
        has_link_mask = np.any(last_con > 0, axis=1)

        if num_awake_eval > 0:
            connected_awake = np.sum(has_link_mask & awake_mask)
            connectivity = float((connected_awake / num_awake_eval) * 100.0)
        else:
            connectivity = 0.0

        progress = ((meta_iter + 1) / args.meta_iterations) * 100
        print(
            f"METRICS_FUZZY|{meta_iter + 1}|{eval_energy:.6f}|{eval_delay:.2f}|{progress:.1f}|{connectivity:.1f}",
            flush=True
        )

        avg_sleep_per_node = np.mean(all_ss >= 0.5, axis=0)
        avg_sleep_ratio = float(np.mean(all_ss >= 0.5))
        n_dead = int(np.sum(last_bat <= 0))
        n_sleep = int(np.round(np.sum((avg_sleep_per_node >= 0.5) & (last_bat > 0))))
        n_awake = int(env_config['num_nodes']) - n_dead - n_sleep
        n_links = int(np.sum(last_con) // 2)

        pos_str = ','.join(f"{x:.3f},{y:.3f}" for x, y in last_pos)
        state_str = ''.join(
            'D' if last_bat[i] <= 0 else ('S' if last_ss_final[i] >= 0.5 else 'A')
            for i in range(env_config['num_nodes'])
        )
        print(f"WSN_STATE|{n_awake}|{n_sleep}|{n_dead}|{n_links}|{pos_str}|{state_str}", flush=True)

        if (meta_iter + 1) % 10 == 0:
            print(
                f"Round {meta_iter + 1} [Fuzzy Logic]: Reward={avg_reward:.3f}, Energy={eval_energy:.6f}, Delay={eval_delay:.2f}ms",
                flush=True
            )
            print(f"REWARD|{avg_reward:.3f}", flush=True)

        fuzzy_policy.adapt_controller(
            avg_energy=eval_energy,
            avg_delay=eval_delay,
            connectivity_pct=connectivity,
            sleep_ratio=avg_sleep_ratio,
            avg_reward=avg_reward,
        )

        time.sleep(0.05)

    print("=" * 60, flush=True)
    print("=== Fuzzy Logic comparison finished ===", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--comm_range', type=float, default=0.15)
    parser.add_argument('--energy_consumption', type=float, default=0.05)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--meta_iterations', type=int, default=1000)
    parser.add_argument('--meta_batch', type=int, default=5)
    parser.add_argument('--adaptation_steps', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--fuzzy', action='store_true', default=False)
    parser.add_argument('--fuzzy_after_training', action='store_true', default=False)
    args = parser.parse_args()

    if not args.resume:
        # عند البدء من الصفر نستخدم seed ثابت للتكرارية
        torch.manual_seed(42)
        np.random.seed(42)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.", flush=True)
        device = 'cpu'

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA initialized: Using {gpu_name}", flush=True)

    env_config = {
        'num_nodes': args.num_nodes,
        'comm_range': args.comm_range,
        'energy_consumption': args.energy_consumption,
        'max_steps': args.max_steps
    }

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Starting training with {args.num_nodes} nodes on DEVICE: {device.upper()}", flush=True)
    print(f"Meta iterations: {args.meta_iterations}, Batch size: {args.meta_batch}", flush=True)
    print("=" * 60, flush=True)

    sample_env = WSNEnv(env_config)
    obs_space = sample_env.observation_space
    act_space = sample_env.action_space

    state_dim = (
        obs_space['node_positions'].shape[0] * obs_space['node_positions'].shape[1] +
        obs_space['battery_levels'].shape[0] +
        (obs_space['connectivity'].shape[0] * (obs_space['connectivity'].shape[0] - 1)) // 2
    )

    action_dims = {
        'transmit_power': act_space['transmit_power'].shape[0],
        'sleep_schedule': act_space['sleep_schedule'].n
    }

    policy = WSNActorCritic(state_dim, action_dims, device=device).to(device)

    agent = MAMLAgent(
        policy_network=policy,
        inner_lr=3e-3,
        meta_lr=1e-4,
        num_updates=args.adaptation_steps,
        device=device
    )

    # --- مهام تقييم ثابتة (نفس الطبولوجيا كل جولة = منحنى تعلم حقيقي بلا ضوضاء) ---
    _rng_state = np.random.get_state()
    np.random.seed(2025)
    _eval_tasks = []
    for _ in range(3):
        _et = WSNEnv(env_config)
        _et.reset()
        _eval_tasks.append({
            'env':      _et,
            'positions': _et.node_positions.copy(),
            'batteries': np.ones(env_config['num_nodes'], dtype=np.float32),
        })
    np.random.set_state(_rng_state)
    # -----------------------------------------------------------------------

    if args.fuzzy:
        run_fuzzy_logic_comparison(args, env_config, _eval_tasks)
        return

    best_avg_reward = -float('inf')
    start_iter = 0

    if args.resume:
        ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        if os.path.exists(ckpt_path):
            start_iter, best_avg_reward = agent.load(ckpt_path)
            print(f"Resumed from checkpoint: {ckpt_path} (iteration {start_iter}, best reward {best_avg_reward:.3f})", flush=True)
        else:
            print(f"No checkpoint found at {ckpt_path}, starting fresh.", flush=True)

    # ضبط وضع policy للتدريب مرة واحدة قبل الحلقة
    policy.train()

    for meta_iter in range(start_iter, args.meta_iterations):
        # قراءة نسبة الموت الإجباري من الواجهة (قابلة للتغيير أثناء التدريب)
        force_death_pct = read_force_death_pct()

        tasks = create_tasks(args.meta_batch, env_config)

        # تطبيق الموت الإجباري على كل مهمة
        if force_death_pct > 0:
            for task in tasks:
                n_kill = int(task.num_nodes * force_death_pct / 100)
                if n_kill > 0:
                    kill_indices = np.random.choice(task.num_nodes, size=n_kill, replace=False)
                    task.battery_levels[kill_indices] = 0.0
                    task.update_connectivity()

        task_data = []
        total_energy = 0.0
        total_delay = 0.0

        for task in tasks:
            # جمع بيانات أولية بالـ base policy
            support_rollout = collect_rollout(task, policy, env_config['max_steps'], deterministic=False)
            # تكيف السياسة على هذه المهمة (adaptation_steps يأثر هنا)
            adapted_policy = agent.adapt(support_rollout, num_steps=args.adaptation_steps)
            # جمع بيانات بالسياسة المتكيفة
            rollout = collect_rollout(task, adapted_policy, env_config['max_steps'])
            task_data.append(rollout)
            total_energy += rollout['avg_energy']
            total_delay += rollout['avg_delay']

        avg_energy = total_energy / args.meta_batch
        avg_delay = total_delay / args.meta_batch

        meta_loss = agent.meta_update(task_data)

        # --- تقييم على مهام ثابتة (بلا تكيف، حتمي) للحصول على منحنى تعلم نظيف ---
        policy.eval()
        _eval_energies, _eval_delays = [], []
        _eval_rollouts = []
        for _idx, _et in enumerate(_eval_tasks):
            # تطبيق الموت الإجباري على مهام التقييم أيضاً
            eval_init = dict(_et)  # نسخة سطحية
            if force_death_pct > 0:
                batteries = _et['batteries'].copy()
                n_kill = int(len(batteries) * force_death_pct / 100)
                if n_kill > 0:
                    # seed ثابت لكل مهمة تقييم حتى يكون الموت متسقاً خلال نفس الجولة
                    _rng = np.random.RandomState(42 + _idx)
                    kill_idx = _rng.choice(len(batteries), size=n_kill, replace=False)
                    batteries[kill_idx] = 0.0
                eval_init['batteries'] = batteries
            _er = collect_rollout(_et['env'], policy, env_config['max_steps'],
                                  deterministic=True, fixed_init=eval_init)
            _eval_energies.append(_er['avg_energy'])
            _eval_delays.append(_er['avg_delay'])
            _eval_rollouts.append(_er)
        policy.train()
        eval_energy = float(np.mean(_eval_energies))
        eval_delay  = float(np.mean(_eval_delays))
        # -----------------------------------------------------------------------

        # مقياس الاتصال: نسبة العقد المستيقظة التي لديها رابط واحد على الأقل (من مهمة التقييم الثابتة)
        last_rollout = _eval_rollouts[0]
        last_bat = last_rollout['states']['battery_levels'][-1]   # (N,)
        last_con = last_rollout['states']['connectivity'][-1]     # (N,N)
        last_pos = last_rollout['states']['node_positions'][-1]   # (N,2)
        all_ss   = last_rollout['actions']['sleep_schedule']       # (T, N)

        last_ss_final = all_ss[-1]
        awake_mask = (last_ss_final < 0.5)
        num_awake_eval = np.sum(awake_mask)
        
        has_link_mask = np.any(last_con > 0, axis=1)
        if num_awake_eval > 0:
            connected_awake = np.sum(has_link_mask & awake_mask)
            connectivity = float((connected_awake / num_awake_eval) * 100.0)
        else:
            connectivity = 0.0

        progress = ((meta_iter + 1) / args.meta_iterations) * 100
        # تقرير مقاييس التقييم الثابت للرسوم البيانية (ينعدم ضوضاء الطبولوجيا)
        print(f"METRICS|{meta_iter + 1}|{eval_energy:.6f}|{eval_delay:.2f}|{progress:.1f}|{connectivity:.1f}", flush=True)

        # إرسال حالة الشبكة للواجهة (awake/sleep/dead/links)
        # sleep_schedule هي float (0.0 أو 1.0)، نستخدم >= 0.5 للتأكد
        # متوسط حالة النوم عبر كل خطوات الـ episode لكل عقدة
        avg_sleep_per_node = np.mean(all_ss >= 0.5, axis=0)       # (N,) بين 0 و1
        n_dead  = int(np.sum(last_bat <= 0))
        n_sleep = int(np.round(np.sum((avg_sleep_per_node >= 0.5) & (last_bat > 0))))
        n_awake = int(env_config['num_nodes']) - n_dead - n_sleep
        n_links = int(np.sum(last_con) // 2)
        # حالة كل عقدة للرسم: نستخدم آخر خطوة للعرض المكاني
        last_ss_final = all_ss[-1]
        pos_str = ','.join(f"{x:.3f},{y:.3f}" for x, y in last_pos)
        state_str = ''.join(
            'D' if last_bat[i] <= 0 else ('S' if last_ss_final[i] >= 0.5 else 'A')
            for i in range(env_config['num_nodes'])
        )
        print(f"WSN_STATE|{n_awake}|{n_sleep}|{n_dead}|{n_links}|{pos_str}|{state_str}", flush=True)

        if (meta_iter + 1) % 10 == 0:
            # استخدام بيانات rollout الموجودة بدلاً من تشغيل eval إضافي
            avg_reward = float(np.mean(task_data[-1]['rewards']))

            print(f"Round {meta_iter + 1}: Loss={meta_loss:.6f}, Reward={avg_reward:.3f}, Energy={avg_energy:.6f}, Delay={avg_delay:.2f}ms", flush=True)
            print(f"REWARD|{avg_reward:.3f}", flush=True)

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                save_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                agent.save(save_path, meta_iter=meta_iter + 1, best_avg_reward=best_avg_reward)
                print(f"New best model saved! Reward: {avg_reward:.3f}", flush=True)

    print("=" * 60, flush=True)
    print("=== Training finished ===", flush=True)

    if args.fuzzy_after_training:
        run_fuzzy_logic_comparison(args, env_config, _eval_tasks)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

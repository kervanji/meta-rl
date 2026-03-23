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


class RuleBasedPolicy:
    """
    Heuristic baseline that uses battery, local density, and node centrality.

    It is deterministic during evaluation so the comparison against Meta-RL is
    reproducible and not driven by baseline randomness.
    """

    def __init__(self, target_sleep_ratio=0.10, min_awake_ratio=0.70, exploration_noise=0.03, seed=42):
        self.target_sleep_ratio = target_sleep_ratio
        self.min_awake_ratio = min_awake_ratio
        self.exploration_noise = exploration_noise
        self.rng = np.random.RandomState(seed)

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

    def get_action(self, state, deterministic=True):
        positions = state['node_positions']
        batteries = state['battery_levels'].astype(np.float32, copy=False)
        connectivity = state['connectivity']
        N = batteries.shape[0]

        tp = np.zeros(N, dtype=np.float32)
        ss = np.ones(N, dtype=np.float32)

        alive_mask = batteries > 0
        alive_indices = np.flatnonzero(alive_mask)
        alive_count = int(alive_indices.size)

        if alive_count == 0:
            return {'transmit_power': tp, 'sleep_schedule': ss}

        alive_positions = positions[alive_mask]
        alive_batteries = batteries[alive_mask]

        # تقدير كثافة الجوار هندسياً حتى لا يعتمد baseline على العشوائية.
        if alive_count > 1:
            diff = alive_positions[:, np.newaxis, :] - alive_positions[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dist, np.inf)

            k = min(3, alive_count - 1)
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

        # عقد ذات بطارية جيدة وكثافة/مركزية أعلى تُفضَّل للبقاء مستيقظة كعقد relay.
        relay_score = (
            0.45 * alive_batteries +
            0.20 * degree_norm +
            0.20 * density +
            0.15 * centrality
        )
        # العقد منخفضة البطارية وعالية التكرار المروري مرشحة للنوم.
        sleep_score = (
            0.55 * (1.0 - alive_batteries) +
            0.30 * density +
            0.15 * (1.0 - centrality)
        )

        min_awake = min(alive_count, max(3, int(np.ceil(self.min_awake_ratio * alive_count))))
        target_sleep = min(
            max(0, alive_count - min_awake),
            int(np.floor(self.target_sleep_ratio * alive_count))
        )

        ss_alive = np.zeros(alive_count, dtype=np.float32)
        if target_sleep > 0:
            relay_cutoff = np.quantile(relay_score, 0.70) if alive_count > 1 else relay_score[0]
            top_relay_cutoff = np.quantile(relay_score, 0.85) if alive_count > 1 else relay_score[0]
            candidate_order = np.argsort(-sleep_score)
            slept = 0

            for idx in candidate_order:
                if slept >= target_sleep:
                    break
                if relay_score[idx] >= relay_cutoff or degree_norm[idx] < 0.35:
                    continue
                ss_alive[idx] = 1.0
                slept += 1

            if slept < target_sleep:
                for idx in candidate_order:
                    if slept >= target_sleep:
                        break
                    if ss_alive[idx] > 0.5 or relay_score[idx] >= top_relay_cutoff:
                        continue
                    ss_alive[idx] = 1.0
                    slept += 1

        tp_alive = (
            0.55 +
            0.25 * (1.0 - density) +
            0.15 * (1.0 - degree_norm) +
            0.05 * centrality
        )
        tp_alive -= np.where(alive_batteries < 0.30, 0.10, 0.0).astype(np.float32)
        tp_alive += np.where((degree_norm < 0.25) | (centrality < 0.35), 0.10, 0.0).astype(np.float32)
        tp_alive = np.clip(tp_alive, 0.35, 1.0)

        if not deterministic:
            tp_alive = np.clip(
                tp_alive + self.rng.uniform(-self.exploration_noise, self.exploration_noise, size=alive_count),
                0.35,
                1.0
            )

        tp_alive[ss_alive > 0.5] = 0.0

        tp[alive_indices] = tp_alive.astype(np.float32, copy=False)
        ss[alive_indices] = ss_alive

        return {
            'transmit_power': tp,
            'sleep_schedule': ss
        }

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
    parser.add_argument('--baseline', action='store_true', default=False)
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

    if args.baseline:
        print("=== Running Heuristic Baseline ===", flush=True)
        baseline_policy = RuleBasedPolicy(target_sleep_ratio=0.10, min_awake_ratio=0.70, seed=42)
        for meta_iter in range(args.meta_iterations):
            force_death_pct = read_force_death_pct()
            
            _eval_energies, _eval_delays = [], []
            _eval_rollouts = []
            
            for _idx, _et in enumerate(_eval_tasks):
                eval_init = dict(_et)
                if force_death_pct > 0:
                    batteries = _et['batteries'].copy()
                    n_kill = int(len(batteries) * force_death_pct / 100)
                    if n_kill > 0:
                        _rng = np.random.RandomState(42 + _idx)
                        kill_idx = _rng.choice(len(batteries), size=n_kill, replace=False)
                        batteries[kill_idx] = 0.0
                    eval_init['batteries'] = batteries
                
                _er = collect_rollout(_et['env'], baseline_policy, env_config['max_steps'],
                                      deterministic=True, fixed_init=eval_init)
                _eval_energies.append(_er['avg_energy'])
                _eval_delays.append(_er['avg_delay'])
                _eval_rollouts.append(_er)
                
            eval_energy = float(np.mean(_eval_energies))
            eval_delay  = float(np.mean(_eval_delays))
            
            last_rollout = _eval_rollouts[0]
            last_bat = last_rollout['states']['battery_levels'][-1]
            last_con = last_rollout['states']['connectivity'][-1]
            last_pos = last_rollout['states']['node_positions'][-1]
            all_ss   = last_rollout['actions']['sleep_schedule']
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
            print(f"METRICS_BASE|{meta_iter + 1}|{eval_energy:.6f}|{eval_delay:.2f}|{progress:.1f}|{connectivity:.1f}", flush=True)
            
            avg_sleep_per_node = np.mean(all_ss >= 0.5, axis=0)
            n_dead  = int(np.sum(last_bat <= 0))
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
                avg_reward = float(np.mean(last_rollout['rewards']))
                print(f"Round {meta_iter + 1} [Heuristic Baseline]: Reward={avg_reward:.3f}, Energy={eval_energy:.6f}, Delay={eval_delay:.2f}ms", flush=True)
                print(f"REWARD|{avg_reward:.3f}", flush=True)
            
            time.sleep(0.05)
            
        print("=" * 60, flush=True)
        print("=== Heuristic baseline finished ===", flush=True)
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

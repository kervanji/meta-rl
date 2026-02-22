import os
import sys
import argparse
import traceback

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


def collect_rollout(env, policy, max_steps=None, deterministic=True):
    if max_steps is None:
        max_steps = 200

    # تخصيص مصفوفات مسبقاً بدلاً من قوائم Python (أسرع بكثير)
    N = env.num_nodes
    states_pos = np.empty((max_steps, N, 2), dtype=np.float32)
    states_bat = np.empty((max_steps, N), dtype=np.float32)
    states_con = np.empty((max_steps, N, N), dtype=np.float32)
    actions_tp = np.empty((max_steps, N), dtype=np.float32)
    actions_ss = np.empty((max_steps, N), dtype=np.float32)
    rewards_arr = np.empty(max_steps, dtype=np.float32)
    dones_arr = np.empty(max_steps, dtype=bool)

    state = env.reset()
    done = False
    step = 0
    total_energy = 0.0
    total_delay = 0.0
    inv_pairs = 1.0 / (N * (N - 1))
    range_energy = env.energy_consumption * env.comm_range

    while not done and step < max_steps:
        action = policy.get_action(state, deterministic=deterministic)
        next_state, reward, done, _ = env.step(action)

        # حساب الطاقة والتأخير
        active_mask = (1.0 - action['sleep_schedule'])
        energy = range_energy * np.mean(active_mask * action['transmit_power'] + (1.0 - active_mask) * 0.01)
        total_energy += energy

        connectivity = np.sum(next_state['connectivity']) * inv_pairs
        total_delay += (1.0 - connectivity) * 100.0 + np.random.uniform(0, 3)

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
        inner_lr=1e-3,
        meta_lr=1e-4,
        num_updates=args.adaptation_steps,
        device=device
    )

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
        tasks = create_tasks(args.meta_batch, env_config)

        task_data = []
        total_energy = 0.0
        total_delay = 0.0
        last_connectivity = 0.0

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
            last_connectivity = rollout['last_connectivity']  # استخدام البيانات الموجودة

        avg_energy = total_energy / args.meta_batch
        avg_delay = total_delay / args.meta_batch

        meta_loss = agent.meta_update(task_data)

        # لا حاجة لإنشاء eval_env جديد - استخدام آخر نتيجة من task_data
        connectivity = last_connectivity

        progress = ((meta_iter + 1) / args.meta_iterations) * 100
        print(f"METRICS|{meta_iter + 1}|{avg_energy:.6f}|{avg_delay:.2f}|{progress:.1f}|{connectivity:.1f}", flush=True)

        # إرسال حالة الشبكة للواجهة (awake/sleep/dead/links)
        last_rollout = task_data[-1]
        # البطارية والمواقع من آخر خطوة (لإظهار الحالة النهائية)
        last_bat = last_rollout['states']['battery_levels'][-1]   # (N,)
        last_con = last_rollout['states']['connectivity'][-1]     # (N,N)
        last_pos = last_rollout['states']['node_positions'][-1]   # (N,2)
        # sleep_schedule هي float (0.0 أو 1.0)، نستخدم >= 0.5 للتأكد
        all_ss = last_rollout['actions']['sleep_schedule']        # (T, N)
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

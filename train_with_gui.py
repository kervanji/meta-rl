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


def collect_rollout(env, policy, max_steps=None):
    if max_steps is None:
        max_steps = float('inf')

    states, actions, rewards, dones = [], [], [], []
    state = env.reset()
    done = False
    step = 0
    total_energy = 0
    total_delay = 0

    while not done and step < max_steps:
        action = policy.get_action(state)
        next_state, reward, done, _ = env.step(action)

        active_mask = (1 - action['sleep_schedule']).astype(np.float32)
        sleep_mask = 1.0 - active_mask
        idle_power = 0.01  # 1% of base consumption for sleeping nodes
        energy = env.energy_consumption * np.mean(
            active_mask * action['transmit_power'] + sleep_mask * idle_power
        )
        total_energy += energy

        connectivity = np.sum(next_state['connectivity']) / (env.num_nodes * (env.num_nodes - 1))
        delay = (1 - connectivity) * 100 + np.random.uniform(0, 3)
        total_delay += delay

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        state = next_state
        step += 1

    avg_energy = total_energy / max(step, 1)
    avg_delay = total_delay / max(step, 1)

    states_stacked = {
        k: np.array([s[k] for s in states], dtype=np.float32)
        for k in states[0].keys()
    }
    return {
        'states': states_stacked,
        'actions': {k: np.array([a[k] for a in actions], dtype=np.float32) for k in actions[0].keys()},
        'rewards': np.array(rewards, dtype=np.float32),
        'dones': np.array(dones, dtype=bool),
        'avg_energy': avg_energy,
        'avg_delay': avg_delay
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
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    env_config = {
        'num_nodes': args.num_nodes,
        'comm_range': args.comm_range,
        'energy_consumption': args.energy_consumption,
        'max_steps': args.max_steps
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Starting training with {args.num_nodes} nodes on {device}", flush=True)
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

    policy = WSNActorCritic(state_dim, action_dims).to(device)

    agent = MAMLAgent(
        policy_network=policy,
        inner_lr=1e-3,
        meta_lr=1e-4,
        num_updates=args.adaptation_steps,
        device=device
    )

    best_avg_reward = -float('inf')

    if args.resume:
        ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        if os.path.exists(ckpt_path):
            agent.load(ckpt_path)
            print(f"Resumed from checkpoint: {ckpt_path}", flush=True)
        else:
            print(f"No checkpoint found at {ckpt_path}, starting fresh.", flush=True)

    for meta_iter in range(args.meta_iterations):
        tasks = create_tasks(args.meta_batch, env_config)

        task_data = []
        total_energy = 0
        total_delay = 0

        for task in tasks:
            rollout = collect_rollout(task, policy, env_config['max_steps'])
            task_data.append(rollout)
            total_energy += rollout['avg_energy']
            total_delay += rollout['avg_delay']

        avg_energy = total_energy / args.meta_batch
        avg_delay = total_delay / args.meta_batch

        meta_loss = agent.meta_update(task_data)

        eval_env = WSNEnv(env_config)
        eval_state = eval_env.reset()
        connectivity = np.sum(eval_state['connectivity']) / (args.num_nodes * (args.num_nodes - 1)) * 100

        progress = ((meta_iter + 1) / args.meta_iterations) * 100
        print(f"METRICS|{meta_iter + 1}|{avg_energy:.6f}|{avg_delay:.2f}|{progress:.1f}|{connectivity:.1f}", flush=True)

        if (meta_iter + 1) % 10 == 0:
            eval_rollout = collect_rollout(eval_env, policy, env_config['max_steps'])
            avg_reward = np.mean(eval_rollout['rewards'])

            print(f"Round {meta_iter + 1}: Loss={meta_loss:.6f}, Reward={avg_reward:.3f}, Energy={avg_energy:.6f}, Delay={avg_delay:.2f}ms", flush=True)
            print(f"REWARD|{avg_reward:.3f}", flush=True)

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                save_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                agent.save(save_path)
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


import os
import sys
import traceback

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

print("Initializing training script...", flush=True)

try:
    import torch
    import numpy as np
    import time
    print("Libraries loaded successfully", flush=True)

    # Add parent directory to path
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
        
        # Calculate energy consumption (based on transmit power AND sleep schedule)
        active_mask = 1 - action['sleep_schedule']  # 1 if active, 0 if sleeping
        energy = env.energy_consumption * np.mean(active_mask * action['transmit_power'])
        total_energy += energy
        
        # Calculate delay (simulated based on network connectivity and distance)
        connectivity = np.sum(next_state['connectivity']) / (env.num_nodes * (env.num_nodes - 1))
        delay = (1 - connectivity) * 100 + np.random.uniform(0, 3)  # ms (reduced noise)
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
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration from GUI
    env_config = {
        'num_nodes': 100,
        'comm_range': 0.15,
        'energy_consumption': 0.05,
        'max_steps': 100
    }
    
    num_meta_iterations = 1000
    meta_batch_size = 5
    num_adaptation_steps = 5
    save_dir = 'checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training with {env_config['num_nodes']} nodes on {device}", flush=True)
    print(f"Meta iterations: {num_meta_iterations}, Batch size: {meta_batch_size}", flush=True)
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
        num_updates=num_adaptation_steps,
        device=device
    )
    
    best_avg_reward = -float('inf')
    
    for meta_iter in range(num_meta_iterations):
        tasks = create_tasks(meta_batch_size, env_config)
        
        task_data = []
        total_energy = 0
        total_delay = 0
        
        for task in tasks:
            rollout = collect_rollout(task, policy, env_config['max_steps'])
            task_data.append(rollout)
            total_energy += rollout['avg_energy']
            total_delay += rollout['avg_delay']
        
        avg_energy = total_energy / meta_batch_size
        avg_delay = total_delay / meta_batch_size
        
        meta_loss = agent.meta_update(task_data)
        
        # Calculate connectivity
        eval_env = WSNEnv(env_config)
        eval_state = eval_env.reset()
        connectivity = np.sum(eval_state['connectivity']) / (env_config['num_nodes'] * (env_config['num_nodes'] - 1)) * 100
        
        # Print metrics for GUI parsing
        progress = ((meta_iter + 1) / num_meta_iterations) * 100
        print(f"METRICS|{meta_iter + 1}|{avg_energy:.6f}|{avg_delay:.2f}|{progress:.1f}|{connectivity:.1f}", flush=True)
        
        if (meta_iter + 1) % 10 == 0:
            eval_rollout = collect_rollout(eval_env, policy, env_config['max_steps'])
            avg_reward = np.mean(eval_rollout['rewards'])
            
            print(f"Round {meta_iter + 1}: Loss={meta_loss:.6f}, Reward={avg_reward:.3f}, Energy={avg_energy:.6f}, Delay={avg_delay:.2f}ms", flush=True)
            print(f"REWARD|{avg_reward:.3f}", flush=True)
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                save_path = os.path.join(save_dir, 'best_model.pt')
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


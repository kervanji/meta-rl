import os
import torch
import numpy as np

import gymnasium as gym
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.envs.wsn_env import WSNEnv
from src.networks.wsn_policy import WSNActorCritic
from src.agents.maml_agent import MAMLAgent

# ==========================================
# 1. إعداد المهام (Task Creation)
# هنا نحدد تنوع المهام التي سيتعلم منها العميل
# ------------------------------------------
# أمثلة للتغيير:
# - 'comm_range': np.random.uniform(0.1, 0.5) -> جعل المهام أكثر تنوعاً وصعوبة.
# - 'num_nodes': np.random.randint(5, 15) -> جعل العميل يتعلم التعامل مع أحجام شبكات مختلفة.
# ==========================================
def create_tasks(num_tasks: int, env_config: Dict = None) -> List[gym.Env]:
    if env_config is None:
        env_config = {
            'num_nodes': 20,
            'comm_range': 0.5,
            'energy_consumption': 0.2,
            'max_steps': 50
        }
    
    tasks = []
    for _ in range(num_tasks):
        config = env_config.copy()
        config['comm_range'] = env_config['comm_range']   # يأخذ قيمة الـGUI
        
        env = WSNEnv(config)
        tasks.append(env)
    
    return tasks

def collect_rollout(env: gym.Env, policy: WSNActorCritic, max_steps: int = None) -> Dict[str, np.ndarray]:
    if max_steps is None:
        max_steps = float('inf')
    
    states, actions, rewards, dones = [], [], [], []
    state = env.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        action = policy.get_action(state)
        
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        state = next_state
        step += 1
    
    states_stacked = {
        k: np.array([s[k] for s in states], dtype=np.float32)
        for k in states[0].keys()
    }
    return {
        'states': states_stacked,
        'actions': {k: np.array([a[k] for a in actions], dtype=np.float32) for k in actions[0].keys()},
        'rewards': np.array(rewards, dtype=np.float32),
        'dones': np.array(dones, dtype=bool)
    }

# ==========================================
# 2. حلقة التدريب الشامل (Meta-Training Loop)
# المعاملات التي تتحكم في سرعة وجودة التعلم
# ------------------------------------------
# أمثلة للتغيير:
# - num_meta_iterations: 2000 -> تدريب أطول للحصول على دقة أعلى.
# - meta_batch_size: 10 -> رؤية مهام أكثر في كل خطوة (استقرار أكثر).
# - num_adaptation_steps: 10 -> زيادة "التفكير" أو التكيف مع المهام الجديدة.
# ==========================================
def train_meta_rl(
    num_meta_iterations: int = 1000,
    meta_batch_size: int = 5,
    num_adaptation_steps: int = 5,
    save_dir: str = 'checkpoints',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    os.makedirs(save_dir, exist_ok=True)
    
    sample_env = WSNEnv()
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
    
    for meta_iter in tqdm(range(num_meta_iterations), desc="Meta-iteration"):
        tasks = create_tasks(meta_batch_size)
        
        task_data = []
        for task in tasks:
            rollout = collect_rollout(task, policy)
            task_data.append(rollout)
        
        meta_loss = agent.meta_update(task_data)
        
        if (meta_iter + 1) % 10 == 0:
            eval_env = WSNEnv()
            eval_rollout = collect_rollout(eval_env, policy)
            avg_reward = np.mean(eval_rollout['rewards'])
            
            print(f"Meta-iteration {meta_iter + 1}:")
            print(f"  Meta-loss: {meta_loss:.4f}")
            print(f"  Eval reward: {avg_reward:.2f}")
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                save_path = os.path.join(save_dir, f'best_model.pt')
                agent.save(save_path)
                print(f"  New best model saved to {save_path}")

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_meta_rl(
        num_meta_iterations=1000,
        meta_batch_size=5,
        num_adaptation_steps=5,
        save_dir='checkpoints',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

if __name__ == "__main__":
    main()

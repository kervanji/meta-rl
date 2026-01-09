import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

from src.envs.wsn_env import WSNEnv
from src.networks.wsn_policy import WSNActorCritic
from src.agents.maml_agent import MAMLAgent


# ==========================================
# 1. إنشاء السياسة (Create Policy)
# تقوم هذه الدالة بتهيئة عميل MAML وتحميل هيكل الشبكة العصبية
# ==========================================
def create_policy(device: str = None, env_config: dict = None) -> MAMLAgent:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sample_env = WSNEnv(env_config) if env_config is not None else WSNEnv()
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
        num_updates=5,
        device=device,
    )

    return agent


# ==========================================
# 2. تقييم العميل (Evaluate Agent)
# تقوم هذه الدالة باختبار النموذج المدرب على بيئة معينة وحساب النتائج
# ==========================================
def evaluate_agent(
    checkpoint_path: str = 'checkpoints/best_model.pt',
    num_episodes: int = 10,
    device: str = None,
    env_config: dict = None,
) -> np.ndarray:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_nodes_cfg = None
    if env_config is not None and 'num_nodes' in env_config:
        num_nodes_cfg = env_config['num_nodes']

    load_checkpoint = (num_nodes_cfg is None) or (num_nodes_cfg == 10)

    agent = create_policy(device=device, env_config=env_config)

    if load_checkpoint:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        agent.load(checkpoint_path)
    else:
        print(
            "تحذير: عدد الحساسات (num_nodes) مختلف عن العدد الذي درِّب عليه النموذج "
            "(10). سيتم استخدام أوزان عشوائية غير مدرَّبة لهذا التكوين."
        )
    policy = agent.policy

    rewards_per_episode = []

    for ep in range(num_episodes):
        env = WSNEnv(env_config) if env_config is not None else WSNEnv()
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = policy.get_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state

        rewards_per_episode.append(ep_reward)
        print(f"Episode {ep + 1}/{num_episodes} reward: {ep_reward:.3f}")

    rewards_per_episode = np.array(rewards_per_episode, dtype=np.float32)
    avg_reward = float(np.mean(rewards_per_episode))
    std_reward = float(np.std(rewards_per_episode))

    print("\nEvaluation finished")
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.3f} ± {std_reward:.3f}")

    raw_title = 'عوائد الحلقات'
    raw_xlabel = 'رقم الحلقة'
    raw_ylabel = 'العائد الكلي'

    title_text = get_display(arabic_reshaper.reshape(raw_title))
    xlabel_text = get_display(arabic_reshaper.reshape(raw_xlabel))
    ylabel_text = get_display(arabic_reshaper.reshape(raw_ylabel))

    # Plot rewards curve (Arabic labels)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_episodes + 1), rewards_per_episode, marker='o')
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    plt.title(title_text)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return rewards_per_episode


# ==========================================
# 3. اختبار سيناريوهات متعددة (Evaluate Multiple Scenarios)
# هنا نقوم باختبار النموذج في ظروف مختلفة (مدى اتصال مختلف)
# لنرى مدى قدرته على التكيف التي تعلمها في MAML
# ==========================================
def evaluate_multiple_envs(
    checkpoint_path: str = 'checkpoints/best_model.pt',
    num_episodes: int = 10,
    device: str = None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = [
        {
            'name': 'baseline',
            'config': {
                'num_nodes': 10,
                'comm_range': 0.3,
                'energy_consumption': 0.05,
                'max_steps': 100,
            },
        },
        {
            'name': 'smaller_range',
            'config': {
                'num_nodes': 10,
                'comm_range': 0.2,
                'energy_consumption': 0.05,
                'max_steps': 100,
            },
        },
        {
            'name': 'larger_range',
            'config': {
                'num_nodes': 10,
                'comm_range': 0.5,
                'energy_consumption': 0.05,
                'max_steps': 100,
            },
        },
    ]

    print("Evaluating on multiple WSN configurations:\n")

    for cfg in configs:
        name = cfg['name']
        env_cfg = cfg['config']
        print(f"=== Config: {name} ===")
        rewards = evaluate_agent(
            checkpoint_path=checkpoint_path,
            num_episodes=num_episodes,
            device=device,
            env_config=env_cfg,
        )
        avg = float(np.mean(rewards))
        std = float(np.std(rewards))
        print(f"Config {name}: average reward = {avg:.3f} ± {std:.3f}\n")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    evaluate_multiple_envs(device=device)


if __name__ == "__main__":
    main()

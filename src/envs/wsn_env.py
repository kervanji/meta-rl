import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class WSNAbstractEnv(gym.Env):
    """فئة أساسية مجردة لبيئات شبكات الحساسات اللاسلكية (WSN)."""
    
    # ==========================================
    # 1. إعدادات البيئة (Environment Configuration)
    # ==========================================
    def __init__(self, config: dict):
        super(WSNAbstractEnv, self).__init__()
        
        # إعدادات البيئة
        self.config = config
        self.num_nodes = config.get('num_nodes', 20)
        self.comm_range = config.get('comm_range', 0.5)
        self.energy_consumption = config.get('energy_consumption', 0.2)
        self.max_steps = config.get('max_steps', 50)
        
        # ==========================================
        # 2. فضاء الملاحظة (Observation Space)
        # يمثل "ماذا يرى" الذكاء الاصطناعي في كل لحظة
        # ==========================================
        self.observation_space = spaces.Dict({
            'node_positions': spaces.Box(low=0, high=1, shape=(self.num_nodes, 2)), # مواقع العقد (x,y) في خريطة [0,1]
            'battery_levels': spaces.Box(low=0, high=1, shape=(self.num_nodes,)), # مستوى البطارية: 0 (فارغ) إلى 1 (مشحون كامل)
            'connectivity': spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_nodes)) # مصفوفة الاتصال: 0 (لا يوجد اتصال) إلى 1 (اتصال قوي)
        })
        
        # ==========================================
        # 3. فضاء الإجراءات (Action Space)
        # يمثل "أدوات التحكم" التي يمتلكها الذكاء الاصطناعي
        # ==========================================
        self.action_space = spaces.Dict({
            'transmit_power': spaces.Box(low=0, high=1, shape=(self.num_nodes,)), # قوة الإرسال: 0 (أقل قوة) إلى 1 (أعلى قوة)
            'sleep_schedule': spaces.MultiBinary(self.num_nodes) # جدول النوم: 0 (مستيقظ/نشط)، 1 (نائم/خامل)
        })
        
        # تهيئة حالة الشبكة
        self.reset()
    
    def reset(self) -> Dict:
        """إعادة تعيين البيئة إلى الحالة الأولية."""
        # مواقع عقد عشوائية في منطقة [0,1]x[0,1]
        self.node_positions = np.random.rand(self.num_nodes, 2)
        
        # تهيئة مستويات البطارية (موحدة بين [0,1])
        self.battery_levels = np.ones(self.num_nodes)
        self.prev_transmit_power = np.zeros(self.num_nodes)
        
        # تهيئة اتصال الشبكة
        self.update_connectivity()
        
        self.current_step = 0
        return self._get_observation()
    
    def update_connectivity(self) -> None:
        """تحديث اتصال الشبكة الثابت بناءً على مواقع العقد ومدى الاتصال الكامل (يُستخدم عند reset)."""
        diff = self.node_positions[:, np.newaxis, :] - self.node_positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        self.connectivity = (dist <= self.comm_range).astype(np.float32)
        np.fill_diagonal(self.connectivity, 0)

    def _update_dynamic_connectivity(self, transmit_power: np.ndarray, sleep_schedule: np.ndarray) -> None:
        """تحديث اتصال الشبكة ديناميكياً بناءً على قوة الإرسال وجدول النوم."""
        diff = self.node_positions[:, np.newaxis, :] - self.node_positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))

        effective_range = self.comm_range * transmit_power
        min_range = np.minimum(effective_range[:, np.newaxis], effective_range[np.newaxis, :])

        awake = (sleep_schedule == 0).astype(np.float32)
        awake_pair = awake[:, np.newaxis] * awake[np.newaxis, :]

        self.connectivity = ((dist <= min_range) & (awake_pair > 0)).astype(np.float32)
        np.fill_diagonal(self.connectivity, 0)
    
    def _get_observation(self) -> Dict:
        """الحصول على الملاحظة الحالية للبيئة."""
        return {
            'node_positions': self.node_positions.copy(),
            'battery_levels': self.battery_levels.copy(),
            'connectivity': self.connectivity.copy()
        }
    
    # ==========================================
    # 4. منطق الخطوة الزمنية (Step Logic)
    # يحدد كيف تتغير البيئة بناءً على قرارات العميل
    # ==========================================
    def step(self, action: Dict) -> Tuple[Dict, float, bool, dict]:
        """
        اتخاذ خطوة في البيئة.
        
        المعاملات:
            action: قاموس يحتوي على إجراءات 'transmit_power' و 'sleep_schedule'
            
        المخرجات:
            observation: ملاحظة الحالة الجديدة
            reward: المكافأة المحسوبة
            done: ما إذا كانت الحلقة قد انتهت
            info: معلومات إضافية
        """
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}
            
        # تطبيق الإجراءات
        transmit_power = action['transmit_power']
        sleep_schedule = action['sleep_schedule']

        # تحديث الاتصال الديناميكي بناءً على الإجراءات الجديدة
        self._update_dynamic_connectivity(transmit_power, sleep_schedule)

        # تحديث مستويات البطارية (نموذج واقعي)
        awake_mask = (sleep_schedule == 0).astype(np.float32)
        sleep_mask = 1.0 - awake_mask
        idle_power = 0.01  # 1% من الاستهلاك الأساسي للعقد النائمة
        self.battery_levels -= self.energy_consumption * (
            awake_mask * transmit_power + sleep_mask * idle_power
        )
        
        # حساب المكافأة
        reward = self._calculate_reward(transmit_power, sleep_schedule)
        self.prev_transmit_power = transmit_power.copy()
        
        # تحديث عداد الخطوات
        self.current_step += 1
        
        # التحقق مما إذا كانت الحلقة قد انتهت
        done = self.current_step >= self.max_steps or np.any(self.battery_levels <= 0)
        
        return self._get_observation(), reward, done, {}
    
    # ==========================================
    # 5. حساب المكافأة (Reward Calculation)
    # يحدد "شخصية" العميل وما الذي نكافئه عليه
    # ------------------------------------------
    # مثال للتغيير:
    # - لزيادة الاهتمام بالبطارية: اجعل وزن energy_efficiency = 0.8
    # - لزيادة الاهتمام بالاتصال: اجعل وزن connectivity_score = 0.8
    # ==========================================
    def _calculate_reward(self, transmit_power: np.ndarray, sleep_schedule: np.ndarray) -> float:
        """حساب المكافأة بناءً على الحالة الحالية والإجراءات المتخذة."""
        # ---------------------------------------------------------
        # مكونات المكافأة:
        # 1. مكافأة الاتصال (وزن 0.5): نسبة العقد المستيقظة المتصلة + عقوبة على العقد المعزولة.
        # 2. مكافأة الطاقة (وزن 0.4): مكافأة النوم وتخفيض قوة البث.
        # 3. عقوبة موت العقد (تناسبية): تُخصم بنسبة العقد الميتة.
        # ---------------------------------------------------------

        awake_mask = (sleep_schedule == 0)  # True للعقد المستيقظة
        num_awake = int(np.sum(awake_mask))

        # --- مكافأة الاتصال ---
        if num_awake == 0:
            connectivity_reward = -1.0  # عقوبة قصوى إذا كانت كل الشبكة نائمة
        else:
            has_link = np.any(self.connectivity > 0, axis=1)  # (N,) bool
            connected_awake = int(np.sum(has_link & awake_mask))
            isolated_awake = num_awake - connected_awake

            connectivity_ratio = connected_awake / num_awake
            isolation_penalty = 2.0 * (isolated_awake / self.num_nodes)
            connectivity_reward = connectivity_ratio - isolation_penalty

        # --- مكافأة الطاقة ---
        sleep_ratio = np.mean(sleep_schedule)          # نسبة العقد النائمة
        power_saving = 1.0 - np.mean(transmit_power)   # توفير قوة البث
        energy_reward = 0.6 * sleep_ratio + 0.4 * power_saving

        # --- المكافأة الإجمالية الموزونة ---
        reward = (
            0.5 * connectivity_reward +
            0.4 * energy_reward
        )

        # --- عقوبة تناسبية لنفاد البطارية ---
        dead_nodes = np.sum(self.battery_levels <= 0)
        if dead_nodes > 0:
            reward -= 5.0 * (dead_nodes / self.num_nodes)

        return reward
    
    # ==========================================
    # 6. العرض المرئي (Rendering)
    # لرسم خريطة الشبكة وتتبع الحالات بصرياً
    # ==========================================
    def render(self, mode: str = 'human', sleep_schedule: np.ndarray = None,
               transmit_power: np.ndarray = None) -> None:
        """عرض الحالة الحالية للبيئة.
        
        ألوان العقد:
          - أزرق  : مستيقظة ونشطة
          - رمادي : نائمة (sleep_schedule == 1)
          - أحمر  : ميتة (battery <= 0)
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')

        # رسم الاتصالات أولاً (تحت العقد)
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if self.connectivity[i, j] > 0:
                    lw = 1.5
                    if transmit_power is not None:
                        lw = 0.5 + 2.5 * float(min(transmit_power[i], transmit_power[j]))
                    ax.plot(
                        [self.node_positions[i, 0], self.node_positions[j, 0]],
                        [self.node_positions[i, 1], self.node_positions[j, 1]],
                        color='#4fc3f7', alpha=0.4, linewidth=lw
                    )

        # تصنيف العقد حسب الحالة
        awake_idx, sleep_idx, dead_idx = [], [], []
        for i in range(self.num_nodes):
            if self.battery_levels[i] <= 0:
                dead_idx.append(i)
            elif sleep_schedule is not None and sleep_schedule[i] == 1:
                sleep_idx.append(i)
            else:
                awake_idx.append(i)

        def _scatter(indices, color, label, marker='o', size=120, alpha=1.0, zorder=3):
            if indices:
                idx = np.array(indices)
                ax.scatter(self.node_positions[idx, 0], self.node_positions[idx, 1],
                           c=color, s=size, label=label, marker=marker,
                           alpha=alpha, zorder=zorder, edgecolors='white', linewidths=0.5)

        _scatter(awake_idx,  '#29b6f6', f'Awake ({len(awake_idx)})')
        _scatter(sleep_idx,  '#78909c', f'Sleeping ({len(sleep_idx)})', alpha=0.6)
        _scatter(dead_idx,   '#ef5350', f'Dead ({len(dead_idx)})',   marker='x', size=140,
                 alpha=1.0, zorder=4)

        # عنوان يعرض إحصائيات سريعة
        connected = int(np.sum(self.connectivity) / 2)
        title = (f'WSN Topology  |  Awake: {len(awake_idx)}  '
                 f'Sleeping: {len(sleep_idx)}  Dead: {len(dead_idx)}  '
                 f'Links: {connected}')
        ax.set_title(title, color='white', fontsize=10, pad=10)
        ax.set_xlabel('X Position', color='#aaaaaa')
        ax.set_ylabel('Y Position', color='#aaaaaa')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_color('#333355')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        legend = ax.legend(loc='upper right', fontsize=9,
                           facecolor='#16213e', edgecolor='#333355', labelcolor='white')

        plt.tight_layout()
        if mode == 'human':
            plt.show()
        else:
            return fig


# ==========================================
# 7. التنفيذ الفعلي (Concrete Implementation)
# هنا نضع القيم الافتراضية التي يبدأ بها البرنامج
# ------------------------------------------
# أمثلة للتغيير والـتأثير:
# - 'num_nodes': 20 -> تزداد صعوبة الإدارة بسبب كثرة العقد.
# - 'comm_range': 0.1 -> يصبح الاتصال صعباً جداً لأن العقد يجب أن تكون قريبة جداً.
# - 'energy_consumption': 0.2 -> تموت العقد بسرعة أكبر، مما يجبر العميل على تعلم "النوم" بسرعة.
# ==========================================
class WSNEnv(WSNAbstractEnv):
    """تنفيذ فعلي لبيئة شبكات الحساسات اللاسلكية."""
    
    def __init__(self, config: dict = None):
        if config is None:
            config = {
                'num_nodes': 20, #عدد الحساسات
                'comm_range': 0.5,#مدى الاتصال بين العقد
                'energy_consumption': 0.02,#معدل استهلاك الطاقة
                'max_steps': 50 #عدد الخطوات
            }
        super().__init__(config)

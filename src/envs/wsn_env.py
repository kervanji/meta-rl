import numpy as np
import networkx as nx
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
        
        # تهيئة اتصال الشبكة
        self.update_connectivity()
        
        self.current_step = 0
        return self._get_observation()
    
    def update_connectivity(self) -> None:
        """تحديث اتصال الشبكة بناءً على مواقع العقد ومدى الاتصال."""
        self.connectivity = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                distance = np.linalg.norm(self.node_positions[i] - self.node_positions[j])
                if distance <= self.comm_range:
                    self.connectivity[i,j] = 1
                    self.connectivity[j,i] = 1
    
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
        
        # تحديث مستويات البطارية (نموذج مبسط)
        for i in range(self.num_nodes):
            if sleep_schedule[i] == 0:  # العقدة نشطة
                self.battery_levels[i] -= self.energy_consumption * transmit_power[i]
        
        # حساب المكافأة (مبسط)
        reward = self._calculate_reward(transmit_power, sleep_schedule)
        
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
        # شرح مكونات المكافأة (Reward Components):
        # 1. جودة الاتصال: كلما كانت العقد متصلة ببعضها زادت المكافأة.
        # 2. كفاءة الطاقة: كلما استخدمنا قوة إرسال أقل زادت المكافأة.
        # 3. كفاءة النوم: نكافئ العميل عندما ينجح في تنويم العقد غير الضرورية.
        # ---------------------------------------------------------
        connectivity_score = np.sum(self.connectivity) / (self.num_nodes * (self.num_nodes - 1)) #جودة الاتصال
        energy_efficiency = 1.0 - np.mean(transmit_power) #كفاءة الطاقة
        battery_health = np.mean(self.battery_levels) # حماية البطارية
        
        # حساب المكافأة الأساسية الموزونة (بدون مكافأة النوم المباشرة)
        reward = (
            0.5 * connectivity_score + # وزن جودة الاتصال
            0.3 * energy_efficiency +  # وزن توفير الطاقة
            0.2 * battery_health       # وزن حماية البطارية
        )

        # ==========================================
        # عقوبة تناسبية لنفاد البطارية (Proportional Penalty):
        # بدلاً من عقوبة ثابتة -5، نخصم بنسبة عدد العقد الميتة
        # ==========================================
        dead_nodes = np.sum(self.battery_levels <= 0)
        if dead_nodes > 0:
            reward -= 5.0 * (dead_nodes / self.num_nodes)  # عقوبة تناسبية
            
        return reward
    
    # ==========================================
    # 6. العرض المرئي (Rendering)
    # لرسم خريطة الشبكة وتتبع الحالات بصرياً
    # ==========================================
    def render(self, mode: str = 'human') -> None:
        """عرض الحالة الحالية للبيئة."""
        plt.figure(figsize=(8, 8))
        
        # رسم العقد
        plt.scatter(self.node_positions[:, 0], self.node_positions[:, 1], 
                   c='blue', s=100, label='Nodes')
        
        # رسم الاتصالات
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if self.connectivity[i,j] > 0:
                    plt.plot([self.node_positions[i,0], self.node_positions[j,0]],
                            [self.node_positions[i,1], self.node_positions[j,1]],
                            'gray', alpha=0.5)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('WSN Topology')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        
        if mode == 'human':
            plt.show()
        else:
            return plt.gcf()


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

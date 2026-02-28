import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import copy

class MAMLAgent:
    """
    عميل التعلم التلوي المستقل عن النموذج (MAML) للتحكم في شبكات الحساسات اللاسلكية (WSN).
    """
    
    # ==========================================
    # 1. إعدادات خوارزمية التعلم (Learning Algorithm Settings)
    # هنا نتحكم في كيفية "تأقلم" العميل مع المهام الجديدة
    # ------------------------------------------
    # أمثلة للتغيير:
    # - inner_lr=0.01: تسريع التعلم الداخلي (التكيف السريع).
    # - meta_lr=0.0005: تغيير سرعة التعلم العام للنموذج.
    # - num_updates: زيادة عدد المحاولات التي يقوم بها العميل لفهم المهمة الجديدة.
    # ==========================================
    def __init__(
        self,
        policy_network: nn.Module,
        inner_lr: float = 1e-3,
        meta_lr: float = 1e-3,
        num_updates: int = 1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        تهيئة عميل MAML.
        
        المعاملات:
            policy_network: الشبكة العصبية التي تمثل السياسة (Policy)
            inner_lr: معدل التعلم لتحديثات الحلقة الداخلية
            meta_lr: معدل التعلم للتحديثات الشاملة (Meta-updates)
            num_updates: عدد تحديثات الحلقة الداخلية
            device: الجهاز المستخدم للحسابات (CPU أو GPU)
        """
        self.policy = policy_network.to(device)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_updates = num_updates
        self.device = device
        
        # إنشاء المحسن الشامل (Meta-optimizer)
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=meta_lr)
        
        # دالة الخسارة
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def _compute_returns(rewards_np: np.ndarray, gamma: float = 0.99) -> np.ndarray:
        """حساب العوائد المخصومة G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …"""
        T = len(rewards_np)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = float(rewards_np[t]) + gamma * G
            returns[t] = G
        return returns

    def _reward_weighted_loss(
        self,
        preds: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        rewards_np: np.ndarray,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """
        Advantage-Weighted Regression (AWR) مع عوائد مخصومة:
        استخدام G_t بدلاً من r_t يمنح الخطوات الأولى إشارةً أقوى
        لأنها تعكس جودة الإجراء على المدى البعيد وليس الفوري فقط.
        """
        returns_np = self._compute_returns(rewards_np, gamma)
        returns_t = torch.from_numpy(returns_np).to(self.device)
        std = returns_t.std()
        adv = (returns_t - returns_t.mean()) / (std + 1e-8)
        # أوزان موجبة ومركزها 1 — high-return steps get w > 1
        weights = torch.clamp(adv + 1.0, min=0.1).detach()
        weights = weights / weights.mean()

        loss = None
        for name in actions.keys():
            # خسارة MSE لكل خطوة زمنية → شكل (T,)
            per_step = (preds[name] - actions[name]).pow(2).mean(dim=-1)
            w_loss = (per_step * weights).mean()
            loss = w_loss if loss is None else loss + w_loss
        return loss if loss is not None else torch.zeros(1, device=self.device)
        
    def adapt(self, task_data: Dict[str, object], num_steps: int = None) -> nn.Module:
        """
        إجراء التكيف مع مهمة جديدة.
        
        المعاملات:
            task_data: قاموس يحتوي على بيانات الحالة، الإجراء، والمكافأة
            num_steps: عدد خطوات التكيف (يتجاوز self.num_updates إذا تم توفيره)
            
        المخرجات:
            شبكة السياسة بعد التكيف
        """
        if num_steps is None:
            num_steps = self.num_updates
            
        # إنشاء نسخة من السياسة لعملية التكيف
        adapted_policy = copy.deepcopy(self.policy)
        adapted_optimizer = optim.SGD(adapted_policy.parameters(), lr=self.inner_lr)
        
        # تحويل بيانات numpy إلى tensors على الجهاز المستخدم
        states_np: Dict[str, np.ndarray] = task_data['states']
        actions_np: Dict[str, np.ndarray] = task_data['actions']

        states = {k: torch.from_numpy(v).float().to(self.device) for k, v in states_np.items()}
        actions = {k: torch.from_numpy(v).float().to(self.device) for k, v in actions_np.items()}
        
        rewards_np: np.ndarray = task_data['rewards']

        # التكيف في الحلقة الداخلية (Inner loop) — AWR-weighted loss
        for _ in range(num_steps):
            # Forward pass
            action_preds, _ = adapted_policy(states)

            # خسارة موزونة بالمكافأة بدلاً من MSE بسيط
            loss = self._reward_weighted_loss(action_preds, actions, rewards_np)

            # تحديث السياسة المتكيفة
            adapted_optimizer.zero_grad()
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_policy
    
    def meta_update(
        self,
        meta_batch: List[Dict[str, object]],
        adaptation_steps: int = None
    ) -> float:
        """
        إجراء تحديث شامل (Meta-update) باستخدام دفعة من المهام.
        
        المعاملات:
            meta_batch: قائمة تحتوي على قواميس بيانات المهام
            adaptation_steps: عدد خطوات التكيف لكل مهمة
            
        المخرجات:
            متوسط الخسارة الشاملة عبر المهام
        """
        if adaptation_steps is None:
            adaptation_steps = self.num_updates
            
        total_meta_loss = 0.0

        # مسح التدرجات السابقة (Gradients)
        self.meta_optimizer.zero_grad()

        # معالجة كل مهمة في الدفعة وتحديث التدرجات في السياسة الأساسية
        for task_data in meta_batch:
            states_np: Dict[str, np.ndarray] = task_data['states']
            actions_np: Dict[str, np.ndarray] = task_data['actions']
            rewards_np: np.ndarray = task_data['rewards']

            states = {k: torch.from_numpy(v).float().to(self.device) for k, v in states_np.items()}
            actions = {k: torch.from_numpy(v).float().to(self.device) for k, v in actions_np.items()}

            # التمرير الأمامي عبر السياسة الأساسية
            preds, _ = self.policy(states)

            # خسارة AWR موزونة بالمكافأة — يتعلم العميل تكرار الأفعال ذات المكافأة العالية
            loss = self._reward_weighted_loss(preds, actions, rewards_np)

            total_meta_loss += loss.item()

            # الانتشار العكسي لهذه المهمة (Backpropagation)
            loss.backward()

        # حساب متوسط الخسارة الشاملة
        avg_meta_loss = total_meta_loss / len(meta_batch)

        # قص التدرجات لمنع الانفجار (Gradient clipping)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

        # تنفيذ خطوة المحسن (Optimizer step)
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()

        return avg_meta_loss
    
    def get_action(self, state: Dict[str, torch.Tensor], deterministic: bool = True) -> Dict[str, torch.Tensor]:
        """
        الحصول على الإجراء من السياسة.
        
        المعاملات:
            state: ملاحظة الحالة الحالية
            deterministic: ما إذا كان يجب إرجاع إجراءات محددة (غير عشوائية)
            
        المخرجات:
            قاموس يحتوي على الإجراءات
        """
        self.policy.eval()
        with torch.no_grad():
            # تحويل الحالة إلى tensor إذا لزم الأمر
            if not isinstance(state, torch.Tensor):
                state = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in state.items()}
            
            # الحصول على الإجراء من السياسة
            action_dict = self.policy(state)
            
            # تحويل المخرجات إلى numpy إذا لزم الأمر
            if isinstance(action_dict, dict):
                action_dict = {k: v.squeeze(0).cpu().numpy() for k, v in action_dict.items()}
            else:
                action_dict = action_dict.squeeze(0).cpu().numpy()
                
            return action_dict
    
    def save(self, path: str, meta_iter: int = 0, best_avg_reward: float = -float('inf')) -> None:
        """حفظ نموذج العميل."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'meta_iter': meta_iter,
            'best_avg_reward': best_avg_reward,
        }, path)
    
    def load(self, path: str):
        """تحميل نموذج العميل. يُرجع (start_iter, best_avg_reward) للاستئناف الصحيح."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('meta_iter', 0)
        best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
        return start_iter, best_avg_reward

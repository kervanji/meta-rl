import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class WSNActorCritic(nn.Module):
    """
    شبكة Actor-Critic للتحكم في شبكات الحساسات اللاسلكية (WSN).
    يقوم الـ Actor بإخراج إجراءات قوة الإرسال وجدول النوم.
    يقوم الـ Critic بتقدير دالة القيمة (Value function).
    """
    
    # ==========================================
    # 1. بنية الشبكة العصبية (Neural Network Architecture)
    # يمكنك هنا تغيير "ذكاء" العميل وقدرته على الاستيعاب
    # ------------------------------------------
    # أمثلة للتغيير:
    # - hidden_dims=[256, 128, 64]: زيادة عدد الطبقات والخلايا يجعل الشبكة أذكى ولكن أبطأ.
    # - استخدام nn.ReLU() أو nn.Tanh(): تغيير دالة التنشيط يؤثر على سرعة استقرار التعلم.
    # ==========================================
    def __init__(
        self,
        state_dim: int,
        action_dims: Dict[str, int],
        hidden_dims: List[int] = [256, 128, 64],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        تهيئة شبكة Actor-Critic.
        
        المعاملات:
            state_dim: أبعاد فضاء الحالة
            action_dims: قاموس يحتوي على أبعاد كل مكون من مكونات الإجراء
            hidden_dims: قائمة بأبعاد الطبقات الخفية
            device: الجهاز المستخدم للحسابات
        """
        super(WSNActorCritic, self).__init__()
        self.device = device
        self._num_nodes = None  # سيُحدَّد عند أول forward
        self._triu_indices = None  # مؤشرات المثلث العلوي (تُحسب مرة واحدة)
        
        # مستخرج سمات مشترك (Shared feature extractor)
        self.shared_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            self.shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # رؤوس الـ Actor (رأس لكل مكون من مكونات الإجراء)
        self.actor_heads = nn.ModuleDict()
        for name, dim in action_dims.items():
            if name == 'sleep_schedule':  # مخرجات ثنائية (0 أو 1)
                self.actor_heads[name] = nn.Sequential(
                    nn.Linear(prev_dim, dim),
                    nn.Sigmoid()
                )
            else:  # مخرجات مستمرة (من 0 إلى 1)
                self.actor_heads[name] = nn.Sequential(
                    nn.Linear(prev_dim, dim),
                    nn.Sigmoid()
                )
        
        # رأس الـ Critic (دالة القيمة)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, 1)
        )
        
        # تهيئة الأوزان (Weights)
        self.apply(self._init_weights)

        # --- انحياز أولي: نبدأ بمزيج مستيقظة/نائمة وقوة بث عالية ---
        # sigmoid(-0.5) ≈ 0.38 → حوالي 38% تنام منذ البداية
        # هذا يعطي الذكاء الاصطناعي نقطة بداية جيدة لتعلم أي العقد تنام وأيها تبقى
        if 'sleep_schedule' in self.actor_heads:
            self.actor_heads['sleep_schedule'][0].bias.data.fill_(-0.5)
        if 'transmit_power' in self.actor_heads:
            # sigmoid(2.0) ≈ 0.88 → بث قوي من البداية لضمان التغطية
            self.actor_heads['transmit_power'][0].bias.data.fill_(2.0)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        التمرير الأمامي عبر الشبكة.
        
        المعاملات:
            state: قاموس يحتوي على tensors الحالة
            
        المخرجات:
            action_dict: قاموس يحتوي على tensors الإجراءات
            value: تقدير دالة القيمة
        """
        # تسطيح ودمج مكونات الحالة
        node_positions = state['node_positions']
        battery_levels = state['battery_levels'].unsqueeze(-1)  # إضافة بعد القناة إذا لزم الأمر
        connectivity = state['connectivity']
        
        batch_size = node_positions.size(0)
        num_nodes = node_positions.size(1)
        
        # حساب triu_indices مرة واحدة وتخزينها (تحسين الأداء)
        if self._triu_indices is None or self._num_nodes != num_nodes:
            self._num_nodes = num_nodes
            self._triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=self.device)
        connectivity_flat = connectivity[:, self._triu_indices[0], self._triu_indices[1]]
        
        # دمج جميع مكونات الحالة
        x = torch.cat([
            node_positions.reshape(batch_size, -1),
            battery_levels.reshape(batch_size, -1),
            connectivity_flat.reshape(batch_size, -1)
        ], dim=1)
        
        # استخراج السمات المشترك
        for layer in self.shared_layers:
            x = layer(x)
        
        # رؤوس الـ Actor
        actions = {}
        for name, head in self.actor_heads.items():
            actions[name] = head(x)
        
        # الـ Critic (دالة القيمة)
        value = self.critic(x)
        
        return actions, value
    
    def get_action(self, state: Dict[str, np.ndarray], deterministic: bool = True) -> Dict[str, np.ndarray]:
        """
        الحصول على الإجراء من السياسة.
        
        المعاملات:
            state: قاموس يحتوي على مصفوفات numpy للحالة
            deterministic: ما إذا كان يجب إرجاع إجراءات محددة
            
        المخرجات:
            action_dict: قاموس يحتوي على مصفوفات numpy للإجراءات
        """
        # لا نستدعي self.eval() هنا لأن ذلك بطيء جداً عند استدعائه آلاف المرات
        with torch.no_grad():
            # تحويل إلى tensor
            state_tensor = {}
            for k, v in state.items():
                if isinstance(v, np.ndarray):
                    v = torch.FloatTensor(v)
                # إضافة بعد الدفعة (Batch dimension) إذا لزم الأمر
                if len(v.shape) == 1:
                    v = v.unsqueeze(0)
                elif len(v.shape) == 2:
                    v = v.unsqueeze(0)
                state_tensor[k] = v.to(self.device)
            
            # التمرير الأمامي (Forward pass)
            actions, _ = self.forward(state_tensor)
            
            # تحويل إلى numpy
            action_dict = {}
            for k, v in actions.items():
                if k == 'sleep_schedule':
                    # تطبيق عتبة للقرار الثنائي
                    action = (v > 0.5).float() if deterministic else torch.bernoulli(v)
                else:
                    action = v
                action_dict[k] = action.squeeze(0).cpu().numpy()
            
            return action_dict
        
    def get_value(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        الحصول على تقدير القيمة لحالة معينة.
        
        المعاملات:
            state: قاموس يحتوي على tensors الحالة
            
        المخرجات:
            value: تقدير دالة القيمة
        """
        _, value = self.forward(state)
        return value

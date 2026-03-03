# network.py
import numpy as np

class PhysicalQuantity:
    """
    物理量：表示一个节点或端口。
    它只提供符号（名字）和维度大小，没有任何状态和计算方法。
    """
    def __init__(self, name, size=1):
        self.name = name
        self.size = size

class Component:
    """
    组件基类：连接若干物理量的能量函数。
    它不存储任何状态，只提供能量的计算公式，并利用有限差分自动推导力。
    """
    def __init__(self, name, connected_quantities):
        self.name = name
        # 记录这个组件连接了哪些物理量（端口）
        self.connected_quantities = connected_quantities

    def compute_energy(self, state_dict):
        """计算当前组件的能量。子类必须实现此方法。"""
        raise NotImplementedError

    def compute_force(self, target_quantity, state_dict, eps=1e-5):
        """
        核心物理方法：利用中心差分法，自动计算目标物理量受到的梯度力。
        F = - ∂E / ∂q
        """
        if target_quantity not in self.connected_quantities:
            return np.zeros(target_quantity.size)
        
        q_val = state_dict[target_quantity.name]
        force = np.zeros_like(q_val)
        
        # 遍历该物理量的每一个维度求偏导数
        for i in range(len(q_val)):
            # 正向微扰
            state_plus = {k: v.copy() for k, v in state_dict.items()}
            state_plus[target_quantity.name][i] += eps
            E_plus = self.compute_energy(state_plus)
            
            # 负向微扰
            state_minus = {k: v.copy() for k, v in state_dict.items()}
            state_minus[target_quantity.name][i] -= eps
            E_minus = self.compute_energy(state_minus)
            
            # 差分近似负梯度
            force[i] = -(E_plus - E_minus) / (2 * eps)
            
        return force

class Simulator:
    """
    仿真器：全局状态管理者与时间推进器。
    它持有所有的物理状态，收集所有的力，并使用欧拉积分更新世界。
    """
    def __init__(self, dt=0.005):
        self.dt = dt
        self.quantities = {}
        self.components = []
        
        # 全局状态字典
        self.state = {}
        self.velocity = {}
        
        # 物理参数字典
        self.masses = {}
        self.dampings = {}
        self.is_fixed = {}

    def add_quantity(self, q, init_val, mass=1.0, damping=1.0, fixed=False):
        """将物理量注册到仿真器，并赋予其初始状态和物理属性"""
        self.quantities[q.name] = q
        self.state[q.name] = np.array(init_val, dtype=float)
        self.velocity[q.name] = np.zeros(q.size, dtype=float)
        self.masses[q.name] = mass
        self.dampings[q.name] = damping
        self.is_fixed[q.name] = fixed

    def add_component(self, comp):
        """将能量组件注册到仿真器"""
        self.components.append(comp)

    def set_fixed(self, q_name, fixed):
        self.is_fixed[q_name] = fixed

    def set_state(self, q_name, val):
        self.state[q_name] = np.array(val, dtype=float)

    def step(self):
        """推进一个物理时间步"""
        # 1. 力的清零初始化
        forces = {name: np.zeros(q.size) for name, q in self.quantities.items()}
        
        # 2. 收集系统总力：每个组件独立计算它对相连引脚的拉扯力，并叠加
        for comp in self.components:
            for q in comp.connected_quantities:
                forces[q.name] += comp.compute_force(q, self.state)
                
        # 3. 状态更新：根据牛顿定律演化所有非固定的物理量
        for name, q in self.quantities.items():
            if self.is_fixed[name]:
                self.velocity[name].fill(0.0)
                continue
                
            damping_force = -self.dampings[name] * self.velocity[name]
            total_force = forces[name] + damping_force
            acceleration = total_force / self.masses[name]
            
            self.velocity[name] += acceleration * self.dt
            self.state[name] += self.velocity[name] * self.dt

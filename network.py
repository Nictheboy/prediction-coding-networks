# network.py
import numpy as np

class PhysicalPredictiveNetwork:
    def __init__(self, x0=0.0, y0=0.0, mass=1.0, damping=1.0, stiffness=1.0):
        self.pos = np.array([x0, y0], dtype=float)
        self.vel = np.array([0.0, 0.0], dtype=float)
        
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness

    def step(self, target_pos, dt, is_pulling=True):
        """
        :param is_pulling: 布尔值。模拟手是否握住了摇杆（是否有外部信号输入）。
        """
        # 1. 计算外部输入的预测误差和弹簧力
        if is_pulling:
            prediction_error = np.array(target_pos) - self.pos
            spring_force = self.stiffness * prediction_error
        else:
            # 如果松开手，外部输入断开，预测误差为 0，弹簧不产生拉力
            spring_force = np.array([0.0, 0.0])
            
        # 2. 内部的摩擦耗散力始终存在
        damping_force = -self.damping * self.vel
        
        # 3. 动力学演化
        total_force = spring_force + damping_force
        acceleration = total_force / self.mass
        
        self.vel += acceleration * dt
        self.pos += self.vel * dt
        
        return self.pos.copy()

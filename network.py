# network.py
import numpy as np
from framework import StateQuantity, Component, Simulator

class SpringOperator(Component):
    """胡克弹簧算子"""
    def __init__(self, name, q1, q2, stiffness):
        self.q1 = q1
        self.q2 = q2
        self.stiffness = stiffness
        super().__init__(name, self.get_energy())

    def get_energy(self):
        return 0.5 * self.stiffness * ((self.q1 - self.q2)**2).sum()

q_tgt_x = StateQuantity(is_shared=True, name="tgt_x", size=1, mass=0.1, damping=0.03)
q_tgt_y = StateQuantity(is_shared=True, name="tgt_y", size=1, mass=0.1, damping=0.03)
q_mu_x = StateQuantity(is_shared=True, name="mu_x", size=1, mass=1.0, damping=100.0)
q_mu_y = StateQuantity(is_shared=True, name="mu_y", size=1, mass=1.0, damping=100.0)

comp_bind_x = SpringOperator("bind_x", q_tgt_x, q_mu_x, stiffness=10000.0)
comp_bind_y = SpringOperator("bind_y", q_tgt_y, q_mu_y, stiffness=10000.0)

sim = Simulator(q_tgt_x, q_tgt_y, dt=0.001)

network_exports = {
    "sim": sim,
    "q_tgt_x": q_tgt_x,
    "q_tgt_y": q_tgt_y,
    "q_mu_x": q_mu_x,
    "q_mu_y": q_mu_y
}

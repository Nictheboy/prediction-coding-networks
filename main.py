# main.py
import pygame
import numpy as np
from environment import PhysicalEnvironment
from network import PhysicalQuantity, Component, Simulator

# ==========================================
# 定义我们具体的网络组件 (搭积木的模块)
# ==========================================
class ErrorEnergyComponent(Component):
    """
    具体的能量积木：计算目标值与预测值之间的平方误差能量（即弹簧势能）
    E = 0.5 * stiffness * (target - mu)^2
    """
    def __init__(self, name, q_target, q_mu, stiffness=100.0):
        # 告诉基类，这个组件连接了 q_target 和 q_mu 这两个端口
        super().__init__(name, [q_target, q_mu])
        self.q_target = q_target
        self.q_mu = q_mu
        self.stiffness = stiffness

    def compute_energy(self, state_dict):
        # 组件内部：只负责根据传入的字典和自身的公式计算出能量常数
        val_target = state_dict[self.q_target.name]
        val_mu = state_dict[self.q_mu.name]
        return 0.5 * self.stiffness * np.sum((val_target - val_mu)**2)


class PygameSimulation:
    def __init__(self, env, simulator, width=800, height=800, steps_per_frame=80):
        self.env = env
        self.sim = simulator
        self.width = width
        self.height = height
        self.steps_per_frame = steps_per_frame
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Modular Predictive Coding")
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 20)

        self.fade_surface = pygame.Surface((self.width, self.height))
        self.fade_surface.fill((0, 0, 0))
        self.fade_surface.set_alpha(6)

    def world_to_screen(self, x, y):
        screen_x = int((x + 1.2) / 2.4 * self.width)
        screen_y = int((-y + 1.2) / 2.4 * self.height)
        return screen_x, screen_y

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            mouse_pressed = pygame.mouse.get_pressed()[0]
            is_pulling = keys[pygame.K_SPACE] or mouse_pressed

            self.screen.blit(self.fade_surface, (0, 0))

            status_text = "ENGAGED (Hand on Joystick)" if is_pulling else "RELEASED (Free Dynamics)"
            color = (0, 255, 0) if is_pulling else (200, 100, 0)
            self.screen.blit(self.font.render(f"State: {status_text} | Hold SPACE", True, color), (20, 20))

            for _ in range(self.steps_per_frame):
                # 1. 环境流逝
                env_pos = self.env.step(self.sim.dt)
                
                # 2. 根据交互状态，配置物理量的约束
                if is_pulling:
                    # 握住摇杆：目标节点的坐标被外界强制接管，质量无限大（fixed=True）
                    self.sim.set_fixed("tgt_x", True)
                    self.sim.set_fixed("tgt_y", True)
                    self.sim.set_state("tgt_x", [env_pos[0]])
                    self.sim.set_state("tgt_y", [env_pos[1]])
                else:
                    # 松开手：环境被拔除！目标节点重新获得自由，将受到内部网络的牵引！
                    self.sim.set_fixed("tgt_x", False)
                    self.sim.set_fixed("tgt_y", False)

                # 3. 仿真器步进：一切交给物理差分和牛顿定律
                self.sim.step()

                # 4. 可视化渲染
                # 从仿真器状态中提取分离的 X 和 Y 重组为坐标点
                tgt_screen = self.world_to_screen(self.sim.state["tgt_x"][0], self.sim.state["tgt_y"][0])
                mu_screen = self.world_to_screen(self.sim.state["mu_x"][0], self.sim.state["mu_y"][0])

                if is_pulling:
                    pygame.draw.line(self.screen, (50, 50, 50), tgt_screen, mu_screen, 1)
                    pygame.draw.circle(self.screen, (0, 255, 0), tgt_screen, 1)  # 绿点代表手握住的摇杆

                # 红点代表网络内部的预测状态
                pygame.draw.circle(self.screen, (255, 50, 50), mu_screen, 1)
                
                # 当松开手时，绿点变成了“幽灵点”（摇杆自身），它会被红点（大脑）用弹簧拉着走！
                if not is_pulling:
                    pygame.draw.circle(self.screen, (100, 100, 0), tgt_screen, 1)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    env = PhysicalEnvironment(velocity=1.5)
    
    # ---------------------------------------------------------
    # 使用新框架搭积木！
    # ---------------------------------------------------------
    
    # 1. 声明物理量 (只定义名字和维度，没有数值)
    q_tgt_x = PhysicalQuantity("tgt_x", size=1)
    q_tgt_y = PhysicalQuantity("tgt_y", size=1)
    q_mu_x  = PhysicalQuantity("mu_x", size=1)
    q_mu_y  = PhysicalQuantity("mu_y", size=1)

    # 2. 声明组件 (定义能量结构，将物理量连接起来。X和Y完全独立)
    comp_x = ErrorEnergyComponent("spring_x", q_tgt_x, q_mu_x, stiffness=1.0)
    comp_y = ErrorEnergyComponent("spring_y", q_tgt_y, q_mu_y, stiffness=1.0)

    # 3. 配置仿真器 (赋予初值和物理属性)
    sim = Simulator(dt=0.001)
    
    # 摇杆端口 (tgt_x, tgt_y)：被手握住时 fixed=True，松开时恢复为具有微小质量的自由物体
    sim.add_quantity(q_tgt_x, [0.0], mass=0.1, damping=0.03, fixed=True)
    sim.add_quantity(q_tgt_y, [0.0], mass=0.1, damping=0.03, fixed=True)
    
    # 内部大脑节点 (mu_x, mu_y)：始终自由，有一定质量和惯性
    sim.add_quantity(q_mu_x, [0.0], mass=1.0, damping=0.03)
    sim.add_quantity(q_mu_y, [0.0], mass=1.0, damping=0.03)

    # 装载积木
    sim.add_component(comp_x)
    sim.add_component(comp_y)

    # 运行
    app = PygameSimulation(env, sim, steps_per_frame=80)
    app.run()

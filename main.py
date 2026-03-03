# main.py
import pygame
import numpy as np
import collections
from environment import PhysicalEnvironment
from network import network_exports

class PygameSimulation:
    def __init__(self, env, network_dict, sim_width=800, sim_height=800, steps_per_frame=60):
        self.env = env
        self.sim = network_dict["sim"]
        
        self.q_tgt_x = network_dict["q_tgt_x"]
        self.q_tgt_y = network_dict["q_tgt_y"]
        self.q_mu_x = network_dict["q_mu_x"]
        self.q_mu_y = network_dict["q_mu_y"]

        # 左侧物理沙盒的尺寸
        self.sim_width = sim_width
        self.sim_height = sim_height
        
        # 整体窗口尺寸：宽度翻倍，右侧留给图表
        self.width = sim_width * 2
        self.height = sim_height
        self.steps_per_frame = steps_per_frame
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Predictive Brain: Physical Sandbox & Energy Monitor")
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 20)
        self.small_font = pygame.font.SysFont("consolas", 14)
        
        # 残影表面现在只覆盖左半边的物理沙盒
        self.fade_surface = pygame.Surface((self.sim_width, self.sim_height))
        self.fade_surface.fill((0, 0, 0))
        self.fade_surface.set_alpha(10)

        # ====== 新增：能量监视器 (心电图) ======
        # 我们用一个定长队列 (deque) 来存储历史能量数据
        self.graph_width = self.sim_width - 40 # 左右留白 20
        self.graph_height = 300                # 心电图区域高度
        self.energy_history = collections.deque(maxlen=self.graph_width)

    def world_to_screen(self, x, y):
        # 映射坐标时，限制在左半屏幕 (0 到 sim_width)
        screen_x = int((x + 1.2) / 2.4 * self.sim_width)
        screen_y = int((-y + 1.2) / 2.4 * self.sim_height)
        return screen_x, screen_y

    def draw_ecg_graph(self, current_energy):
        """在右上方绘制心电图风格的能量曲线"""
        # 1. 定义右侧上方横条的区域并用深灰色清空背景
        graph_rect = pygame.Rect(self.sim_width, 0, self.sim_width, self.graph_height)
        pygame.draw.rect(self.screen, (15, 15, 18), graph_rect)
        pygame.draw.line(self.screen, (100, 100, 100), (self.sim_width, self.graph_height), (self.width, self.graph_height), 2)

        # 2. 画一点暗色的网格线增加科技感 (Oscilloscope style)
        for y in range(50, self.graph_height, 50):
            pygame.draw.line(self.screen, (30, 40, 30), (self.sim_width, y), (self.width, y), 1)

        # 3. 将最新能量存入队列
        self.energy_history.append(current_energy)

        # 4. 动态计算 Y 轴缩放比例 (自适应量程)
        if len(self.energy_history) > 1:
            max_e = max(self.energy_history)
            if max_e < 1e-5: max_e = 1e-5 # 防止除以零
            
            points = []
            for i, e in enumerate(self.energy_history):
                # X 坐标向右平移到右半屏幕
                px = self.sim_width + 20 + i
                # Y 坐标从底部 (graph_height - 20) 向上画，按最大值动态缩放
                py = self.graph_height - 20 - (e / max_e) * (self.graph_height - 50)
                points.append((px, py))
                
            # 用青蓝色画出平滑的抗锯齿连线
            pygame.draw.aalines(self.screen, (0, 255, 200), False, points)

        # 5. 渲染实时文字数据
        title_text = self.font.render("Sensory Spring Energy (Prediction Error)", True, (200, 200, 200))
        val_text = self.font.render(f"E = {current_energy:.4f}", True, (0, 255, 200))
        
        self.screen.blit(title_text, (self.sim_width + 20, 15))
        self.screen.blit(val_text, (self.sim_width + 20, 40))


    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            is_free = pygame.key.get_pressed()[pygame.K_SPACE] or pygame.mouse.get_pressed()[0]
            is_pulling = not is_free

            # 仅在左半屏幕覆盖残影层
            self.screen.blit(self.fade_surface, (0, 0))

            if is_pulling:
                color, text = (0, 255, 0), "LEARNING (Tethered) | Press SPACE to Dream"
            else:
                color, text = (255, 100, 255), "DREAMING (Free Run) | Release SPACE to Learn"
            self.screen.blit(self.font.render(text, True, color), (20, 20))

            # 物理推演
            for _ in range(self.steps_per_frame):
                env_pos = self.env.step(self.sim.dt)
                
                if is_pulling:
                    controls = {self.q_tgt_x: env_pos[0], self.q_tgt_y: env_pos[1]}
                else:
                    controls = {} 
                    
                result_state = self.sim.step(controls)

                # 计算当前物理帧的弹簧总能量：E = 0.5 * k * (\Delta x^2 + \Delta y^2)
                tgt_x, tgt_y = result_state[self.q_tgt_x][0], result_state[self.q_tgt_y][0]
                mu_x, mu_y = result_state[self.q_mu_x][0], result_state[self.q_mu_y][0]
                # 这里我们直接按网络中的 10000 刚度来计算客观势能
                spring_energy = 0.5 * 10000.0 * ((tgt_x - mu_x)**2 + (tgt_y - mu_y)**2)

                # 左侧物理沙盒渲染
                env_screen = self.world_to_screen(*env_pos)
                pygame.draw.circle(self.screen, (0, 255, 0), env_screen, 2)

                mu_screen = self.world_to_screen(mu_x, mu_y)
                pygame.draw.circle(self.screen, (255, 50, 50), mu_screen, 2)

                if is_pulling:
                    pygame.draw.line(self.screen, (80, 80, 80), env_screen, mu_screen, 1)
                else:
                    tgt_screen = self.world_to_screen(tgt_x, tgt_y)
                    pygame.draw.circle(self.screen, (255, 255, 0), tgt_screen, 1)

            # 在所有物理帧跑完后，渲染右侧心电图（避免拖慢计算速度）
            self.draw_ecg_graph(spring_energy)

            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()

if __name__ == "__main__":
    env = PhysicalEnvironment(velocity=10)
    app = PygameSimulation(env, network_exports, steps_per_frame=60)
    app.run()

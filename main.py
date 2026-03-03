# main.py
import pygame
import numpy as np
from environment import PhysicalEnvironment
from network import PhysicalPredictiveNetwork

class PygameSimulation:
    def __init__(self, env, network, width=800, height=800, steps_per_frame=80, dt=0.005):
        self.env = env
        self.network = network
        self.width = width
        self.height = height
        self.steps_per_frame = steps_per_frame
        self.dt = dt
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Predictive Coding: True Physics Time")
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
        self.screen.fill((0, 0, 0))

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            mouse_pressed = pygame.mouse.get_pressed()[0]
            is_pulling = keys[pygame.K_SPACE] or mouse_pressed

            self.screen.blit(self.fade_surface, (0, 0))

            status_text = "ENGAGED (Pulling)" if is_pulling else "RELEASED (Free Coasting)"
            color = (0, 255, 0) if is_pulling else (100, 100, 100)
            text_surface = self.font.render(f"State: {status_text} | Hold SPACE or Mouse", True, color)
            self.screen.blit(text_surface, (20, 20))

            # 核心修改：真正的基于 dt 的演化
            for _ in range(self.steps_per_frame):
                
                # 1. 环境流逝 dt，计算新的外部坐标 (如果松开手，可以让环境继续运动，或者你想让它停下都可以。这里让它继续跑)
                target_pos = self.env.step(self.dt)
                
                # 2. 网络流逝 dt
                pred_pos = self.network.step(target_pos, self.dt, is_pulling=is_pulling)

                screen_target = self.world_to_screen(*target_pos)
                screen_pred = self.world_to_screen(*pred_pos)

                if is_pulling:
                    pygame.draw.line(self.screen, (50, 50, 50), screen_target, screen_pred, 1)
                    pygame.draw.circle(self.screen, (0, 255, 0), screen_target, 1)

                pygame.draw.circle(self.screen, (255, 50, 50), screen_pred, 1)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    # 实例化环境，你可以通过 velocity 调整摇杆的基础速度
    env = PhysicalEnvironment(velocity=1.5)
    
    # 你的完美参数
    brain_network = PhysicalPredictiveNetwork(
        x0=0.0, y0=0.0,   
        mass=3.0,         
        damping=0.03,      
        stiffness=1.0   
    )
    
    # 现在，如果你把 dt 改成 0.01，整个世界（包括摇杆和振子）都会精确地快两倍！
    sim = PygameSimulation(env=env, network=brain_network, steps_per_frame=100, dt=0.001)
    sim.run()

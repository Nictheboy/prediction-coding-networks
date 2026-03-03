import numpy as np
import itertools
import pygame

class FaceTrajectoryStream:
    """
    环境封装类：提供绝对连续的 (x, y) 物理坐标流。
    拓扑结构已根据优化：外圈 -> 眼镜 -> 右脸边缘 -> 嘴巴 -> 左脸边缘 -> 循环
    """
    def __init__(self, points_per_unit=3000):
        self.points_per_unit = points_per_unit
        self.path = self._build_continuous_path()
        self._stream = itertools.cycle(self.path)
        
    def _add_line(self, x1, y1, x2, y2):
        length = np.hypot(x2 - x1, y2 - y1)
        num_points = max(2, int(length * self.points_per_unit))
        return np.column_stack((
            np.linspace(x1, x2, num_points, endpoint=False),
            np.linspace(y1, y2, num_points, endpoint=False)
        ))

    def _add_arc(self, cx, cy, r, start_theta, end_theta, cw=True):
        """支持精确顺时针(cw)/逆时针绘制的圆弧"""
        if cw:
            while end_theta > start_theta:
                end_theta -= 2 * np.pi
        else:
            while end_theta < start_theta:
                end_theta += 2 * np.pi
                
        length = abs(end_theta - start_theta) * r
        num_points = max(2, int(length * self.points_per_unit))
        theta = np.linspace(start_theta, end_theta, num_points, endpoint=False)
        return np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))

    def _build_continuous_path(self):
        segments = []
        
        # 预计算四个关键节点的精确坐标与极角
        # A: 左镜腿交点, B: 右镜腿交点, C: 左嘴角交点, D: 右嘴角交点
        xA, yA = -np.sqrt(1 - 0.3**2), 0.3
        xB, yB =  np.sqrt(1 - 0.3**2), 0.3
        xC, yC = -np.sqrt(1 - (-0.4)**2), -0.4
        xD, yD =  np.sqrt(1 - (-0.4)**2), -0.4

        tA = np.pi - np.arcsin(0.3)
        tB = np.arcsin(0.3)
        tD = -np.arcsin(0.4)
        tC = -np.pi + np.arcsin(0.4)

        # ====== 全新优化的无缝拓扑循环 ======

        # 1. 完整画出一圈脸部轮廓 (A -> B -> D -> C -> A)
        segments.append(self._add_arc(0, 0, 1, tA, tB, cw=True))              # 顶弧
        segments.append(self._add_arc(0, 0, 1, tB, tD, cw=True))              # 右弧
        segments.append(self._add_arc(0, 0, 1, tD, tC, cw=True))              # 底弧
        segments.append(self._add_arc(0, 0, 1, tC, tA - 2*np.pi, cw=True))    # 左弧

        # 2. 从 A 深入，画眼镜 (A -> B)
        segments.append(self._add_line(xA, yA, -0.6, 0.3))                    # 左镜腿
        segments.append(self._add_arc(-0.4, 0.3, 0.2, np.pi, -np.pi, cw=True))# 完整左眼
        segments.append(self._add_arc(-0.4, 0.3, 0.2, np.pi, 0, cw=True))     # 左眼上半圈(过半)
        segments.append(self._add_line(-0.2, 0.3, 0.2, 0.3))                  # 鼻梁
        segments.append(self._add_arc(0.4, 0.3, 0.2, np.pi, -np.pi, cw=True)) # 完整右眼
        segments.append(self._add_arc(0.4, 0.3, 0.2, np.pi, 0, cw=True))      # 右眼上半圈(过半)
        segments.append(self._add_line(0.6, 0.3, xB, yB))                     # 右镜腿

        # 3. 顺着右脸颊向下滑动，前往嘴巴 (B -> D)
        segments.append(self._add_arc(0, 0, 1, tB, tD, cw=True))

        # 4. 画嘴巴，从右到左 (D -> C)
        segments.append(self._add_line(xD, yD, 0.3, -0.4))                    # 嘴右侧平线
        segments.append(self._add_arc(0, -0.4, 0.3, 0, -np.pi, cw=True))      # 嘴部下凹
        segments.append(self._add_line(-0.3, -0.4, xC, yC))                   # 嘴左侧平线

        # 5. 顺着左脸颊向上滑动，完美回到起点 A (C -> A)
        segments.append(self._add_arc(0, 0, 1, tC, tA - 2*np.pi, cw=True))

        return np.vstack(segments)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._stream)


class PygameOscilloscope:
    """
    真正的物理仿真绘图库 (Pygame)。
    完全不破坏封装，仅通过调用 next(stream) 读取数据，
    利用屏幕缓冲区的半透明覆盖实现极客风格的物理余辉。
    """
    def __init__(self, stream, width=600, height=600, steps_per_frame=80):
        self.stream = stream
        self.width = width
        self.height = height
        self.steps_per_frame = steps_per_frame
        
        # 初始化 Pygame 引擎
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Environment: Predictive Coding Target")
        self.clock = pygame.time.Clock()

        # 创建一个用于制造“渐隐尾迹”的半透明黑色图层
        self.fade_surface = pygame.Surface((self.width, self.height))
        self.fade_surface.fill((0, 0, 0))
        # 透明度越低 (如 3 或 5)，余辉停留时间越长
        self.fade_surface.set_alpha(4) 

    def run(self):
        running = True
        self.screen.fill((0, 0, 0)) # 初始纯黑背景

        while running:
            # 处理窗口关闭事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 1. 铺上一层半透明黑底，让旧的轨迹变暗一点点
            self.screen.blit(self.fade_surface, (0, 0))

            # 2. 从流中吸取极其微小的物理时间步 dt 产生的新位置，并绘制出最亮的点
            for _ in range(self.steps_per_frame):
                x, y = next(self.stream)
                
                # 将 (-1.2, 1.2) 的物理坐标系映射到屏幕像素坐标系
                screen_x = int((x + 1.2) / 2.4 * self.width)
                # Pygame 的 Y 轴是向下的，所以需要反转 y
                screen_y = int((-y + 1.2) / 2.4 * self.height) 
                
                # 绘制荧光绿色的点，就像模拟示波器上的电子束
                pygame.draw.circle(self.screen, (50, 255, 100), (screen_x, screen_y), 1)

            # 刷新屏幕显示
            pygame.display.flip()
            
            # 控制帧率上限为 60 帧/秒
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    # 物理流：极高的密度，绝对的连续性
    env_stream = FaceTrajectoryStream(points_per_unit=3000)
    
    # 渲染器：纯粹的读取与缓冲
    # steps_per_frame 控制电子束跑得多快
    animator = PygameOscilloscope(stream=env_stream, steps_per_frame=180)
    animator.run()

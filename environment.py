# environment.py
import numpy as np

class PhysicalEnvironment:
    """
    一个真正基于物理时间 dt 和解析几何的连续环境。
    通过轨道参数 s (弧长) 实现了无限精度的连续运动。
    """
    def __init__(self, velocity=1.5):
        """
        :param velocity: 绿点（摇杆）在轨道上运动的线速度 (单位长度/秒)
        """
        self.velocity = velocity
        self.segments = []             # 存储线段或圆弧的解析参数
        self.cumulative_lengths = [0.0] # 存储到每一段为止的累积总弧长
        
        self._build_analytical_path()
        self.total_length = self.cumulative_lengths[-1]
        self.current_s = 0.0           # 当前绿点走过的弧长

    def _add_line(self, x1, y1, x2, y2):
        length = np.hypot(x2 - x1, y2 - y1)
        self.segments.append(('line', x1, y1, x2, y2, length))
        self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)

    def _add_arc(self, cx, cy, r, start_theta, end_theta):
        """
        :param start_theta, end_theta: 绝对极角。差值的正负自然决定了顺/逆时针
        """
        length = abs(end_theta - start_theta) * r
        self.segments.append(('arc', cx, cy, r, start_theta, end_theta, length))
        self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)

    def _build_analytical_path(self):
        # 预计算四个关键节点的精确坐标与极角
        xA, yA = -np.sqrt(1 - 0.3**2), 0.3
        xB, yB =  np.sqrt(1 - 0.3**2), 0.3
        xC, yC = -np.sqrt(1 - (-0.4)**2), -0.4
        xD, yD =  np.sqrt(1 - (-0.4)**2), -0.4

        tA = np.pi - np.arcsin(0.3)
        tB = np.arcsin(0.3)
        tD = -np.arcsin(0.4)
        tC = -np.pi + np.arcsin(0.4)

        # 1. 脸部轮廓 (A -> B -> D -> C -> A) 顺时针，角度不断减小
        self._add_arc(0, 0, 1, tA, tB)
        self._add_arc(0, 0, 1, tB, tD)
        self._add_arc(0, 0, 1, tD, tC)
        self._add_arc(0, 0, 1, tC, tA - 2*np.pi)

        # 2. 眼镜 (A -> B)
        self._add_line(xA, yA, -0.6, 0.3)
        self._add_arc(-0.4, 0.3, 0.2, np.pi, -np.pi)  # 完整左眼
        self._add_arc(-0.4, 0.3, 0.2, np.pi, 0)       # 左眼上半圈
        self._add_line(-0.2, 0.3, 0.2, 0.3)           # 鼻梁
        self._add_arc(0.4, 0.3, 0.2, np.pi, -np.pi)   # 完整右眼
        self._add_arc(0.4, 0.3, 0.2, np.pi, 0)        # 右眼上半圈
        self._add_line(0.6, 0.3, xB, yB)

        # 3. 顺右脸颊滑下
        self._add_arc(0, 0, 1, tB, tD)

        # 4. 画嘴巴 (D -> C)
        self._add_line(xD, yD, 0.3, -0.4)
        self._add_arc(0, -0.4, 0.3, 0, -np.pi)        # 嘴部下凹
        self._add_line(-0.3, -0.4, xC, yC)

        # 5. 顺左脸颊滑上回到 A
        self._add_arc(0, 0, 1, tC, tA - 2*np.pi)

    def step(self, dt):
        """核心：物理时间推移，基于速度和 dt 计算新的距离 ds"""
        self.current_s = (self.current_s + self.velocity * dt) % self.total_length
        return self._get_position_at(self.current_s)

    def _get_position_at(self, s):
        """解析几何插值：根据当前弧长 s，精确定位 (x, y) 坐标"""
        for i, seg in enumerate(self.segments):
            start_s = self.cumulative_lengths[i]
            end_s = self.cumulative_lengths[i+1]
            
            if start_s <= s <= end_s or (i == len(self.segments)-1 and s >= end_s):
                local_s = s - start_s
                
                if seg[0] == 'line':
                    _, x1, y1, x2, y2, length = seg
                    t = local_s / length if length > 0 else 0
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    return np.array([x, y])
                    
                elif seg[0] == 'arc':
                    _, cx, cy, r, st, et, length = seg
                    # 按照 local_s 与总弧长的比例，线性插值极角
                    theta = st + (et - st) * (local_s / length)
                    return np.array([cx + r * np.cos(theta), cy + r * np.sin(theta)])
                    
        return np.array([0.0, 0.0])

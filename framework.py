# framework.py
import numpy as np

# ==========================================
# 1. 接口与核心概念
# ==========================================
class Quantity:
    def __init__(self, name=None):
        self.name = name
        self.components = []

    def get_name(self): return self.name
    def add_component(self, comp):
        if comp not in self.components: self.components.append(comp)
    def get_components(self): return self.components

    def __add__(self, other): return FunctionQuantity("add", [self, other])
    def __radd__(self, other): return FunctionQuantity("add", [other, self])
    def __sub__(self, other): return FunctionQuantity("sub", [self, other])
    def __rsub__(self, other): return FunctionQuantity("sub", [other, self])
    def __mul__(self, other): return FunctionQuantity("mul", [self, other])
    def __rmul__(self, other): return FunctionQuantity("mul", [other, self])
    def __matmul__(self, other): return FunctionQuantity("matmul", [self, other])
    def __pow__(self, other): return FunctionQuantity("pow", [self, other])
    
    def sum(self): return FunctionQuantity("sum", [self])
    def tanh(self): return FunctionQuantity("tanh", [self])
    def reshape(self, shape): return FunctionQuantity("reshape", [self, shape])


class StateQuantity(Quantity):
    def __init__(self, is_shared, name=None, size=1, init_val=None, mass=1.0, damping=1.0):
        super().__init__(name)
        self.is_shared = is_shared
        self.size = size
        self.mass = mass
        self.damping = damping
        self.init_val = np.array(init_val, dtype=float) if init_val is not None else np.zeros(size)
        
        # [核心新增] 屏蔽字典：记录对特定组件的敏感度系数
        self.insensitivities = {}

    def set_insensitivity(self, comp_name, scale=0.0):
        """
        设置对特定能量组件的不敏感度（实现三极管效应/非互易性）。
        scale=1.0: 正常受力 (牛顿第三定律)。
        scale=0.0: 绝对屏蔽反作用力 (等价于针对该组件广义质量无穷大)。
        """
        self.insensitivities[comp_name] = scale


class FunctionQuantity(Quantity):
    def __init__(self, func_name, args, name=None):
        super().__init__(name)
        self.func_name = func_name
        self.args = args
    def get_func_name(self): return self.func_name
    def get_args(self): return self.args


# ==========================================
# 2. 组件基类
# ==========================================
class Component:
    def __init__(self, name, energy_func_quantity):
        self.name = name
        self.energy = energy_func_quantity
        self._link_graph(self.energy)

    def _link_graph(self, node):
        if isinstance(node, StateQuantity):
            node.add_component(self)
        elif isinstance(node, FunctionQuantity):
            for arg in node.get_args(): self._link_graph(arg)

    def get_name(self): return self.name
    def get_energy(self): return self.energy


# ==========================================
# 3. 解析与仿真器引擎 (含 AutoGrad)
# ==========================================
class Simulator:
    def __init__(self, *controllable_states, dt=0.001):
        self.dt = dt
        self.states = set()
        self.components = set()

        queue = list(controllable_states)
        visited = set(queue)

        while queue:
            curr_state = queue.pop(0)
            self.states.add(curr_state)
            for comp in curr_state.get_components():
                if comp not in self.components:
                    self.components.add(comp)
                    self._extract_states_from_ast(comp.get_energy(), queue, visited)

        self.state_dict = {s: s.init_val.copy() for s in self.states}
        self.vel_dict = {s: np.zeros(s.size) for s in self.states}

    def _extract_states_from_ast(self, node, queue, visited):
        if isinstance(node, StateQuantity):
            if node not in visited:
                visited.add(node)
                queue.append(node)
        elif isinstance(node, FunctionQuantity):
            for arg in node.get_args():
                self._extract_states_from_ast(arg, queue, visited)

    def _autograd(self, energy_ast):
        values = {}
        topo_order = []
        visited = set()

        def forward(node):
            if node in visited: return values[node]
            visited.add(node)

            if isinstance(node, StateQuantity):
                val = self.state_dict[node]
                values[node] = val
                topo_order.append(node)
                return val
            elif isinstance(node, FunctionQuantity):
                args = [forward(a) if isinstance(a, Quantity) else a for a in node.get_args()]
                op = node.get_func_name()
                if op == "add": val = args[0] + args[1]
                elif op == "sub": val = args[0] - args[1]
                elif op == "mul": val = args[0] * args[1]
                elif op == "matmul": val = args[0] @ args[1]
                elif op == "tanh": val = np.tanh(args[0])
                elif op == "sum": val = np.sum(args[0])
                elif op == "reshape": val = np.reshape(args[0], args[1])
                elif op == "pow": val = args[0] ** args[1]
                else: val = 0.0

                values[node] = val
                topo_order.append(node)
                return val
            return node 
        forward(energy_ast)

        grads = {node: np.zeros_like(values[node]) if isinstance(node, Quantity) else 0.0 for node in topo_order}
        grads[energy_ast] = np.ones_like(values[energy_ast])

        for node in reversed(topo_order):
            if not isinstance(node, FunctionQuantity): continue
            g = grads[node]
            op = node.get_func_name()
            args = node.get_args()
            a = args[0]
            b = args[1] if len(args) > 1 else None

            va = values[a] if isinstance(a, Quantity) else a
            vb = values[b] if isinstance(b, Quantity) else b

            if op == "add":
                if isinstance(a, Quantity): grads[a] += g
                if isinstance(b, Quantity): grads[b] += g
            elif op == "sub":
                if isinstance(a, Quantity): grads[a] += g
                if isinstance(b, Quantity): grads[b] -= g
            elif op == "mul":
                if isinstance(a, Quantity): grads[a] += g * vb
                if isinstance(b, Quantity): grads[b] += g * va
            elif op == "matmul":
                if np.ndim(va) == 2 and np.ndim(vb) == 1:
                    if isinstance(a, Quantity): grads[a] += np.outer(g, vb)
                    if isinstance(b, Quantity): grads[b] += va.T @ g
                elif np.ndim(va) == 2 and np.ndim(vb) == 2:
                    if isinstance(a, Quantity): grads[a] += g @ vb.T
                    if isinstance(b, Quantity): grads[b] += va.T @ g
            elif op == "tanh":
                if isinstance(a, Quantity): grads[a] += g * (1.0 - va**2)
            elif op == "sum":
                if isinstance(a, Quantity): grads[a] += g * np.ones_like(va)
            elif op == "reshape":
                if isinstance(a, Quantity): grads[a] += np.reshape(g, np.shape(va))
            elif op == "pow":
                if isinstance(a, Quantity): grads[a] += g * vb * (va ** (vb - 1))

        return {s: grads[s] for s in self.states if s in grads}

    def step(self, controls_dict):
        for state, val in controls_dict.items():
            arr = np.array(val, dtype=float)
            if arr.shape == (): arr = np.full((state.size,), float(arr))
            else: arr = np.reshape(arr, (state.size,))
            self.state_dict[state] = arr
            self.vel_dict[state].fill(0.0)

        forces = {s: np.zeros(s.size) for s in self.states}

        for comp in self.components:
            state_grads = self._autograd(comp.get_energy())
            comp_name = comp.get_name()
            
            for s, grad in state_grads.items():
                if s not in controls_dict:
                    # [核心修改]：乘以不敏感度系数。
                    # 如果未设置，get 默认返回 1.0 (正常受力)。
                    # 如果设置为 0.0，力瞬间归零，实现了单向信息流！
                    scale = s.insensitivities.get(comp_name, 1.0)
                    forces[s] += -grad.flatten() * scale

        for s in self.states:
            if s in controls_dict: continue
            damping_force = -s.damping * self.vel_dict[s]
            accel = (forces[s] + damping_force) / s.mass
            self.vel_dict[s] += accel * self.dt
            self.state_dict[s] += self.vel_dict[s] * self.dt

        return {s: self.state_dict[s].copy() for s in self.states if s.is_shared}

from ..core.parameter import Parameter


class SGD:
    """SGD优化器"""
    
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for name, param in self.params:
            if hasattr(param, 'grad'):
                param.grad = None
    
    def step(self):
        for name, param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                param.data -= self.lr * param.grad


class Adam:
    """Adam优化器"""
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}
    
    def zero_grad(self):
        for name, param in self.params:
            if hasattr(param, 'grad'):
                param.grad = None
    
    def step(self):
        self.t += 1
        for name, param in self.params:
            if param.grad is not None:
                if name not in self.m:
                    self.m[name] = 0
                    self.v[name] = 0
                
                self.m[name] = self.betas[0] * self.m[name] + (1 - self.betas[0]) * param.grad
                self.v[name] = self.betas[1] * self.v[name] + (1 - self.betas[1]) * (param.grad ** 2)
                
                m_hat = self.m[name] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[name] / (1 - self.betas[1] ** self.t)
                
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
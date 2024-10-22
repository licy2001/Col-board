import torch.nn as nn


###### version 1
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        """
        更新方法，用于在训练过程中更新模型参数的滑动平均版本。
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1.0 - self.mu) * param.data.to(
                    self.shadow[name].device
                ) + self.mu * self.shadow[name].data

    def ema(self, module):
        """
        EMA 方法，用于将模型参数设置为其滑动平均版本。
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        """
        创建一个指数移动平均（Exponential Moving Average，EMA）版本的神经网络模型的副本
        """
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(
                inner_module.config.device
            )
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        """
        返回当前滑动平均版本的状态字典。
        """
        return self.shadow

    def load_state_dict(self, state_dict):
        """
        从给定的状态字典加载滑动平均版本
        """
        self.shadow = state_dict


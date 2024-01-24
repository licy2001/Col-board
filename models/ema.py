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


###### 第二个版本的EMA
"""
ema = EMA(model, 0.999)
ema.register()

# 训练过程中，更新完参数后，同步update shadow weights
def train():
    optimizer.step()
    ema.update()

# eval前，apply shadow weights；eval之后，恢复原来模型的参数
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()
"""


class EMA(object):
    def __init__(self, model, decay):
        # 如果模型是 DataParallel 的实例，获取原始模型
        if isinstance(model, nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data.to("cuda") + self.decay * self.shadow[name].to("cuda")
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {
            # 'model_state_dict': self.model.state_dict(),
            'decay': self.decay,
            'shadow': self.shadow,
            'backup': self.backup
        }

    def load_state_dict(self, state_dict):
        # self.model.load_state_dict(state_dict['model_state_dict'])
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup']

"""
SpikeMode - SNN 计算模式控制器
==============================

支持在 **位精确模式** 和 **时间动力学模式** 之间灵活切换。

模式说明
--------
- **BIT_EXACT**: 位精确模式，每次 forward 前清除膜电位，确保组合逻辑正确
- **TEMPORAL**: 时间动力学模式，保留膜电位残差，用于训练和时间建模

控制层次
--------
1. 全局模式 - 适合整个项目的默认行为
2. 上下文管理器 - 适合训练/推理阶段切换 (类似 torch.no_grad())
3. 实例级覆盖 - 适合特定组件需要固定行为

使用示例
--------
```python
from atomic_ops import SpikeMode, VecAND

# 1. 默认位精确模式 (推理/验证)
gate = VecAND()
result = gate(a, b)  # 每次调用独立，无残差

# 2. 全局切换到时间动力学模式 (训练)
SpikeMode.set_global_mode(SpikeMode.TEMPORAL)
for epoch in range(100):
    output = model(input)  # 残差累积，构建时间动力学

# 3. 上下文管理器 (局部切换)
with SpikeMode.temporal():
    loss = model(input)
    loss.backward()

with SpikeMode.bit_exact():
    result = model(input)  # 推理

# 4. 实例级覆盖 (特定组件固定行为)
gate = VecAND(mode=SpikeMode.TEMPORAL)  # 该实例始终保留残差
```

作者: MofNeuroSim Project
"""

import threading
from contextlib import contextmanager


class SpikeMode:
    """SNN 计算模式控制器

    提供三个层次的模式控制:
    1. 全局模式: 通过 set_global_mode() 设置
    2. 上下文模式: 通过 with SpikeMode.temporal(): 或 with SpikeMode.bit_exact(): 临时切换
    3. 实例模式: 组件初始化时传入 mode 参数

    优先级: 实例模式 > 上下文模式 > 全局模式

    Attributes:
        BIT_EXACT: 位精确模式常量
        TEMPORAL: 时间动力学模式常量
    """

    # 模式常量
    BIT_EXACT = 'bit_exact'    # 位精确模式: 每次 forward 前清除膜电位
    TEMPORAL = 'temporal'       # 时间动力学模式: 保留膜电位残差

    # 线程安全的全局状态
    _local = threading.local()
    _global_mode = BIT_EXACT

    @classmethod
    def _get_context_stack(cls):
        """获取当前线程的上下文栈 (线程安全)"""
        if not hasattr(cls._local, 'context_stack'):
            cls._local.context_stack = []
        return cls._local.context_stack

    @classmethod
    def get_mode(cls):
        """获取当前有效模式

        优先级: 上下文模式 > 全局模式

        Returns:
            str: 当前模式 (BIT_EXACT 或 TEMPORAL)
        """
        stack = cls._get_context_stack()
        if stack:
            return stack[-1]
        return cls._global_mode

    @classmethod
    def set_global_mode(cls, mode):
        """设置全局模式

        Args:
            mode: SpikeMode.BIT_EXACT 或 SpikeMode.TEMPORAL

        Raises:
            ValueError: 如果 mode 不是有效模式
        """
        if mode not in (cls.BIT_EXACT, cls.TEMPORAL):
            raise ValueError(f"Invalid mode: {mode}. Use SpikeMode.BIT_EXACT or SpikeMode.TEMPORAL")
        cls._global_mode = mode

    @classmethod
    def get_global_mode(cls):
        """获取全局模式 (忽略上下文)

        Returns:
            str: 全局模式设置
        """
        return cls._global_mode

    @classmethod
    @contextmanager
    def mode(cls, mode):
        """上下文管理器: 临时切换到指定模式

        Args:
            mode: SpikeMode.BIT_EXACT 或 SpikeMode.TEMPORAL

        Yields:
            None

        Example:
            with SpikeMode.mode(SpikeMode.TEMPORAL):
                output = model(input)  # 时间动力学模式
            # 退出后恢复原模式
        """
        if mode not in (cls.BIT_EXACT, cls.TEMPORAL):
            raise ValueError(f"Invalid mode: {mode}. Use SpikeMode.BIT_EXACT or SpikeMode.TEMPORAL")
        stack = cls._get_context_stack()
        stack.append(mode)
        try:
            yield
        finally:
            stack.pop()

    @classmethod
    @contextmanager
    def bit_exact(cls):
        """上下文管理器: 临时切换到位精确模式

        Example:
            with SpikeMode.bit_exact():
                result = model(input)  # 位精确推理
        """
        with cls.mode(cls.BIT_EXACT):
            yield

    @classmethod
    @contextmanager
    def temporal(cls):
        """上下文管理器: 临时切换到时间动力学模式

        Example:
            with SpikeMode.temporal():
                loss = model(input)
                loss.backward()  # 训练，保留时间残差
        """
        with cls.mode(cls.TEMPORAL):
            yield

    @classmethod
    def should_reset(cls, instance_mode=None):
        """判断是否应该在 forward 前重置膜电位

        供逻辑门内部使用，根据当前模式决定是否清除状态。

        Args:
            instance_mode: 实例级模式覆盖，None 表示跟随全局/上下文

        Returns:
            bool: True 表示应该在 forward 前 reset
        """
        mode = instance_mode if instance_mode is not None else cls.get_mode()
        return mode == cls.BIT_EXACT

    @classmethod
    def is_bit_exact(cls):
        """检查当前是否为位精确模式

        Returns:
            bool: True 表示当前为位精确模式
        """
        return cls.get_mode() == cls.BIT_EXACT

    @classmethod
    def is_temporal(cls):
        """检查当前是否为时间动力学模式

        Returns:
            bool: True 表示当前为时间动力学模式
        """
        return cls.get_mode() == cls.TEMPORAL

"""
训练模式枚举
===========

定义 SNN 层的训练模式，支持不同的训练策略。

作者: MofNeuroSim Project
"""


class TrainingMode:
    """SNN 训练模式枚举

    支持的模式:
    - NONE: 纯推理模式，权重为 buffer，无梯度
    - STE: 位精确 STE 训练，使用纯 SNN 门电路计算梯度
    - TEMPORAL: 时间动力学训练 (未来扩展)

    使用示例:
    ```python
    from atomic_ops import TrainingMode, SpikeFP32Linear_MultiPrecision

    # 推理模式 (默认)
    linear_infer = SpikeFP32Linear_MultiPrecision(64, 32)

    # 位精确 STE 训练模式
    linear_train = SpikeFP32Linear_MultiPrecision(
        64, 32,
        training_mode=TrainingMode.STE
    )
    ```
    """

    # 纯推理模式
    # - 权重为 buffer (非 Parameter)
    # - 无梯度计算
    NONE = None

    # 位精确 STE 训练模式
    # - 权重为脉冲格式 Parameter
    # - Backward 使用纯 SNN 门电路
    # - 符合 CLAUDE.md 纯 SNN 约束
    STE = 'ste'

    # 时间动力学训练模式 (未来扩展)
    # - 保留膜电位残差
    # - 使用时间依赖的学习规则
    TEMPORAL = 'temporal'

    @classmethod
    def is_trainable(cls, mode):
        """检查模式是否需要训练支持"""
        return mode in (cls.STE, cls.TEMPORAL)

    @classmethod
    def is_ste(cls, mode):
        """检查是否为 STE 模式"""
        return mode == cls.STE

    @classmethod
    def is_temporal(cls, mode):
        """检查是否为时间动力学模式"""
        return mode == cls.TEMPORAL

    @classmethod
    def validate(cls, mode):
        """验证训练模式是否有效"""
        valid_modes = (cls.NONE, cls.STE, cls.TEMPORAL)
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid training_mode: {mode}. "
                f"Expected one of: None, 'ste', 'temporal'"
            )
        return mode

"""Linear layers"""
from .fp8 import *
from .fp16 import *
from .fp32 import *

# Backward compatibility alias
SpikeFP32Linear = SpikeFP32Linear_MultiPrecision

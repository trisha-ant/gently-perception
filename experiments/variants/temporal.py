from perception.temporal import perceive_temporal
from ._adapter import adapt

perceive = adapt(perceive_temporal)

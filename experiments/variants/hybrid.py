from perception.hybrid import perceive_hybrid
from ._adapter import adapt

perceive = adapt(perceive_hybrid)

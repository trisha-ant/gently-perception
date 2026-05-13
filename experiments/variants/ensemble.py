from perception.ensemble import perceive_ensemble
from ._adapter import adapt

perceive = adapt(perceive_ensemble)

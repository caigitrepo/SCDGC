from models.scdgc import SCDGC
from models.baseline import Baseline

model_factory = {
    'scdgc' : SCDGC, 
    'baseline': Baseline,
}
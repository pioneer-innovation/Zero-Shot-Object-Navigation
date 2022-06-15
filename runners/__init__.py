from .nonadaptivea3c_train import nonadaptivea3c_train,nonadaptivea3c_train_seen
from .nonadaptivea3c_val import nonadaptivea3c_val,nonadaptivea3c_val_unseen,nonadaptivea3c_val_seen
from .savn_train import savn_train,savn_train_seen
from .savn_val import savn_val,savn_val_unseen,savn_val_seen

trainers = [ 
    'vanilla_train',
    'learned_train',
]

testers = [
    'vanilla_val',
    'learned_val',
]

variables = locals()
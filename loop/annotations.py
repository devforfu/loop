from torch.nn import Module
from typing import Callable, List, Optional

Activation = Callable
MaybeActivation = Optional[Activation]
ListOfModules = List[Module]

from .registry import Registry
from .train_modules.base_train_module import BaseTrainModule

MODEL_REGISTRY = Registry()
MODEL_REGISTRY.register("BaseTrainModule")(BaseTrainModule)
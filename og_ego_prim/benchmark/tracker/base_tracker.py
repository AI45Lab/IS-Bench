from abc import ABC, abstractmethod


class EvalTracker(ABC):

    def __init__(self):
        self.task = None
        self.scene = None
        self.model = None
    
    @abstractmethod
    def save_tracking(self, save_path: str):
        pass

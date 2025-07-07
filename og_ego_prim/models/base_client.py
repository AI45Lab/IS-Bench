from abc import ABC, abstractmethod
from typing import List, Dict


class BaseClient(ABC):

    @abstractmethod
    def model(
        self, 
        prompt: str, 
        image_file: List[str] | str = None, 
        gen_args: Dict = None,
    ):
        pass

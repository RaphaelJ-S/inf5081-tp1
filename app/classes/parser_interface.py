from abc import ABC, abstractmethod


class Parser(ABC):

    @abstractmethod
    def parse(self, line: str):
        pass

    @abstractmethod
    def access_data(self) -> dict:
        pass

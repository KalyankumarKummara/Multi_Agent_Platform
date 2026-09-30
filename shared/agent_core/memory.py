from abc import ABC, abstractmethod
from typing import Any


class Memory(ABC):

    @abstractmethod
    def read(self, key: str) -> Any:
        pass

    @abstractmethod
    def write(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def search(self, query: str) -> list[Any]:
        pass

class InMemoryStore(Memory):

    def __init__(self):
        self._data: dict[str, Any] = {}

    def read(self, key: str) -> Any:
        return self._data.get(key)

    def write(self, key: str, value: Any) -> None:
        self._data[key] = value

    def search(self, query: str) -> list[Any]:
        results = []

        for value in self._data.values():
            if query.lower() in str(value).lower():
                results.append(value)

        return results
    
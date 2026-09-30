from enum import Enum


class AgentStatus(str, Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    ERROR = "error"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    STOPPED = "stopped"
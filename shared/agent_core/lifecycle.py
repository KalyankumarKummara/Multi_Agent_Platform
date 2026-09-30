from .state import AgentStatus


class AgentLifecycle:
    def __init__(self):
        self.status = AgentStatus.CREATED

    def initialize(self):
        self.status = AgentStatus.INITIALIZING

    def mark_ready(self):
        self.status = AgentStatus.READY

    def start(self):
        self.status = AgentStatus.RUNNING

    def wait(self):
        self.status = AgentStatus.WAITING

    def error(self):
        self.status = AgentStatus.ERROR

    def recover(self):
        self.status = AgentStatus.RECOVERING

    def stop(self):
        self.status = AgentStatus.STOPPING

    def stopped(self):
        self.status = AgentStatus.STOPPED
# todo spostare la classe in OD_Pair

class Agent:
    def __init__(self, agent_id, src, dst):
        self.id = agent_id
        self.src = src
        self.dst = dst
        self.path = None
        self.delay = 0

    def __str__(self):
        return f"Agent {self.id}, delay = {self.delay}  ->  {self.path}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Agent) and self.id == other.id
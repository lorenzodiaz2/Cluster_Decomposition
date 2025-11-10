class Path:
    def __init__(self, visits):
        self.visits = visits

    def __str__(self):
        return self.visits.__str__()

    def compare(self, other) -> int:
        return sum(self.visits[t] == other.visits[t] for t in range(min(len(self.visits), len(other.visits)))) # / min(len(other.nodes), len(self.nodes))

    def get_or_default(self, t):
        return self.visits[t] if t <= len(self.visits) - 1 else None
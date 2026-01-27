class Facility:
    def __init__(
        self,
        facility_id,
        position,
        capacity,
        activation_cost
    ) -> None:
        self.id = facility_id
        self.position = position
        self.capacity = capacity
        self.activation_cost = activation_cost
        self.is_open = False
        self.shipment_by_client: dict[int, int] = {}  # id client -> quanto gli dò

    def __str__(self):
        return f"Facility {self.id}  position = {self.position}  capacity = {self.capacity}  opening cost = {self.activation_cost}"

    def __eq__(self, other):
        return isinstance(other, Facility) and self.id == other.id

    def __hash__(self):
        return hash(self.id)
from random import random, choice

from core.action import Action, Move, Obtain
from core.message import Message
from core.player import Player
from core.snapshots import HelperSurroundingsSnapshot
from core.views.cell_view import CellView


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return (abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5


class RandomPlayer(Player):
    def __init__(self, id: int, ark_x: int, ark_y: int):
        super().__init__(id, ark_x, ark_y)
        print(f"I am {self}")

    def check_surroundings(self, snapshot: HelperSurroundingsSnapshot):
        self.position = snapshot.position
        self.sight = snapshot.sight

        msg = snapshot.time_elapsed + self.id
        if not self.is_message_valid(msg):
            msg = msg & 0xFF

        return msg

    def get_my_cell(self) -> CellView:
        xcell, ycell = tuple(map(int, self.position))
        if not self.sight.cell_is_in_sight(xcell, ycell):
            raise Exception(f"{self} failed to find own cell")

        return self.sight.get_cellview_at(xcell, ycell)

    def find_closest_animal(self) -> tuple[int, int] | None:
        closest_animal = None
        closest_dist = -1
        closest_pos = None
        for cellview in self.sight:
            if len(cellview.animals) > 0:
                dist = distance(*self.position, cellview.x, cellview.y)
                if closest_animal is None or dist < closest_dist:
                    closest_animal = choice(tuple(cellview.animals))
                    closest_dist = dist
                    closest_pos = (cellview.x, cellview.y)

        return closest_pos

    def get_random_move(self) -> tuple[float, float]:
        old_x, old_y = self.position
        dx, dy = random() - 0.5, random() - 0.5

        while not (self.can_move_to(old_x + dx, old_y + dy)):
            dx, dy = random() - 0.5, random() - 0.5

        return old_x + dx, old_y + dy

    def get_action(self, messages: list[Message]) -> Action:
        # for msg in messages:
        #     print(f"{self.id}: got {msg.contents} from {msg.from_helper.id}")

        if not self.is_flock_empty():
            return Move(*self.move_towards(*self.ark_position))

        cellview = self.get_my_cell()

        if len(cellview.animals) > 0:
            random_animal = choice(tuple(cellview.animals))
            return Obtain(random_animal)

        closest_animal = self.find_closest_animal()
        if closest_animal:
            return Move(*self.move_towards(*closest_animal))

        return Move(*self.get_random_move())

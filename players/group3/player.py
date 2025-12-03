import math
from random import choice, random
from core.animal import Animal, Gender
from core.message import Message
from core.player import Player
from core.snapshots import HelperSurroundingsSnapshot
from core.views.cell_view import CellView
from core.views.player_view import Kind
from core.action import Action, Move, Obtain, Release
import core.constants as c


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return (abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5


class Player3(Player):
    def __init__(
        self,
        id: int,
        ark_x: int,
        ark_y: int,
        kind: Kind,
        num_helpers: int,
        species_populations: dict[str, int],
    ):
        super().__init__(id, ark_x, ark_y, kind, num_helpers, species_populations)
        self.ark_species: set[Animal] = set()
        self.is_raining = False
        self.hellos_received = []
        # self.angle = math.radians(random() * 360)
        samples, total_weight = self.angle_weights()
        self.angle = self.find_angle(samples, total_weight)
        self.cooldowns = {}
        self.days_remaining = 1008
        self.next_move = 0

    def check_surroundings(self, snapshot: HelperSurroundingsSnapshot) -> int:
        self.position = snapshot.position
        self.flock = snapshot.flock
        self.update_ark_memory(snapshot)
        for animal_id, cooldown in self.cooldowns.items():
            if cooldown > 0:
                self.cooldowns[animal_id] = cooldown - 1
        for animal_id in list(self.cooldowns.keys()):
            if self.cooldowns[animal_id] == 0:
                del self.cooldowns[animal_id]

        self.sight = snapshot.sight
        self.is_raining = snapshot.is_raining

        # if I didn't receive any messages, broadcast "hello"
        # a "hello" message is when a player's id bit is set
        if len(self.hellos_received) == 0:
            msg = 1 << (self.id % 8)
        else:
            # else, acknowledge all "hello"'s I got last turn
            # do this with a bitwise OR of all IDs I got
            msg = 0
            for hello in self.hellos_received:
                msg |= hello
            self.hellos_received = []

        if not self.is_message_valid(msg):
            msg = msg & 0xFF

        return msg

    def get_action(self, messages: list[Message]) -> Action | None:
        for msg in messages:
            if 1 << (msg.from_helper.id % 8) == msg.contents:
                self.hellos_received.append(msg.contents)
        # noah shouldn't do anything
        if self.kind == Kind.Noah:
            return None

        # If it's raining, go to ark
        if self.is_raining:
            self.days_remaining -= 1
            if (
                distance(*self.position, self.ark_position[0], self.ark_position[1])
                >= self.days_remaining - 12
            ):
                # move towards ark if we need a lot of steps to get to ark
                # otherwise keep searching
                if self.next_move > 0:
                    self.next_move -= 1
                return Move(*self.move_towards(*self.ark_position))

        if self.next_move > 0:
            self.next_move -= 1
            return Move(*self.move_dir())
        # If I am holding an animal that exists in my ark memory, drop it and add a cooldown
        ark_memory_info = set()
        for animal in self.ark_species or []:
            ark_memory_info.add((animal.species_id, animal.gender))
        for animal in self.flock:
            if (animal.species_id, animal.gender) in ark_memory_info:
                self.cooldowns[animal.species_id] = 20  # e.g., 5 turns cooldown
                self.angle = math.radians(random() * 360)
                self.next_move = 5
                return Release(animal)  # Drop the animal

        # if self.is_flock_full():
        #     return Move(*self.move_towards(*self.ark_position))

        # If I have obtained an animal, go to ark
        # if not self.is_flock_empty():
        if self.is_flock_full():
            # if len(self.flock) >= 4:
            return Move(*self.move_towards(*self.ark_position))

        # If I've reached an animal, I'll obtain it
        """ cellview = self._get_my_cell()
        if len(cellview.animals) > 0:
            # This means the random_player will even attempt to
            # (unsuccessfully) obtain animals in other helpers' flocks
            random_animal = choice(tuple(cellview.animals))
            return Obtain(random_animal) """

        # don't move too far from the ark
        if distance(*self.position, self.ark_position[0], self.ark_position[1]) >= 997:
            self.angle = math.radians(random() * 360)
            # print("distance too far")
            return Move(*self.move_towards(*self.ark_position))

        cellview = self._get_my_cell()
        cellview.animals
        # grab an animal that does not appear to be in flock or in the ark
        free_animals = self.get_free_animals_in_cell(cellview)
        if len(free_animals) > 0:
            animal_to_obtain = choice(tuple(free_animals))
            print(f"Helper {self.id}: Obtained animal into flock")
            print(f"Helper {self.id}: New flock size: {len(self.flock) + 1}")
            return Obtain(animal_to_obtain)

        # If I see any animals, I'll chase the closest one
        closest_animal = self._find_closest_desirable_animal()
        if closest_animal:
            dist_animal = distance(*self.position, *closest_animal)
            if dist_animal > 0.01:
                # print(f"Helper {self.id}: Moving towards animal")
                # This means the random_player will even approach
                # animals in other helpers' flocks
                return Move(*self.move_towards(*closest_animal))

        return Move(*self.move_dir())

    def get_distance(self, from_x, from_y, to_x, to_y):
        print(math.sqrt((to_x - from_x) ** 2 + (to_y - from_y) ** 2))
        return math.sqrt((to_x - from_x) ** 2 + (to_y - from_y) ** 2)

    def _get_my_cell(self) -> CellView:
        xcell, ycell = tuple(map(int, self.position))
        if not self.sight.cell_is_in_sight(xcell, ycell):
            raise Exception(f"{self} failed to find own cell")

        return self.sight.get_cellview_at(xcell, ycell)

    def _find_closest_desirable_animal(self) -> tuple[int, int] | None:
        cur_flock_animal_types = set()
        for animal in self.flock:
            cur_flock_animal_types.add(animal.species_id)
        closest_animal = None
        closest_dist = -1
        closest_pos = None
        for cellview in self.sight:
            if len(cellview.animals) > 0 and not self.is_animal_likely_in_flock(
                cellview
            ):
                dist = distance(*self.position, cellview.x, cellview.y)
                if closest_animal is None or dist < closest_dist:
                    desirable_animals = []
                    for animal in cellview.animals:
                        if not self.should_pursue_animal(animal):  # type: ignore
                            print(
                                f"Not pursuing animal {animal.species_id} as both genders are already in ark."
                            )
                            continue
                        if animal.species_id in cur_flock_animal_types:
                            print(
                                f"Not pursuing animal {animal.species_id} as it's already in flock."
                            )
                            continue
                        # print(f"Helper {self.id}: Considering animal {animal.species_id} at cell ({cellview.x}, {cellview.y})")
                        desirable_animals.append(animal)
                    # closest_animal = choice(tuple(cellview.animals))
                    if len(desirable_animals) == 0:
                        continue
                    closest_animal = choice(tuple(desirable_animals))
                    closest_dist = dist
                    closest_pos = (cellview.x, cellview.y)

        return closest_pos

    def move_dir(self) -> tuple[float, float]:
        step_size = c.MAX_DISTANCE_KM * 0.99
        x0, y0 = self.position
        x1, y1 = (
            x0 + step_size * math.cos(self.angle),
            y0 + step_size * math.sin(self.angle),
        )
        if self.can_move_to(x1, y1):
            # print(x1, y1)
            return x1, y1
        # print("move away")
        self.angle = math.radians(random() * 360)
        return x0, y0

    def is_animal_likely_in_flock(self, cellview: CellView) -> bool:
        """
        Check if animals in a cell are likely already in another helper's flock.

        Returns True if there are helpers present in the cell (excluding self),
        which indicates the animals are probably in those helpers' flocks.

        Args:
            cellview: The cell to check

        Returns:
            True if animals in this cell are likely carried by other helpers
        """
        # If there are no animals in the cell, return False
        if len(cellview.animals) == 0:
            return False

        # Check if there are any helpers in this cell besides myself
        for helper_view in cellview.helpers:
            # Exclude myself (Noah has id=0, other helpers have id > 0)
            if helper_view.id != self.id:
                # There's another helper here, so animals are likely in their flock
                return True

        # No other helpers in this cell, animals are free to obtain
        return False

    def get_free_animals_in_cell(self, cellview: CellView) -> set[Animal]:
        """
        Get animals in a cell that are NOT likely in another helper's flock.

        Returns animals that:
        1. Are not already in the ark
        2. Are not in my own flock
        3. Are not in a cell with other helpers (likely in their flock)
        4. Are not on cooldown from being recently dropped

        Args:
            cellview: The cell to check

        Returns:
            Set of animals that are free to obtain
        """
        if self.is_animal_likely_in_flock(cellview):
            return set()

        # Filter out animals already in ark or in my flock and those on cooldown
        free_animals = set()
        for animal in cellview.animals:
            if animal not in self.ark_species and animal not in self.flock:
                if animal.species_id not in self.cooldowns:
                    free_animals.add(animal)
                else:
                    print(f"Animal {animal.species_id} skipped: on cooldown.")

        return free_animals

    def update_ark_memory(self, snapshot: HelperSurroundingsSnapshot) -> None:
        """Update our memory of animals on the ark"""
        # If no ark view, do nothing
        if snapshot.ark_view is None:
            return None

        # Update memory
        self.ark_species = snapshot.ark_view.animals.copy()
        # print(f"Ark memory updated: {len(self.ark_species)} animals remembered.")

    def should_pursue_animal(self, animal: Animal) -> bool:
        """Decide whether to pursue a given animal based on whether it is already in the ark."""
        ark_animals_with_gender: set[tuple[int, Gender]] = set()
        for animal in self.ark_species or []:
            ark_animals_with_gender.add((animal.species_id, animal.gender))

        if (animal.species_id, Gender.Male) in ark_animals_with_gender and (
            animal.species_id,
            Gender.Female,
        ) in ark_animals_with_gender:
            return False  # Both
        if animal.species_id in self.cooldowns:
            print(f"Animal {animal.species_id} skipped: on cooldown.")
            return False  # On cooldown
        return True

    def angle_weights(self):
        num_samples = 360
        samples = []
        cumu = 0.0
        for i in range(0, num_samples):
            theta = 2 * math.pi * i / num_samples
            d = self.max_distance_to_boundary(theta)
            cumu = cumu + d
            samples.append((theta, cumu))
        tot_wt = cumu
        return samples, tot_wt

    def max_distance_to_boundary(self, theta):
        ark_x = self.ark_position[0]
        ark_y = self.ark_position[1]
        dx = math.cos(theta)
        dy = math.sin(theta)

        t_list = []
        if dx > 0:
            t_right = (c.X - ark_x) / dx
            t_list.append(t_right)
        elif dx < 0:
            t_left = (0 - ark_x) / dx
            t_list.append(t_left)

        if dy > 0:
            t_top = (c.Y - ark_y) / dy
            t_list.append(t_top)
        elif dy < 0:
            t_bottom = (0 - ark_y) / dy
            t_list.append(t_bottom)

        if len(t_list) == 0:
            return 0
        return min(min(t_list), 1008)

    def find_angle_for_target(self, samples, target):
        left = 0
        right = len(samples) - 1
        while left < right:
            mid = (left + right) // 2
            if samples[mid][1] >= target:
                right = mid
            else:
                left = mid + 1

        return samples[left][0]

    def find_angle(self, samples, total_weight):
        if self.kind == Kind.Noah:
            return -100
        k = self.id - 1
        print(k)
        target = total_weight * ((float(k) + 0.5) / float(self.num_helpers - 1))
        print(target)
        theta = self.find_angle_for_target(samples, target)
        print(theta)
        return theta

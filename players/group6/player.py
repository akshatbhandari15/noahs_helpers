from core.player import Player
from core.snapshots import HelperSurroundingsSnapshot
from core.action import Move, Obtain
from core.views.player_view import Kind
from core.animal import Animal, Gender
import math
from random import random
import logging

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Grid dimensions
GRID_WIDTH = 1000
GRID_HEIGHT = 1000

# Debug flag: set to True to enable debug print output
DEBUG = False

# Noah broadcast configuration
# N determines the probability split:
# HAVE messages: (N-1)/N probability
# NEED messages: 1/N probability
# Example: N=3 means HAVE 67%, NEED 33%
NOAH_BROADCAST_RATIO_N = 3
NOAH_BROADCAST_INTERVAL = 10  # Broadcast every N turns

# Patrol configuration
PATROL_SPACING = 10

# Priority calculation constants
PRIORITY_MULTIPLIER_NONE = 0.5  # Species not on ark at all
PRIORITY_MULTIPLIER_INCOMPLETE = 0.1  # Species missing one gender
PRIORITY_MULTIPLIER_COMPLETE = 10.0  # Species already complete
PRIORITY_NOAH_BOOST = 0.5  # Multiplier when Noah prioritizes
PRIORITY_PURSUIT_PENALTY = 0.2  # Penalty per helper pursuing same species
DEFAULT_POPULATION = 10  # Default if species not in populations dict

# Signal encoding constants
SIGNAL_TYPE_HAVE = 0
SIGNAL_TYPE_NEED = 1

# ============================================================================
# SAFETY CONFIGURATION
# ============================================================================
MAX_CHASE_ATTEMPTS = 10
CHASE_COOLDOWN_TURNS = 40  # or whatever you want
MAX_SAFE_DISTANCE = 1008  # Maximum turns we can be from ark
SAFETY_MARGIN = 50  # Extra safety buffer
VISION_RADIUS = 5  # helper sight radius

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)
noah_logger = logging.getLogger(f"{__name__}.noah")

if not logger.handlers:
    log_level = logging.DEBUG if DEBUG else logging.WARNING
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

if not noah_logger.handlers:
    log_level = logging.DEBUG if DEBUG else logging.WARNING
    noah_logger.setLevel(log_level)
    noah_handler = logging.StreamHandler()
    noah_handler.setLevel(log_level)
    noah_formatter = logging.Formatter("NOAH: %(message)s")
    noah_handler.setFormatter(noah_formatter)
    noah_logger.addHandler(noah_handler)

# ============================================================================
# SHARED STATE (Minimal - Only for coordination)
# ============================================================================

_PATROL_STRIPS: list[dict] = []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _gender_to_string(gender: Gender) -> str:
    """Convert Gender enum to string representation."""
    return "Female" if gender == Gender.Female else "Male"


def _encode_noah_message(
    species_id: int,
    gender: Gender,
    signal_type: int,
    species_to_top64_id: dict[int, int],
) -> int:
    """Encode a Noah broadcast message.

    Encoding: Bit 0=TYPE, Bit 1=GENDER, Bits 2-7=SPECIES
    """
    top64_id = species_to_top64_id[species_id]
    gender_bit = 1 if gender == Gender.Female else 0
    return (top64_id << 2) | (gender_bit << 1) | signal_type


def _build_verified_status(animals) -> dict[int, dict]:
    """Build verified status dictionary from iterable of animals."""
    verified_status: dict[int, dict] = {}
    for animal in animals:
        if animal.species_id not in verified_status:
            verified_status[animal.species_id] = {"male": False, "female": False}
        if animal.gender == Gender.Male:
            verified_status[animal.species_id]["male"] = True
        elif animal.gender == Gender.Female:
            verified_status[animal.species_id]["female"] = True
    return verified_status


# ============================================================================
# PLAYER CLASS
# ============================================================================


class Player6(Player):
    def __init__(
        self,
        id: int,
        ark_x: int,
        ark_y: int,
        kind: Kind,
        num_helpers,
        species_populations: dict[str, int],
    ):
        super().__init__(id, ark_x, ark_y, kind, num_helpers, species_populations)

        # Store species populations for priority calculations
        self._species_populations = species_populations
        self._species_id_populations: dict[int, int] = {}
        for species_name, pop in species_populations.items():
            species_id = self._extract_species_id(species_name)
            if species_id is not None:
                self._species_id_populations[species_id] = pop

        # Deterministically map the 64 rarest species to IDs 0-63
        sorted_species = sorted(
            self._species_id_populations.items(), key=lambda x: (x[1], x[0])
        )
        self.top_64_species = [s_id for s_id, _ in sorted_species[:64]]
        self.species_to_top64_id = {
            s_id: i for i, s_id in enumerate(self.top_64_species)
        }
        self.top64_id_to_species = {
            i: s_id for i, s_id in enumerate(self.top_64_species)
        }

        # Initialize per-player ark beliefs (NO GLOBAL STATE)
        self.ark_beliefs: dict[int, dict] = {}
        for species_id in self.top_64_species:
            self.ark_beliefs[species_id] = {"male": False, "female": False}

        # Noah-specific initialization
        if kind == Kind.Noah:
            self.last_broadcast_turn = -1
            self.broadcast_queue_index = 0
            self.completed_have_index = 0

        # Helper-specific initialization
        if kind == Kind.Helper:
            self._patrol_spacing = PATROL_SPACING
            strip_index = self._claim_patrol_strip(id, num_helpers)
            self._setup_patrol_parameters(id, strip_index)

            # Track what other helpers are pursuing (from broadcasts)
            self._noah_priority_species: int | None = None

            # Store ids of animals that appear to be in other helpers' flocks
            self._animals_seen_in_flocks: set[int] = set()
            self._max_distance_from_ark = 0
            self._rain_started = False
            self.snapshot: HelperSurroundingsSnapshot | None = None
            self._cluster_claim_winners: dict[tuple[int, int], int] = {}
            self._current_claim: tuple[int, int] | None = None
            self._current_target_key: tuple[int, int, int] | None = None
            self._current_target_turns: int = 0
            # map from (species_id, col, row) -> turn_until_which_we_should_ignore
            self._banned_targets: dict[tuple[int, int, int], int] = {}
            self._chase_attempts_by_key: dict[tuple[int, int, int], int] = {}
            self._covered_y_intervals: list[tuple[int, int]] = []

    def _update_animals_seen_in_flocks(
        self, snapshot: HelperSurroundingsSnapshot
    ) -> None:
        """Mark animals that appear to be in other helpers' flocks.

        Heuristic: any animals in a cell that contains at least one other helper
        (id != self.id) are treated as that helper's flock animals.
        """
        self._animals_seen_in_flocks.clear()
        if not snapshot or not hasattr(snapshot, "sight"):
            return

        for cellview in snapshot.sight:
            if not getattr(cellview, "helpers", None):
                continue

            # If there is any other helper in this cell, treat this cell's animals as flock
            if any(helper.id != self.id for helper in cellview.helpers):
                for animal in cellview.animals:
                    self._animals_seen_in_flocks.add(id(animal))

    def _make_target_key(self, animal: Animal, tx: int, ty: int):
        """Stable key for a chase target: species, gender, and cluster.

        This stays the same as long as the animal is in the same 16x8 cluster.
        """
        _, (col, row) = self._encode_cluster(tx, ty)
        return (animal.species_id, col, row)

    def _extract_species_id(self, species_name) -> int | None:
        """Extract species_id from species name."""
        if isinstance(species_name, int):
            return species_name
        elif len(species_name) == 1 and species_name.isalpha():
            return ord(species_name.lower()) - ord("a")
        elif "_" in species_name:
            return int(species_name.split("_")[-1])
        else:
            try:
                return int(species_name)
            except ValueError:
                return None

    def _encode_cluster(self, x: float, y: float) -> tuple[int, tuple[int, int]]:
        """Quantize (x, y) position into a 7-bit cluster id (16x8 grid)."""
        col = int(x // (GRID_WIDTH / 16))
        row = int(y // (GRID_HEIGHT / 8))
        col = max(0, min(15, col))
        row = max(0, min(7, row))
        bits = (col << 3) | row  # 4 bits col, 3 bits row
        return bits, (col, row)

    def _decode_cluster(self, bits: int) -> tuple[int, int]:
        """Decode 7-bit cluster into (col, row)."""
        col = (bits >> 3) & 0x0F
        row = bits & 0x07
        return col, row

    # ========================================================================
    # PATROL STRIP MANAGEMENT
    # ========================================================================

    def _initialize_global_patrol_strips(self, num_helpers: int) -> None:
        """Initialize patrol strips - divide grid into vertical sections."""
        global _PATROL_STRIPS
        if len(_PATROL_STRIPS) > 0:
            return

        ark_y = self.ark_position[1]

        # Divide helpers between regions above and below ark
        # Make sure both regions have at least 1 helper if possible
        helpers_above = max(1, int(round(num_helpers * ark_y / GRID_HEIGHT)))
        helpers_below = max(0, num_helpers - helpers_above)

        # If we have 0 helpers below, redistribute
        if helpers_below == 0 and num_helpers > 1:
            helpers_above = num_helpers // 2
            helpers_below = num_helpers - helpers_above

        # Create strips for above ark region
        if helpers_above > 0:
            strip_width_above = GRID_WIDTH / helpers_above
            for i in range(helpers_above):
                x_min = int(i * strip_width_above)
                x_max = (
                    int((i + 1) * strip_width_above - 1)
                    if i < helpers_above - 1
                    else GRID_WIDTH - 1
                )
                _PATROL_STRIPS.append(
                    {
                        "x_min": x_min,
                        "x_max": x_max,
                        "owner": i,
                        "done": False,
                        "region": "above",
                    }
                )

        # Create strips for below ark region
        if helpers_below > 0:
            strip_width_below = GRID_WIDTH / helpers_below
            for i in range(helpers_below):
                x_min = int(i * strip_width_below)
                x_max = (
                    int((i + 1) * strip_width_below - 1)
                    if i < helpers_below - 1
                    else GRID_WIDTH - 1
                )
                _PATROL_STRIPS.append(
                    {
                        "x_min": x_min,
                        "x_max": x_max,
                        "owner": helpers_above + i,
                        "done": False,
                        "region": "below",
                    }
                )

        logger.info(
            f"Created {len(_PATROL_STRIPS)} patrol strips: "
            f"{helpers_above} above ark, {helpers_below} below ark"
        )

    def _claim_patrol_strip(self, helper_id: int, num_helpers: int) -> int:
        global _PATROL_STRIPS
        if len(_PATROL_STRIPS) == 0:
            self._initialize_global_patrol_strips(num_helpers)

        # Assume: Noah is id 0, helpers are 1..num_helpers
        # Map helper 1 -> strip 0, helper 2 -> strip 1, etc.
        helper_index = max(0, helper_id - 1)
        return helper_index % len(_PATROL_STRIPS)

    def _setup_patrol_parameters(self, helper_id: int, strip_index: int) -> None:
        strip = _PATROL_STRIPS[strip_index]
        self._patrol_strip_index = strip_index
        self._patrol_x_min = strip["x_min"]
        self._patrol_x_max = strip["x_max"]

        ark_y = self.ark_position[1]

        if strip.get("region") == "above":
            helpers_above = sum(1 for s in _PATROL_STRIPS if s.get("region") == "above")
            helper_index_in_region = helper_id
            rows_in_region = max(1, ark_y)
            row_spacing = max(1, rows_in_region // max(1, helpers_above))
            self._patrol_row = (helper_index_in_region * row_spacing) % rows_in_region
            self._patrol_row_step = self._patrol_spacing
            self._patrol_max_row = ark_y
        else:
            helpers_above = sum(1 for s in _PATROL_STRIPS if s.get("region") == "above")
            helper_index_in_region = helper_id - helpers_above
            bottom_space = GRID_HEIGHT - ark_y
            rows_in_region = max(1, bottom_space)
            row_spacing = max(
                1, rows_in_region // max(1, len(_PATROL_STRIPS) - helpers_above)
            )
            self._patrol_row = (
                ark_y + (helper_index_in_region * row_spacing) % rows_in_region
            )
            self._patrol_row_step = self._patrol_spacing
            self._patrol_max_row = GRID_HEIGHT

        self._patrol_dir = self._patrol_strip_index % 2 == 0
        self._patrol_active = True

    # ========================================================================
    # COMMUNICATION
    # ========================================================================

    def check_surroundings(self, snapshot: HelperSurroundingsSnapshot) -> int:
        self._update_snapshot(snapshot)

        # Update what animals we see in other helpers' flocks (no global state)
        if self.kind == Kind.Helper:
            self._update_safety_tracking(snapshot)
            self._update_animals_seen_in_flocks(snapshot)

        if self.kind == Kind.Noah:
            return self._noah_broadcast()
        else:
            return self._helper_broadcast()

    def _update_snapshot(self, snapshot: HelperSurroundingsSnapshot) -> None:
        self.position = snapshot.position
        self.flock = snapshot.flock
        self.snapshot = snapshot

        # Update ark beliefs when at ark
        if self._at_ark():
            self._update_ark_beliefs_from_ark(snapshot)

    def _at_ark(self) -> bool:
        """Check if at the ark."""
        return (int(self.position[0]), int(self.position[1])) == self.ark_position

    def _update_ark_beliefs_from_ark(
        self, snapshot: HelperSurroundingsSnapshot
    ) -> None:
        """Update ark beliefs when visiting the ark."""
        ark_animals = snapshot.sight.get_cellview_at(*self.ark_position).animals

        for animal in ark_animals:
            if animal.species_id not in self.ark_beliefs:
                self.ark_beliefs[animal.species_id] = {"male": False, "female": False}
            if animal.gender == Gender.Male:
                self.ark_beliefs[animal.species_id]["male"] = True
            elif animal.gender == Gender.Female:
                self.ark_beliefs[animal.species_id]["female"] = True

    # ========================================================================
    # NOAH BROADCASTING
    # ========================================================================

    def _noah_broadcast(self) -> int:
        """Noah broadcasts confirmations and needs using HAVE/NEED protocol."""
        if not self.snapshot:
            noah_logger.debug("No snapshot available")
            return 0

        snapshot = self.snapshot

        # Get actual ark animals
        if snapshot.ark_view is not None:
            actual_ark_animals = snapshot.ark_view.animals
        elif hasattr(snapshot, "sight"):
            ark_cellview = snapshot.sight.get_cellview_at(*self.ark_position)
            actual_ark_animals = ark_cellview.animals
        else:
            return 0

        # Update ark beliefs from actual animals
        for animal in actual_ark_animals:
            if animal.species_id not in self.ark_beliefs:
                self.ark_beliefs[animal.species_id] = {"male": False, "female": False}
            if animal.gender == Gender.Male:
                self.ark_beliefs[animal.species_id]["male"] = True
            elif animal.gender == Gender.Female:
                self.ark_beliefs[animal.species_id]["female"] = True

        # Rate limit: broadcast every N turns
        current_turn = snapshot.time_elapsed if hasattr(snapshot, "time_elapsed") else 0
        if current_turn - self.last_broadcast_turn < NOAH_BROADCAST_INTERVAL:
            return 0
        self.last_broadcast_turn = current_turn

        # If no animals at ark, don't broadcast
        if not actual_ark_animals:
            # noah_logger.debug("No animals at ark")
            return 0

        # Build verified status
        verified_status = _build_verified_status(actual_ark_animals)

        # Find first incomplete species
        first_incomplete_index = None
        first_incomplete_species_id = None
        first_incomplete_missing_genders = []

        for idx, species_id in enumerate(self.top_64_species):
            status = verified_status.get(species_id, {"male": False, "female": False})
            if not (status["male"] and status["female"]):
                first_incomplete_index = idx
                first_incomplete_species_id = species_id
                if not status["male"]:
                    first_incomplete_missing_genders.append(Gender.Male)
                if not status["female"]:
                    first_incomplete_missing_genders.append(Gender.Female)
                break

        # If all complete, broadcast HAVE about any animal
        if first_incomplete_index is None:
            return self._broadcast_have_from_animals(actual_ark_animals)

        # Probabilistic selection: HAVE (N-1)/N, NEED 1/N
        broadcast_have = (current_turn % NOAH_BROADCAST_RATIO_N) < (
            NOAH_BROADCAST_RATIO_N - 1
        )

        if broadcast_have:
            # Broadcast HAVE about any animal at ark
            have_candidates = [
                (animal.species_id, animal.gender)
                for animal in actual_ark_animals
                if animal.species_id in self.species_to_top64_id
            ]

            if have_candidates:
                self.completed_have_index %= len(have_candidates)
                target_species_id, target_gender = have_candidates[
                    self.completed_have_index
                ]
                self.completed_have_index = (self.completed_have_index + 1) % len(
                    have_candidates
                )

                message = _encode_noah_message(
                    target_species_id,
                    target_gender,
                    SIGNAL_TYPE_HAVE,
                    self.species_to_top64_id,
                )
                noah_logger.info(
                    f"HAVE: species_{target_species_id} ({_gender_to_string(target_gender)})"
                )
                return message
            else:
                return self._broadcast_have_from_animals(actual_ark_animals)

        # Broadcast NEED
        if first_incomplete_missing_genders and first_incomplete_species_id is not None:
            # Check if all rarer species are complete
            all_rarer_complete = all(
                verified_status.get(
                    self.top_64_species[idx], {"male": False, "female": False}
                ).get("male", False)
                and verified_status.get(
                    self.top_64_species[idx], {"male": False, "female": False}
                ).get("female", False)
                for idx in range(first_incomplete_index)
            )

            if all_rarer_complete:
                self.broadcast_queue_index %= len(first_incomplete_missing_genders)
                target_gender = first_incomplete_missing_genders[
                    self.broadcast_queue_index
                ]
                self.broadcast_queue_index = (self.broadcast_queue_index + 1) % len(
                    first_incomplete_missing_genders
                )

                message = _encode_noah_message(
                    first_incomplete_species_id,
                    target_gender,
                    SIGNAL_TYPE_NEED,
                    self.species_to_top64_id,
                )
                noah_logger.info(
                    f"NEED: species_{first_incomplete_species_id} ({_gender_to_string(target_gender)})"
                )
                return message

        # Fallback to HAVE
        return self._broadcast_have_from_animals(actual_ark_animals)

    def _broadcast_have_from_animals(self, animals) -> int:
        """Broadcast HAVE message from first available top 64 animal."""
        for animal in animals:
            if animal.species_id in self.species_to_top64_id:
                message = _encode_noah_message(
                    animal.species_id,
                    animal.gender,
                    SIGNAL_TYPE_HAVE,
                    self.species_to_top64_id,
                )
                noah_logger.info(
                    f"HAVE: species_{animal.species_id} ({_gender_to_string(animal.gender)})"
                )
                return message
        return 0

    # ========================================================================
    # HELPER BROADCASTING
    # ========================================================================

    def _helper_broadcast(self) -> int:
        """Helper broadcasts which region it is claiming to hunt in.

        Format (helpers only):
        bit 7: 1 if claiming, 0 = idle
        bits 6-0: 7-bit cluster id (16x8 grid over 1000x1000)
        """
        # If we're not currently chasing anything, no claim
        if self._current_claim is None:
            return 0

        col, row = self._current_claim
        cluster_bits = (col << 3) | row
        return 0x80 | cluster_bits  # top bit = claim flag

    # ========================================================================
    # ACTION SELECTION
    # ========================================================================

    def get_action(self, messages) -> Move | Obtain | None:
        if self.kind == Kind.Noah:
            return None

        self._process_messages(messages)
        self._current_claim = None

        if self._should_return_to_ark():
            return self._return_to_ark()

        obtain_action = self._try_obtain_at_current_position()
        if obtain_action:
            return obtain_action

        exploration_budget = self._can_explore_more()
        if exploration_budget > 0:
            chase_action = self._try_chase_nearby_animal()
            if chase_action:
                # Make sure chasing won't take us too far
                # (this is a rough check - could be more sophisticated)
                return chase_action

        # Continue patrol if we have exploration budget
        if exploration_budget > 5:  # Need at least 5 turns buffer for patrol
            return self._patrol_for_animals()

        return self._return_to_ark()

    def _process_messages(self, messages) -> None:
        """Process communication signals from Noah and other helpers."""
        self._cluster_claim_winners.clear()

        for message in messages:
            sender_id = message.from_helper.id
            signal = message.contents

            if signal == 0:
                continue

            is_noah = sender_id == 0

            if is_noah:
                # Noah uses: Bit 0=TYPE, Bit 1=GENDER, Bits 2-7=SPECIES
                signal_type = signal & 1
                gender_bit = (signal >> 1) & 1
                top64_id = (signal >> 2) & 0x3F

                if top64_id not in self.top64_id_to_species:
                    continue

                species_id = self.top64_id_to_species[top64_id]

                if species_id not in self.ark_beliefs:
                    self.ark_beliefs[species_id] = {"male": False, "female": False}

                if signal_type == SIGNAL_TYPE_HAVE:
                    if gender_bit == 0:
                        self.ark_beliefs[species_id]["male"] = True
                    else:
                        self.ark_beliefs[species_id]["female"] = True
                elif signal_type == SIGNAL_TYPE_NEED:
                    if gender_bit == 0:
                        self.ark_beliefs[species_id]["male"] = False
                    else:
                        self.ark_beliefs[species_id]["female"] = False

                    # NEED implies we HAVE all more common species
                    if species_id in self.top_64_species:
                        species_index = self.top_64_species.index(species_id)
                        for idx in range(species_index + 1, len(self.top_64_species)):
                            more_common_species_id = self.top_64_species[idx]
                            if more_common_species_id not in self.ark_beliefs:
                                self.ark_beliefs[more_common_species_id] = {
                                    "male": False,
                                    "female": False,
                                }
                            self.ark_beliefs[more_common_species_id]["male"] = True
                            self.ark_beliefs[more_common_species_id]["female"] = True

                    self._noah_priority_species = species_id
            else:
                # Helper messages: bit 7 = claim flag, bits 6-0 = cluster
                if signal == 0:
                    continue

                is_claim = (signal & 0x80) != 0
                if not is_claim:
                    continue

                cluster_bits = signal & 0x7F
                col, row = self._decode_cluster(cluster_bits)
                key = (col, row)

                # Election rule: smallest helper id wins the cluster
                winner = self._cluster_claim_winners.get(key)
                if winner is None or sender_id < winner:
                    self._cluster_claim_winners[key] = sender_id

    def _should_return_to_ark(self) -> bool:
        """Check if helper should return to ark."""

        # Always return if flock full
        if self.is_flock_full():
            return True

        current_distance = self._calculate_distance_to_ark()

        # BEFORE RAIN: only enforce a max radius
        if not self._rain_started:
            if current_distance >= MAX_SAFE_DISTANCE - SAFETY_MARGIN:
                logger.info(
                    f"[Helper {self.id}] Too far from ark before rain "
                    f"({current_distance:.1f}), returning"
                )
                return True
            return False

        # AFTER RAIN: use time+distance budget
        if self._can_explore_more() <= 0:
            logger.info(
                f"[Helper {self.id}] Budget exhausted after rain, "
                f"distance={current_distance:.1f}, returning"
            )
            return True

        return False

    def _return_to_ark(self) -> Move:
        """Return to ark."""
        # logger.debug(f"[Helper {self.id}] Returning to ark")
        return Move(*self.move_towards(*self.ark_position))

    # ========================================================================
    # OBTAINING ANIMALS
    # ========================================================================

    def _try_obtain_at_current_position(self) -> Obtain | None:
        """Try to obtain animal at current position - checks nearby cells."""
        if self.is_flock_full():
            return None

        cur_x, cur_y = self.position[0], self.position[1]

        # The helper is at float position (cur_x, cur_y)
        # Animals are located at integer cell coordinates
        # Check the cell we're in (using floor convention)
        cell_x = cur_x
        cell_y = cur_y

        sight = self.snapshot.sight
        # Get animals in current cell
        if sight.cell_is_in_sight(int(cell_x), int(cell_y)):
            cellview = sight.get_cellview_at(int(cell_x), int(cell_y))
            valid_animals = {a for a in cellview.animals if a.gender != Gender.Unknown}
            unclaimed_animals = self._get_unclaimed_animals(valid_animals)

            if not unclaimed_animals:
                return None
            best_animal = min(
                unclaimed_animals,
                key=lambda a: self._get_species_priority(a.species_id),
            )
            logger.debug(
                f"[Helper {self.id}] Obtaining species_{best_animal.species_id} "
                f"at position ({cur_x:.2f}, {cur_y:.2f}) in cell ({cell_x}, {cell_y})"
            )
            target_key = self._make_target_key(best_animal, int(cell_x), int(cell_y))
            self._banned_targets.pop(target_key, None)
            self._chase_attempts_by_key.pop(target_key, None)
            if self._current_target_key == target_key:
                self._current_target_key = None
                self._current_target_turns = 0
            return Obtain(best_animal)

    def _get_unclaimed_animals(self, animals: set[Animal]) -> set[Animal]:
        unclaimed = set()
        for animal in animals:
            if id(animal) in self._animals_seen_in_flocks:
                continue

            # Skip if we already have this species/gender
            if any(
                a.species_id == animal.species_id and a.gender == animal.gender
                for a in self.flock
            ):
                continue

            # NEW: use ark_beliefs instead of only ark_view
            status = self.ark_beliefs.get(animal.species_id)
            if status and status["male"] and status["female"]:
                # Ark already has a complete pair, don't bother
                continue

            unclaimed.add(animal)

        return unclaimed

    def _is_too_close_to_other_helper(self) -> bool:
        """Return True if we are very close to another helper and should yield."""
        if not self.snapshot or not hasattr(self.snapshot, "sight"):
            return False
        current_turn = getattr(self.snapshot, "time_elapsed", 0)

        for cellview in self.snapshot.sight:
            if not getattr(cellview, "helpers", None):
                continue
            for helper in cellview.helpers:
                if helper.id == self.id:
                    continue
                # Treat helper as located at cellview.x, cellview.y
                dist = math.sqrt(
                    (cellview.x - self.position[0]) ** 2
                    + (cellview.y - self.position[1]) ** 2
                )
                if dist < 1.5 and self.id > helper.id:
                    # We are the "loser" in this tie, so yield
                    if self._current_target_key is not None:
                        self._banned_targets[self._current_target_key] = (
                            current_turn + CHASE_COOLDOWN_TURNS
                        )
                    return True
        return False

    # ========================================================================
    # CHASING ANIMALS
    # ========================================================================

    def _try_chase_nearby_animal(self) -> Move | None:
        """Try to chase the closest unclaimed animal, coordinating via 1-byte claims.

        If we keep chasing essentially the same animal (same cell, species, gender)
        for more than 5 turns, give up on it and either choose a new one or
        fall back to patrolling.
        """

        if self._is_too_close_to_other_helper():
            self._current_target_key = None
            self._current_target_turns = 0
            self._current_claim = None
            return None

        candidates = self._find_chase_candidates()
        if not candidates:
            self._current_target_key = None
            self._current_target_turns = 0
            return None

        candidates.sort(
            key=lambda x: (self._get_species_priority(x[0].species_id), x[3])
        )

        current_turn = getattr(self.snapshot, "time_elapsed", 0)

        for target_animal, tx, ty, dist in candidates:
            if dist < 1.0:
                continue

            target_key = self._make_target_key(target_animal, tx, ty)

            # Which region does this animal belong to?
            cluster_bits, (col, row) = self._encode_cluster(tx, ty)
            key = (col, row)
            winner_id = self._cluster_claim_winners.get(key)
            if winner_id is not None and winner_id < self.id:
                continue

            # print(f"Helper {self.id}", self._current_target_key, target_key, self._current_target_turns)

            # How many total attempts have we made for this key (not just consecutively)?
            total_attempts = self._chase_attempts_by_key.get(target_key, 0) + 1

            if total_attempts > MAX_CHASE_ATTEMPTS:
                # Put this key on cooldown and do not chase this turn
                self._banned_targets[target_key] = current_turn + CHASE_COOLDOWN_TURNS
                self._chase_attempts_by_key[target_key] = total_attempts
                self._current_target_key = None
                self._current_target_turns = 0
                self._current_claim = None
                return None

            # Record the attempts and also keep the "consecutive" count for logging only
            self._chase_attempts_by_key[target_key] = total_attempts
            if self._current_target_key == target_key:
                self._current_target_turns += 1
            else:
                self._current_target_turns = 1

            self._current_target_key = target_key
            self._current_claim = (col, row)
            return Move(*self.move_towards(tx, ty))

        self._current_target_key = None
        self._current_target_turns = 0
        self._current_claim = None
        return None

    def _find_chase_candidates(self) -> list[tuple[Animal, int, int, float]]:
        """Find all unclaimed animals in sight."""
        candidates = []
        current_turn = getattr(self.snapshot, "time_elapsed", 0)

        for cellview in self.snapshot.sight:
            unclaimed = self._get_unclaimed_animals(cellview.animals)
            if unclaimed:
                dist = math.sqrt(
                    (cellview.x - self.position[0]) ** 2
                    + (cellview.y - self.position[1]) ** 2
                )
                for animal in unclaimed:
                    target_key = self._make_target_key(animal, cellview.x, cellview.y)

                    # If this (species,cluster) is on cooldown, skip it
                    cooldown_until = self._banned_targets.get(target_key)
                    if cooldown_until is not None and current_turn < cooldown_until:
                        continue

                    candidates.append((animal, cellview.x, cellview.y, dist))
        return candidates

    # ========================================================================
    # PRIORITY CALCULATION
    # ========================================================================

    def _get_species_priority(self, species_id: int) -> float:
        """Calculate priority for a species (lower = higher priority)."""
        population = self._species_id_populations.get(species_id, DEFAULT_POPULATION)
        status = self.ark_beliefs.get(species_id, {"male": False, "female": False})

        if not status["male"] and not status["female"]:
            priority = population * PRIORITY_MULTIPLIER_NONE
        elif not (status["male"] and status["female"]):
            priority = population * PRIORITY_MULTIPLIER_INCOMPLETE
        else:
            priority = population * PRIORITY_MULTIPLIER_COMPLETE

        if self._noah_priority_species == species_id:
            priority *= PRIORITY_NOAH_BOOST

        return priority

    # ========================================================================
    # PATROL
    # ========================================================================

    def _patrol_for_animals(self) -> Move:
        """Patrol the grid searching for animals."""
        # logger.debug(f"[Helper {self.id}] Patrolling")
        target = self._get_patrol_target()
        if target:
            # Use Euclidean distance consistently
            target_distance = math.sqrt(
                (target[0] - self.ark_position[0]) ** 2
                + (target[1] - self.ark_position[1]) ** 2
            )

            if target_distance >= MAX_SAFE_DISTANCE - SAFETY_MARGIN:
                logger.debug(
                    f"[Helper {self.id}] Patrol target too far ({target_distance:.1f}), returning"
                )
                return Move(*self.move_towards(*self.ark_position))

            return Move(*self.move_towards(*target))
        logger.debug(f"Helper {self.id} FUCK getting random move")
        return Move(*self._get_random_move())

    def _get_patrol_target(self) -> tuple[float, float] | None:
        """Get next patrol target using boustrophedon pattern - FIXED float precision."""
        if not getattr(self, "_patrol_active", False):
            return None

        # DON'T ROUND - preserve float precision
        cur_x = self.position[0]
        cur_y = self.position[1]

        # Check bounds with float comparison
        if cur_x < self._patrol_x_min:
            return (float(self._patrol_x_min), cur_y)  # Keep current y as float
        if cur_x > self._patrol_x_max:
            return (float(self._patrol_x_max), cur_y)  # Keep current y as float

        # Target row and end x
        row_y = float(max(0, min(GRID_HEIGHT - 1, self._patrol_row)))
        end_x = float(self._patrol_x_max if self._patrol_dir else self._patrol_x_min)

        # Check if we've reached the end of current row (with tolerance for float comparison)
        at_end_x = abs(cur_x - end_x) < 0.5
        at_row_y = abs(cur_y - row_y) < 0.5

        if at_end_x and at_row_y:
            self._advance_to_next_patrol_row()
            if not self._patrol_active:
                return None
            row_y = float(max(0, min(GRID_HEIGHT - 1, self._patrol_row)))
            end_x = float(
                self._patrol_x_max if self._patrol_dir else self._patrol_x_min
            )

        return (end_x, row_y)

    def _advance_to_next_patrol_row(self) -> None:
        """Advance to next patrol row, skipping rows that add no new coverage."""
        # Mark coverage of the row we just completed
        row = self._patrol_row
        start_y = max(0, row - VISION_RADIUS)
        end_y = min(GRID_HEIGHT - 1, row + VISION_RADIUS)
        self._add_covered_interval(start_y, end_y)

        candidate = self._patrol_row + self._patrol_row_step

        while candidate < self._patrol_max_row:
            band_start = max(0, candidate - VISION_RADIUS)
            band_end = min(GRID_HEIGHT - 1, candidate + VISION_RADIUS)
            if not self._is_band_fully_covered(band_start, band_end):
                # This row adds new coverage; use it
                self._patrol_row = candidate
                self._patrol_dir = not self._patrol_dir
                return
            candidate += self._patrol_row_step

        # If we got here, there is no candidate row that adds coverage -> strip done
        self._finish_current_strip()
        self._try_reassign_to_unfinished_strip()

    def _is_band_fully_covered(self, start: int, end: int) -> bool:
        """Return True if [start,end] is entirely within the union of covered intervals."""
        # Covered intervals are merged, so just check if any single interval contains it
        for s, e in self._covered_y_intervals:
            if s <= start and e >= end:
                return True
        return False

    def _add_covered_interval(self, start: int, end: int) -> None:
        """Add [start, end] to self._covered_y_intervals, merging overlaps."""
        new_intervals: list[tuple[int, int]] = []
        placed = False

        for s, e in self._covered_y_intervals:
            if end < s - 1:
                # Our interval is completely before this one
                if not placed:
                    new_intervals.append((start, end))
                    placed = True
                new_intervals.append((s, e))
            elif start > e + 1:
                # Our interval is completely after this one
                new_intervals.append((s, e))
            else:
                # Overlap -> merge
                start = min(start, s)
                end = max(end, e)

        if not placed:
            new_intervals.append((start, end))

        self._covered_y_intervals = new_intervals

    def _finish_current_strip(self) -> None:
        """Mark current strip as done."""
        global _PATROL_STRIPS
        _PATROL_STRIPS[self._patrol_strip_index]["done"] = True
        _PATROL_STRIPS[self._patrol_strip_index]["owner"] = None

    def _try_reassign_to_unfinished_strip(self) -> None:
        """Try to claim an unfinished strip, or explore a new area."""
        global _PATROL_STRIPS

        # First, try to find any unfinished strip
        for i, strip in enumerate(_PATROL_STRIPS):
            if not strip["done"] and strip["owner"] is None:
                self._assign_to_strip(i)
                self._covered_y_intervals = []
                logger.debug(f"[Helper {self.id}] Reassigned to strip {i}")
                return

        # All original strips done - look for helpers that haven't finished yet
        # and split their territory to help them
        for i, strip in enumerate(_PATROL_STRIPS):
            if not strip["done"] and strip["owner"] is not None:
                # This strip is still being worked on - offer to help by splitting it
                if self._try_split_strip(i):
                    logger.debug(f"[Helper {self.id}] Split strip {i} to help")
                    return

        # Absolutely everything explored - NOW restart own strip as last resort
        logger.debug(f"[Helper {self.id}] All areas explored, restarting own strip")
        self._restart_own_strip()

    def _try_split_strip(self, strip_index: int) -> bool:
        """Try to split a strip in half and take one half."""
        global _PATROL_STRIPS
        strip = _PATROL_STRIPS[strip_index]

        # Only split if strip is wide enough (at least 20 units)
        strip_width = strip["x_max"] - strip["x_min"]
        if strip_width < 20:
            return False

        # Split the strip in half
        mid_x = (strip["x_min"] + strip["x_max"]) // 2

        # Original strip keeps left half
        old_x_max = strip["x_max"]
        strip["x_max"] = mid_x

        # Create new strip for right half and assign to this helper
        new_strip = {
            "x_min": mid_x + 1,
            "x_max": old_x_max,
            "owner": self.id,
            "done": False,
            "region": strip["region"],
        }
        _PATROL_STRIPS.append(new_strip)

        # Assign ourselves to the new strip
        self._patrol_strip_index = len(_PATROL_STRIPS) - 1
        self._patrol_x_min = new_strip["x_min"]
        self._patrol_x_max = new_strip["x_max"]

        ark_y = self.ark_position[1]
        if new_strip["region"] == "above":
            self._patrol_row = 0
            self._patrol_max_row = ark_y
        else:
            self._patrol_row = ark_y
            self._patrol_max_row = GRID_HEIGHT

        self._patrol_dir = self.id % 2 == 0
        self._patrol_active = True

        logger.info(
            f"[Helper {self.id}] Created new strip [{new_strip['x_min']}, {new_strip['x_max']}] "
            f"by splitting strip {strip_index}"
        )
        return True

    def _restart_own_strip(self) -> None:
        """Restart patrolling the current strip from the beginning (last resort)."""
        strip = _PATROL_STRIPS[self._patrol_strip_index]

        # Reset strip as not done
        strip["done"] = False
        strip["owner"] = self.id

        # Reset patrol parameters to start from beginning
        ark_y = self.ark_position[1]
        if strip.get("region") == "above":
            self._patrol_row = 0
            self._patrol_max_row = ark_y
        else:
            self._patrol_row = ark_y
            self._patrol_max_row = GRID_HEIGHT
        self._covered_y_intervals = []
        self._patrol_dir = self._patrol_strip_index % 2 == 0
        self._patrol_active = True

    def _assign_to_strip(self, strip_index: int) -> None:
        """Assign to a specific strip."""
        global _PATROL_STRIPS
        strip = _PATROL_STRIPS[strip_index]

        strip["owner"] = self.id
        self._patrol_strip_index = strip_index
        self._patrol_x_min = strip["x_min"]
        self._patrol_x_max = strip["x_max"]

        ark_y = self.ark_position[1]
        if strip.get("region") == "above":
            self._patrol_row = 0
            self._patrol_max_row = ark_y
        else:
            self._patrol_row = ark_y
            self._patrol_max_row = GRID_HEIGHT

        self._patrol_dir = strip_index % 2 == 0
        self._patrol_active = True

    def move_in_dir(self) -> tuple[float, float] | None:
        """Compute a target location for patrol movement."""
        return self._get_patrol_target()

    def _get_random_move(self) -> tuple[float, float]:
        """Get a random valid move."""
        old_x, old_y = self.position
        dx, dy = random() - 0.5, random() - 0.5

        while not self.can_move_to(old_x + dx, old_y + dy):
            dx, dy = random() - 0.5, random() - 0.5

        return old_x + dx, old_y + dy

    # ========================================================================
    # SAFE RADIUS
    # ========================================================================

    def _calculate_distance_to_ark(self) -> float:
        """Calculate Euclidean distance to ark (actual movement distance)."""
        dx = self.position[0] - self.ark_position[0]
        dy = self.position[1] - self.ark_position[1]
        return math.sqrt(dx * dx + dy * dy)

    def _update_safety_tracking(self, snapshot: HelperSurroundingsSnapshot) -> None:
        """Track safety metrics each turn."""
        if self.kind != Kind.Helper:
            return

        current_distance = self._calculate_distance_to_ark()
        self._max_distance_from_ark = max(self._max_distance_from_ark, current_distance)

        # Detect rain start
        if snapshot.is_raining and not self._rain_started:
            self._rain_started = True
            self._distance_when_rain_started = current_distance
            # NEW: remember when the rain started (turn index)
            self._rain_start_turn = getattr(snapshot, "time_elapsed", 0)

            logger.info(
                f"[Helper {self.id}] RAIN STARTED! "
                f"Distance to ark: {self._distance_when_rain_started:.1f}, "
                f"start_turn: {self._rain_start_turn}"
            )

    def _can_explore_more(self) -> int:
        """How many more turns we can still explore.

        Before rain:
            Limit purely by distance so we don't wander absurdly far.
        After rain:
            Budget is: time_since_rain_started + distance_to_ark
            We must keep that ≤ MAX_SAFE_DISTANCE - SAFETY_MARGIN.
        """
        current_distance = self._calculate_distance_to_ark()

        # BEFORE RAIN: same heuristic as before (pure distance cap)
        if not self._rain_started:
            return int(MAX_SAFE_DISTANCE - SAFETY_MARGIN - current_distance)

        # AFTER RAIN: time + distance budget
        rain_start_turn = getattr(self, "_rain_start_turn", None)
        if rain_start_turn is None or not self.snapshot:
            # Failsafe: if we don't know, be conservative
            return 0

        current_turn = getattr(self.snapshot, "time_elapsed", rain_start_turn)
        time_since_rain = max(0, current_turn - rain_start_turn)

        # Budget: time_since_rain + distance_to_ark <= MAX_SAFE_DISTANCE - SAFETY_MARGIN
        used_budget = time_since_rain + current_distance
        remaining_budget = MAX_SAFE_DISTANCE - SAFETY_MARGIN - used_budget

        return max(0, int(remaining_budget))

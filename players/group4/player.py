from __future__ import annotations
import math
import random
from typing import Optional

from core.action import Action, Move, Obtain, Release
from core.animal import Animal, Gender
from core.message import Message
from core.player import Player
from core.snapshots import HelperSurroundingsSnapshot
from core.views.cell_view import CellView
from core.views.player_view import Kind

import core.constants as c


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


class Player4(Player):
    """Helper implementation that patrols safe regions, targets rare species early,
    and coordinates via messages."""

    SAFE_MANHATTAN_LIMIT = c.START_RAIN  # can get back to ark before deadline
    ASSIGNMENT_TURN_LIMIT = 1000  # helpers restrict to assigned species early game

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
        """Initialize long-lived state about assignments, priorities, and movement.

        All helpers share the same logic, but each helper caches its assigned region,
        rarity priority table, and local tracking structures (blocked cells, pending
        obtains, etc.) so we can respond deterministically to arena updates.
        """

        # Runtime state refreshed every turn from the arena snapshot.
        self.turn = 0
        self.is_raining = False
        self.sight = None
        self.ark_view = None
        self.rain_start_turn: Optional[int] = None
        self.total_time_estimate: Optional[int] = None
        self.target_return_turn: Optional[int] = None

        # Helpers are numbered after Noah (id 0), so compute per-helper indices.
        self.helper_index = self.id - 1 if self.kind == Kind.Helper else None
        self.helper_count = max(1, self.num_helpers - 1) if self.num_helpers else 1
        self.region_index = (
            (self.helper_index if self.helper_index is not None else 0)
            if self.kind == Kind.Helper
            else None
        )

        self.assignment_broadcasted = False

        # Movement targets live inside the global safe area and a per-helper slice.
        self.safe_bounds = self._compute_safe_bounds()
        self.region_bounds = self._compute_region_bounds()
        self.patrol_target: Optional[tuple[float, float]] = None
        self.tracking_cell: Optional[tuple[int, int]] = None

        # Pre-compute rarity ranks so greedy selection remains deterministic.
        self.species_priority = self._build_species_priority(species_populations)
        population_values = sorted(
            priority[0] for priority in self.species_priority.values()
        )
        max_pop = max(population_values, default=0) + 100
        self.default_priority = (max_pop, 999)
        rare_index = (
            min(len(population_values) - 1, max(0, int(len(population_values) * 0.35)))
            if population_values
            else 0
        )
        self.rare_cutoff = population_values[rare_index] if population_values else 0

        self.species_on_ark: dict[int, set[Gender]] = {}
        self.known_assignments: dict[int, int] = {}
        self.helpers_returning: set[int] = set()
        self.pending_obtain: Optional[Animal] = None
        self.unavailable_animals: set[Animal] = set()
        # Cells that we want to skip until a certain turn because they were contested.
        self.blocked_cells: dict[tuple[int, int], int] = {}
        # Rotating index for broadcasting species-on-ark updates
        self.species_broadcast_index = 0
        # Track animals in other helpers' flocks to avoid redundancy
        # Format: {(species_id, gender): set[helper_id]} - which helpers have this animal
        self.animals_in_other_flocks: dict[tuple[int, Gender], set[int]] = {}
        # Track last turn we heard from each helper (for cleanup)
        self.helper_last_seen: dict[int, int] = {}
        # Rotating index for broadcasting our flock animals
        self.flock_broadcast_index = 0
        # Track rare-first sweep progress so we can change behavior after first return
        self.first_sweep_done = False
        self.first_sweep_returned = False
        # Keep state when we've committed to head back or stay parked on the Ark
        self.force_return = False
        self.hunkered_down = False
        # Target species assignment for early game (per-helper)
        self.target_species: Optional[set[int]] = self._compute_target_species(
            species_populations
        )
        self.assignment_turn_limit = self._compute_assignment_turn_limit(
            species_populations
        )
        # Remember where we were patrolling before a forced return so we can resume
        self.resume_patrol_target: Optional[tuple[float, float]] = None
        # Track species count and a throughput mode for sparse-helper / dense-species scenarios
        self.species_count = len(species_populations)
        self.throughput_mode = self._compute_throughput_mode()

    # === Territory & Priority Helpers ===

    def _build_species_priority(
        self, species_populations: dict[str, int]
    ) -> dict[int, tuple[int, int]]:
        """Convert species population map into sortable priority tuples."""
        priority: dict[int, tuple[int, int]] = {}
        for letter, population in species_populations.items():
            sid = ord(letter) - ord("a")
            priority[sid] = (population, sid)
        return priority

    def _compute_safe_bounds(self) -> tuple[float, float, float, float]:
        """Return the axis-aligned bounding box that fits inside the safe 1008 steps."""
        ax, ay = self.ark_position
        return (
            max(0.0, ax - self.SAFE_MANHATTAN_LIMIT),
            min(float(c.X - 1), ax + self.SAFE_MANHATTAN_LIMIT),
            max(0.0, ay - self.SAFE_MANHATTAN_LIMIT),
            min(float(c.Y - 1), ay + self.SAFE_MANHATTAN_LIMIT),
        )

    def _compute_region_bounds(self) -> Optional[tuple[float, float, float, float]]:
        """Split the safe diamond into square-ish grids and return this helper's slice."""
        if self.kind != Kind.Helper or self.region_index is None:
            return None

        cols = math.ceil(math.sqrt(self.helper_count))
        rows = math.ceil(self.helper_count / cols)
        region_width = (self.safe_bounds[1] - self.safe_bounds[0]) / max(cols, 1)
        region_height = (self.safe_bounds[3] - self.safe_bounds[2]) / max(rows, 1)

        row = self.region_index // cols
        col = self.region_index % cols

        min_x = self.safe_bounds[0] + col * region_width
        max_x = min(self.safe_bounds[0] + (col + 1) * region_width, self.safe_bounds[1])
        min_y = self.safe_bounds[2] + row * region_height
        max_y = min(
            self.safe_bounds[2] + (row + 1) * region_height, self.safe_bounds[3]
        )

        return (min_x, max_x, min_y, max_y)

    def _compute_target_species(
        self, species_populations: dict[str, int]
    ) -> Optional[set[int]]:
        """Specialize only when we have enough helpers; otherwise stay generalist."""
        if self.kind != Kind.Helper or self.helper_index is None:
            return None
        if not species_populations:
            return None
        if self.helper_count <= 3:
            return None

        species_by_rarity: list[tuple[int, int]] = []
        for letter, pop in species_populations.items():
            sid = ord(letter) - ord("a")
            species_by_rarity.append((sid, pop))

        species_by_rarity.sort(key=lambda entry: (entry[1], entry[0]))
        species_ids = [sid for sid, _ in species_by_rarity]
        species_count = len(species_ids)
        helper_density = self.helper_count / max(1, species_count)

        coverage = 1
        if helper_density < 0.8:
            coverage = min(3, species_count)
        elif helper_density < 1.5:
            coverage = min(2, species_count)

        start = self.helper_index % species_count if species_count else 0
        assignments = {
            species_ids[(start + offset) % species_count] for offset in range(coverage)
        }
        assigned_names = ",".join(chr(sid + ord("a")) for sid in sorted(assignments))
        print(f"Helper {self.id} assigned to target species {assigned_names}")
        return assignments

    def _compute_assignment_turn_limit(
        self, species_populations: dict[str, int]
    ) -> int:
        """Dynamic specialization window based on helper-to-species ratio."""
        species_count = max(1, len(species_populations))
        helper_density = self.helper_count / species_count

        if helper_density < 0.8:
            return 500
        if helper_density > 1.6:
            return 1400
        return 900

    def _compute_throughput_mode(self) -> bool:
        """Detect cases with few helpers and many species where frequent trips help."""
        # Disabled: we now require full flocks before returning even in sparse-helper settings.
        return False

    def _assignment_window_active(self) -> bool:
        """Whether helpers should restrict to their assigned species."""
        return (
            self.kind == Kind.Helper
            and self.helper_count > 3
            and self.turn < self.assignment_turn_limit
            and bool(self.target_species)
        )

    def _effective_safe_limit(self) -> float:
        """Shrink the safe radius once rain begins to mimic a moving storm."""
        base_limit = float(self.SAFE_MANHATTAN_LIMIT)
        if not self.is_raining:
            return base_limit

        rain_start = (
            self.rain_start_turn if self.rain_start_turn is not None else self.turn
        )
        elapsed_since_rain = max(0.0, float(self.turn - rain_start))
        factor = max(0.0, 1.0 - (elapsed_since_rain / c.START_RAIN))
        return max(0.0, factor * base_limit)

    def _is_point_safe(self, x: float, y: float) -> bool:
        """Check Manhattan distance constraint back to the Ark."""
        ax, ay = self.ark_position
        limit = self._effective_safe_limit()
        return abs(x - ax) + abs(y - ay) <= limit

    # === Timing & Return Planning ===

    def _update_time_inference(self) -> None:
        """Infer total game length once rain starts so we can aim for last-turn returns."""
        if self.is_raining and self.rain_start_turn is None:
            self.rain_start_turn = self.turn
            self.total_time_estimate = self.turn + c.START_RAIN

        if self.total_time_estimate is None and self.rain_start_turn is not None:
            self.total_time_estimate = self.rain_start_turn + c.START_RAIN

        if self.total_time_estimate is not None:
            self._update_return_plan()

    def _turns_left_estimate(self) -> float:
        """Aggressive estimate of remaining turns; optimistic until rain gives a bound."""
        if self.total_time_estimate is not None:
            return max(0.0, float(self.total_time_estimate - self.turn))
        if self.rain_start_turn is not None:
            return max(0.0, float(self.rain_start_turn + c.START_RAIN - self.turn))
        return float(c.MAX_T - self.turn)

    def _steps_to_ark(self) -> int:
        """Estimated turns needed to reach the Ark from the current position."""
        step = max(c.MAX_DISTANCE_KM * 0.99, 0.1)
        return max(1, math.ceil(self._distance_from_ark() / step))

    def _return_buffer(self) -> int:
        """How many turns we reserve to get home; scales with distance and flock fullness."""
        buffer = self._steps_to_ark() + 6
        if self.is_flock_full():
            buffer += 2
        return buffer

    def _update_return_plan(self) -> None:
        """Set a target return turn aimed near the game end while leaving just enough travel time."""
        if self.total_time_estimate is None:
            return

        self.target_return_turn = max(
            self.turn, int(self.total_time_estimate - self._return_buffer())
        )

    # === Messaging & Snapshot Handling ===

    def check_surroundings(self, snapshot: HelperSurroundingsSnapshot) -> int:
        """Refresh local state from the engine snapshot and decide what to broadcast."""
        self.turn = snapshot.time_elapsed
        self.is_raining = snapshot.is_raining
        self.position = snapshot.position
        old_flock_size = len(self.flock)
        self.flock = snapshot.flock
        self.sight = snapshot.sight
        self.ark_view = snapshot.ark_view

        # Reset flock broadcast index if flock changed (animals added/removed)
        if len(self.flock) != old_flock_size:
            self.flock_broadcast_index = 0

        self._update_time_inference()

        # Noah always has ark_view (they're on the Ark), helpers get it when visiting
        if snapshot.ark_view:
            self._update_ark_species(snapshot.ark_view)
        elif self.kind == Kind.Noah:
            # Noah should always have ark_view, but handle gracefully if missing
            pass

        self._handle_pending_obtain()
        self._update_phase_flags()
        if self.hunkered_down and not self.is_in_ark():
            # Somehow pushed out of the Ark while hunkered: re-enter immediately
            self.force_return = True
        if self.is_raining and self.is_in_ark():
            self.hunkered_down = True

        return self._compose_message()

    def _update_phase_flags(self) -> None:
        """Flip from the rare-first sweep to normal play once we've deposited animals."""
        was_first_sweep = not self.first_sweep_done
        if self.is_in_ark() and self.flock:
            self.first_sweep_returned = True
            self.first_sweep_done = True
        elif not self.first_sweep_done and self.turn >= c.START_RAIN // 2:
            self.first_sweep_done = True

        if was_first_sweep and self.first_sweep_done:
            # Reset patrol so normal safe-region behavior picks new anchors
            self.tracking_cell = None
            self.patrol_target = None

    def _is_first_sweep_active(self) -> bool:
        if self.helper_count <= 3:
            return False
        return not (self.first_sweep_done or self.first_sweep_returned)

    def _handle_pending_obtain(self) -> None:
        """Detect whether the animal we attempted to obtain actually joined the flock."""
        if self.pending_obtain is None:
            return

        if self.pending_obtain in self.flock:
            self.unavailable_animals.discard(self.pending_obtain)
        else:
            self.unavailable_animals.add(self.pending_obtain)
            position_cell = (int(self.position[0]), int(self.position[1]))
            self._block_cell_temporarily(position_cell, duration=5)
            self.tracking_cell = None
            self.patrol_target = None

        self.pending_obtain = None

    def _update_ark_species(self, ark_view) -> None:
        """Refresh complete Ark species status when at the Ark, ensuring latest information."""
        # Noah is always on the Ark, so always do a complete refresh
        # Helpers do a complete refresh when visiting, incremental otherwise
        if self.kind == Kind.Noah or self.is_in_ark():
            # Rebuild species_on_ark from scratch based on current Ark state
            # This ensures we get the latest status regardless of missed messages
            self.species_on_ark = {}
            for animal in ark_view.animals:
                if animal.gender == Gender.Unknown:
                    continue
                if animal.species_id not in self.species_on_ark:
                    self.species_on_ark[animal.species_id] = set()
                self.species_on_ark[animal.species_id].add(animal.gender)
            # Reset broadcast cycling so we re-share the full Ark manifest
            self.species_broadcast_index = 0
        else:
            # When not at Ark, just update incrementally (from messages or partial views)
            for animal in ark_view.animals:
                if animal.gender == Gender.Unknown:
                    continue
                if animal.species_id not in self.species_on_ark:
                    self.species_on_ark[animal.species_id] = set()
                self.species_on_ark[animal.species_id].add(animal.gender)

    def _is_species_complete_on_ark(self, species_id: int) -> bool:
        """Check if a species has both genders on the Ark."""
        genders = self.species_on_ark.get(species_id, set())
        return Gender.Male in genders and Gender.Female in genders

    def _is_gender_on_ark(self, species_id: int, gender: Gender) -> bool:
        """Check if a specific gender of a species is already on the Ark."""
        if gender == Gender.Unknown:
            return False  # Unknown gender doesn't count as redundant
        genders = self.species_on_ark.get(species_id, set())
        return gender in genders

    def _get_next_species_to_broadcast(self) -> Optional[int]:
        """Get the next complete species to broadcast to the team."""
        complete_species = [
            sid
            for sid in self.species_on_ark.keys()
            if self._is_species_complete_on_ark(sid)
        ]
        if not complete_species:
            return None

        complete_species.sort()
        idx = self.species_broadcast_index % len(complete_species)
        species = complete_species[idx]
        self.species_broadcast_index = (idx + 1) % len(complete_species)
        return species

    def _get_next_ark_species_gender_to_broadcast(self) -> Optional[tuple[int, Gender]]:
        """Get the next species-gender pair on the Ark to broadcast (for Noah).

        Cycles through all species and their genders present on the Ark.
        """
        # Build list of all (species_id, gender) pairs on the Ark
        ark_species_genders = []
        for species_id, genders in self.species_on_ark.items():
            for gender in genders:
                if gender != Gender.Unknown:
                    ark_species_genders.append((species_id, gender))

        if not ark_species_genders:
            return None

        # Sort for deterministic ordering
        ark_species_genders.sort(key=lambda x: (x[0], x[1].value))
        idx = self.species_broadcast_index % len(ark_species_genders)
        species_gender = ark_species_genders[idx]
        self.species_broadcast_index = (idx + 1) % len(ark_species_genders)
        return species_gender

    def _get_next_flock_animal_to_broadcast(self) -> Optional[tuple[int, Gender]]:
        """Get the next animal from our flock to broadcast (species_id, gender)."""
        # Only broadcast animals with known gender (skip Unknown)
        flock_animals = [
            (animal.species_id, animal.gender)
            for animal in self.flock
            if animal.gender != Gender.Unknown
        ]
        if not flock_animals:
            return None

        # Sort for deterministic ordering: by species_id first, then by gender value
        flock_animals.sort(key=lambda x: (x[0], x[1].value))
        idx = self.flock_broadcast_index % len(flock_animals)
        animal_info = flock_animals[idx]
        self.flock_broadcast_index = (idx + 1) % len(flock_animals)
        return animal_info

    def _compose_message(self) -> int:
        """Continuously broadcast Ark status, flock animals, or fallback to status bits."""
        # Noah always broadcasts the current Ark state (all species and their genders)
        if self.kind == Kind.Noah:
            # Noah continuously broadcasts all species-gender pairs on the Ark
            # Message format: 0x80 | 0x20 | (species_id & 0x3F) | ((1 if Female else 0) << 6)
            # Same format as flock-animal messages, but helpers can distinguish by sender ID
            ark_species_gender = self._get_next_ark_species_gender_to_broadcast()
            if ark_species_gender is not None:
                species_id, gender = ark_species_gender
                if species_id < 64:  # Support up to 64 species (0-63)
                    gender_bit = 1 if gender == Gender.Female else 0
                    msg = 0x80 | 0x20 | (species_id & 0x3F) | (gender_bit << 6)
                    return msg if self.is_message_valid(msg) else (msg & 0xFF)
            # If no animals on Ark yet, send a status message
            return 0

        if self.kind != Kind.Helper:
            return 0

        # First message: broadcast region assignment
        if not self.assignment_broadcasted and self.region_index is not None:
            msg = 0x80 | (self.region_index & 0x3F)
            self.assignment_broadcasted = True
            return msg if self.is_message_valid(msg) else (msg & 0xFF)

        # Highest priority: share Ark-complete species in a round-robin so teammates
        # constantly prune their search space, regardless of distance from the Ark.
        species_to_broadcast = self._get_next_species_to_broadcast()
        if species_to_broadcast is not None:
            msg = 0x80 | 0x40 | (species_to_broadcast & 0x3F)
            return msg if self.is_message_valid(msg) else (msg & 0xFF)

        # Second priority: broadcast our flock animals so others know what we're carrying
        # Message format: 0x80 | 0x20 | (species_id & 0x3F) | ((1 if Female else 0) << 6)
        # Bits 0-5: species_id (6 bits = 0-63), Bit 6: gender (0=Male, 1=Female)
        # Note: Bit 6 (0x40) is available here because flock-animal flag is bit 5 (0x20)
        # This allows 64 species (0-63) and 2 genders with clean bit separation
        flock_animal = self._get_next_flock_animal_to_broadcast()
        if flock_animal is not None:
            species_id, gender = flock_animal
            if species_id < 64:  # Support up to 64 species (0-63)
                gender_bit = 1 if gender == Gender.Female else 0
                msg = 0x80 | 0x20 | (species_id & 0x3F) | (gender_bit << 6)
                return msg if self.is_message_valid(msg) else (msg & 0xFF)

        # Regular status message: returning flag + flock size
        msg = 0
        if self._should_return_to_ark():
            msg |= 0x40

        msg |= min(len(self.flock), 0x07)

        return msg if self.is_message_valid(msg) else (msg & 0xFF)

    def _process_messages(self, messages: list[Message]) -> None:
        """Decode broadcasts from neighbors and keep track of their assignments/state."""
        helpers_seen_this_turn = set()

        for msg in messages:
            helper_id = msg.from_helper.id
            helpers_seen_this_turn.add(helper_id)
            self.helper_last_seen[helper_id] = self.turn

            if msg.contents & 0x80:
                # Check bit 5 (0x20) FIRST for flock-animal messages, because Female animals
                # set both bit 5 (flock flag) and bit 6 (gender), which would match
                # species-on-ark check if we checked bit 6 first
                if msg.contents & 0x20:
                    # Message format: 0x80 | 0x20 | (species_id & 0x3F) | ((gender_bit) << 6)
                    # Bits 0-5: species_id (6 bits = 0-63), Bit 6: gender (0=Male, 1=Female)
                    species_id = msg.contents & 0x3F  # bits 0-5
                    gender_bit = (msg.contents >> 6) & 0x01  # bit 6
                    gender = Gender.Female if gender_bit else Gender.Male

                    # Distinguish: Noah (id=0) broadcasts Ark genders, helpers broadcast flock animals
                    if helper_id == 0:
                        # Noah's message: update Ark species-gender knowledge
                        if species_id not in self.species_on_ark:
                            self.species_on_ark[species_id] = set()
                        self.species_on_ark[species_id].add(gender)
                    else:
                        # Helper's message: track what other helpers are carrying
                        key = (species_id, gender)
                        if key not in self.animals_in_other_flocks:
                            self.animals_in_other_flocks[key] = set()
                        self.animals_in_other_flocks[key].add(helper_id)
                elif msg.contents & 0x40:
                    # Species-on-ark message: mark this species as complete
                    # Format: 0x80 | 0x40 | (species_id & 0x3F)
                    # Note: This check comes after flock-animal check to avoid false matches
                    species_id = msg.contents & 0x3F
                    if species_id not in self.species_on_ark:
                        self.species_on_ark[species_id] = set()
                    self.species_on_ark[species_id].add(Gender.Male)
                    self.species_on_ark[species_id].add(Gender.Female)
                else:
                    # Assignment message
                    self.known_assignments[helper_id] = msg.contents & 0x3F
            elif msg.contents & 0x40:
                self.helpers_returning.add(helper_id)

        # Clean up entries for helpers we haven't heard from in a while (out of range or released animals)
        # Remove entries if we haven't heard from helper in last 10 turns
        # Reduced from 20 to be more responsive - with max 4 animals per flock,
        # round-robin takes at most 4 turns, so 10 gives plenty of buffer
        stale_threshold = 10
        for key in list(self.animals_in_other_flocks.keys()):
            self.animals_in_other_flocks[key] = {
                h_id
                for h_id in self.animals_in_other_flocks[key]
                if self.turn - self.helper_last_seen.get(h_id, -stale_threshold)
                < stale_threshold
            }
            if not self.animals_in_other_flocks[key]:
                del self.animals_in_other_flocks[key]

        # After processing all messages, check for conflicts and implement tie-breaking
        self._resolve_flock_conflicts()

    def _resolve_flock_conflicts(self) -> None:
        """Implement tie-breaking: if multiple helpers have same animal, lower ID keeps it.

        Conservative: Only release if we're very confident a lower-ID helper has it and will deliver it.
        """
        animals_to_release = []
        for animal in self.flock:
            if animal.gender == Gender.Unknown:
                continue

            key = (animal.species_id, animal.gender)
            other_helpers = self.animals_in_other_flocks.get(key, set())

            if other_helpers:
                # Very conservative: Only release if there's a helper with LOWER ID AND we've heard from them VERY recently
                # This prevents releasing based on stale information
                recent_threshold = (
                    2  # Only trust information from last 2 turns (very recent)
                )
                recent_lower_id_helpers = [
                    h_id
                    for h_id in other_helpers
                    if h_id < self.id
                    and self.turn - self.helper_last_seen.get(h_id, -recent_threshold)
                    < recent_threshold
                ]

                # Only release if we've heard from a lower-ID helper VERY recently (within 2 turns)
                # This is conservative - we'd rather keep a potential duplicate than release unnecessarily
                if recent_lower_id_helpers:
                    # A lower ID helper has it very recently, we should release
                    animals_to_release.append(animal)

        # Release animals we shouldn't keep (but only if we're very confident)
        for animal in animals_to_release:
            if animal in self.flock:
                # Mark as unavailable temporarily to avoid re-picking immediately
                self.unavailable_animals.add(animal)
                # We'll release it in the release method

    def _is_animal_in_other_flocks(self, species_id: int, gender: Gender) -> bool:
        """Check if this animal (species_id, gender) is in another helper's flock.

        Conservative: Only avoid if we're very confident another helper has it and will deliver it.
        """
        if gender == Gender.Unknown:
            return False

        key = (species_id, gender)
        other_helpers = self.animals_in_other_flocks.get(key, set())

        if not other_helpers:
            return False

        # Very conservative: Only avoid if there's a helper with LOWER ID AND we've heard from them VERY recently
        # This ensures we only avoid when we're confident they still have it
        recent_threshold = 2  # Only trust information from last 2 turns (very recent)
        recent_lower_id_helpers = [
            h_id
            for h_id in other_helpers
            if h_id < self.id
            and self.turn - self.helper_last_seen.get(h_id, -recent_threshold)
            < recent_threshold
        ]

        # Only avoid if there's a very recent lower-ID helper (within 2 turns)
        # This is conservative - we'd rather pick up a duplicate than miss an animal
        return len(recent_lower_id_helpers) > 0

    def _release_complete_species_animals(self) -> Optional[Action]:
        """Release animals that are redundant: complete species, duplicate genders on Ark, or conflicts."""
        if not self.flock:
            return None

        animals_to_release = []
        seen_pairs: dict[tuple[int, Gender], Animal] = {}
        for animal in self.flock:
            # Release if species is complete on Ark
            if self._is_species_complete_on_ark(animal.species_id):
                animals_to_release.append(animal)
            # Release if this gender is already on Ark
            elif self._is_gender_on_ark(animal.species_id, animal.gender):
                animals_to_release.append(animal)
            # Only release for tie-breaking if we're very confident (very conservative)
            # This check is already conservative in _is_animal_in_other_flocks, so we keep it
            # but it will only trigger in very clear conflict cases
            elif self._is_animal_in_other_flocks(animal.species_id, animal.gender):
                animals_to_release.append(animal)
            # Release duplicates within our own flock
            elif animal.gender != Gender.Unknown:
                key = (animal.species_id, animal.gender)
                if key in seen_pairs:
                    animals_to_release.append(animal)
                else:
                    seen_pairs[key] = animal

        if not animals_to_release:
            return None

        worst_animal = max(animals_to_release, key=lambda a: self._score_animal(a))
        return Release(worst_animal)

    # === Perception, Scoring & Target Selection ===

    def _get_my_cell(self) -> CellView:
        """Return the precise cell view that matches our floating-point coordinates."""
        xcell, ycell = tuple(map(int, self.position))
        if self.sight is None or not self.sight.cell_is_in_sight(xcell, ycell):
            raise Exception(f"{self} cannot determine its current cell")

        return self.sight.get_cellview_at(xcell, ycell)

    def _distance_from_ark(self) -> float:
        """Euclidean distance to Ark, helpful for conservative returns."""
        return _distance(*self.position, *self.ark_position)

    def _manhattan_distance_to_ark(self) -> float:
        ax, ay = self.ark_position
        return abs(self.position[0] - ax) + abs(self.position[1] - ay)

    def _flock_species_count(self, species_id: int) -> int:
        """Count how many animals in our flock already belong to the given species."""
        return sum(1 for animal in self.flock if animal.species_id == species_id)

    def _has_species_gender_in_flock(self, species_id: int, gender: Gender) -> bool:
        if gender == Gender.Unknown:
            return False
        return any(
            animal.species_id == species_id and animal.gender == gender
            for animal in self.flock
        )

    def _species_priority(self, species_id: int) -> tuple[int, int]:
        """Look up rarity tuple or fall back to the default high score."""
        return self.species_priority.get(species_id, self.default_priority)

    def _score_animal(
        self, animal: Animal, assume_unknown_desired: bool = False
    ) -> tuple[int, int, int, int, int, int]:
        """
        Lower scores are better.
        New scoring strongly incentivizes completing species pairs.
        """
        population, sid = self._species_priority(animal.species_id)

        genders_on_ark = self.species_on_ark.get(animal.species_id, set())
        flock_genders = {
            a.gender for a in self.flock if a.species_id == animal.species_id
        }

        pairing_bonus = 0
        if animal.gender != Gender.Unknown:
            if len(genders_on_ark) == 1 and animal.gender not in genders_on_ark:
                pairing_bonus -= 3
            if len(flock_genders) == 1 and animal.gender not in flock_genders:
                pairing_bonus -= 2

        if animal.gender == Gender.Unknown and assume_unknown_desired:
            pairing_bonus -= 1

        # Only penalize if the EXACT same gender is already seen (on Ark or in flock)
        # This allows picking the opposite gender to complete pairs
        seen_genders = genders_on_ark.union(flock_genders)
        duplicate_species_penalty = (
            1
            if (animal.gender != Gender.Unknown and animal.gender in seen_genders)
            else 0
        )
        unknown_penalty = 1 if animal.gender == Gender.Unknown else 0
        duplicates = self._flock_species_count(animal.species_id)

        return (
            duplicate_species_penalty,
            population,
            -pairing_bonus,
            duplicates,
            unknown_penalty,
            sid,
        )

    def _best_animal_in_cell(
        self, cellview: CellView, assume_unknown: bool = False
    ) -> tuple[Animal, tuple[int, int, int, int, int, int]] | tuple[None, None]:
        """Return the highest ranked animal; prefer rare early, but allow any if none rare."""

        def candidates() -> list[Animal]:
            picks = []
            for animal in cellview.animals:
                if animal in self.flock:
                    continue
                if animal in self.unavailable_animals:
                    continue
                if self._is_species_complete_on_ark(animal.species_id):
                    continue
                if self._is_gender_on_ark(animal.species_id, animal.gender):
                    continue
                if self._has_species_gender_in_flock(animal.species_id, animal.gender):
                    continue
                # Skip animals that are in other helpers' flocks (unless we have priority)
                if self._is_animal_in_other_flocks(animal.species_id, animal.gender):
                    continue
                picks.append(animal)
            return picks

        animals = candidates()
        if not animals:
            return (None, None)

        def best_among(
            pool: list[Animal],
        ) -> tuple[Animal, tuple[int, int, int, int, int, int]]:
            chosen: Optional[Animal] = None
            chosen_score: Optional[tuple[int, int, int, int, int, int]] = None
            for animal in pool:
                score = self._score_animal(
                    animal, assume_unknown_desired=assume_unknown
                )
                if chosen is None or chosen_score is None or score < chosen_score:
                    chosen = animal
                    chosen_score = score
            return chosen, chosen_score  # type: ignore

        if self._is_first_sweep_active():
            rare_pool = []
            for a in animals:
                population, _ = self._species_priority(a.species_id)
                if population <= self.rare_cutoff:
                    rare_pool.append(a)
            if rare_pool:
                return best_among(rare_pool)

        return best_among(animals)

    def _block_cell_temporarily(self, cell: tuple[int, int], duration: int = 6) -> None:
        """Block a cell for a few turns to avoid hovering or repeated contention."""
        existing_expiry = self.blocked_cells.get(cell, self.turn)
        self.blocked_cells[cell] = max(existing_expiry, self.turn + duration)

    def _purge_blocked_cells(self) -> None:
        """Remove stale entries so we eventually reconsider cells after timeout."""
        expired = [
            cell for cell, expiry in self.blocked_cells.items() if expiry <= self.turn
        ]
        for cell in expired:
            del self.blocked_cells[cell]

    def _should_return_to_ark(self) -> bool:
        """Decide when to abandon exploration and head back to the Ark."""
        if self.kind != Kind.Helper:
            return False

        # Full flock should always head home to unload/update ark state.
        if self.is_flock_full():
            return True

        self._update_return_plan()
        turns_left = self._turns_left_estimate()
        dist_to_ark = self._manhattan_distance_to_ark()
        limit = self._effective_safe_limit()
        buffer = self._return_buffer()

        if turns_left <= buffer:
            return True

        if self.target_return_turn is not None and self.turn >= self.target_return_turn:
            return True

        if limit <= 0.0 and dist_to_ark > 0.0:
            return True

        if self.is_raining and limit > 0.0 and dist_to_ark >= limit * 1.05:
            return True

        if self._is_first_sweep_active():
            return False

        return False

    def _pick_new_patrol_target(self) -> None:
        """Select a random waypoint within our assigned region or the safe bounds."""
        if self._is_first_sweep_active() or not self.region_bounds:
            self.patrol_target = self._random_point_in_safe_area()
            return

        min_x, max_x, min_y, max_y = self.region_bounds
        for _ in range(10):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            if self._is_point_safe(x, y):
                self.patrol_target = (x, y)
                return

        self.patrol_target = self._random_point_in_safe_area()

    def _random_point_in_safe_area(self) -> tuple[float, float]:
        """Sample a coordinate for patrolling, unrestricted during the first sweep."""
        if self._is_first_sweep_active():
            return (
                random.uniform(0.0, float(c.X - 1)),
                random.uniform(0.0, float(c.Y - 1)),
            )

        min_x, max_x, min_y, max_y = self.safe_bounds
        for _ in range(20):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            if self._is_point_safe(x, y):
                return (x, y)

        return self.ark_position

    def _update_tracking_cell(self) -> None:
        """Find the best visible cell to chase next, respecting blocked cells."""
        if self.sight is None:
            return

        if self.is_flock_full():
            # Do not chase new animals when full; keep roaming/returning instead.
            self.tracking_cell = None
            return

        self._purge_blocked_cells()

        best_cell = None
        best_score = None
        for cellview in self.sight:
            if not cellview.animals:
                continue
            if (not self._is_first_sweep_active()) and not self._is_point_safe(
                cellview.x, cellview.y
            ):
                continue
            if any(helper.id != self.id for helper in cellview.helpers):
                continue

            if (cellview.x, cellview.y) in self.blocked_cells:
                continue
            best_animal, score = self._best_animal_in_cell(
                cellview, assume_unknown=True
            )
            if best_animal is None or score is None:
                continue
            dist = _distance(*self.position, cellview.x, cellview.y)
            candidate_score = (*score, dist)
            if best_cell is None or best_score is None or candidate_score < best_score:
                best_cell = (cellview.x, cellview.y)
                best_score = candidate_score

        if best_cell:
            self.tracking_cell = best_cell

    def _tracking_target_active(self) -> bool:
        """Validate current tracking cell and discard it when conditions change."""
        if not self.tracking_cell:
            return False

        tx, ty = self.tracking_cell
        expiry = self.blocked_cells.get((tx, ty))
        if expiry is not None:
            if expiry > self.turn:
                self.tracking_cell = None
                return False
            del self.blocked_cells[(tx, ty)]

        if (not self._is_first_sweep_active()) and not self._is_point_safe(tx, ty):
            self.tracking_cell = None
            return False

        if self.sight and self.sight.cell_is_in_sight(tx, ty):
            cell = self.sight.get_cellview_at(tx, ty)
            if any(helper.id != self.id for helper in cell.helpers):
                self._block_cell_temporarily((tx, ty), duration=5)
                self.tracking_cell = None
                return False

            if not cell.animals:
                self.tracking_cell = None
                return False

        return True

    # === Action Selection & Movement ===

    def get_action(self, messages: list[Message]) -> Action | None:
        """Main decision tree: release, obtain, chase, or roam."""
        self._process_messages(messages)

        if self.kind == Kind.Noah:
            return None

        release_complete_action = self._release_complete_species_animals()
        if release_complete_action:
            return release_complete_action

        my_cell = self._get_my_cell()
        self._prune_unavailable_animals(my_cell)

        if self._should_return_to_ark():
            if self.resume_patrol_target is None:
                self.resume_patrol_target = self.patrol_target or self.position
            self.force_return = True

        if self.hunkered_down and self.is_in_ark():
            self.tracking_cell = None
            self.patrol_target = None
            return Move(*self.ark_position)

        if self.force_return:
            self.tracking_cell = None
            self.patrol_target = None
            if self.is_in_ark():
                if self.is_raining:
                    self.hunkered_down = True
                    self.force_return = False
                    return Move(*self.ark_position)
                self.force_return = False
                if self.resume_patrol_target:
                    self.patrol_target = self.resume_patrol_target
                self.resume_patrol_target = None
            else:
                # While returning, allow in-place upgrades only; no chasing.
                upgrade_action = self._maybe_release_for_priority(my_cell)
                if upgrade_action:
                    return upgrade_action
            return Move(*self.move_towards(*self.ark_position))

        release_action = self._maybe_release_for_priority(my_cell)
        if release_action:
            return release_action

        obtain_candidate = self._select_animal_here(my_cell)
        if obtain_candidate:
            self.pending_obtain = obtain_candidate
            return Obtain(obtain_candidate)

        self._update_tracking_cell()
        move_target = self._select_move_target()

        next_pos = self.move_towards(*move_target)
        if _distance(*next_pos, *self.position) < 0.05:
            next_pos = self._random_safe_step()

        return Move(*next_pos)

    def _select_animal_here(self, cellview: CellView) -> Optional[Animal]:
        cell_coords = (cellview.x, cellview.y)
        if cell_coords in self.blocked_cells:
            return None

        if self.is_flock_full():
            return None

        animal, _ = self._best_animal_in_cell(cellview)
        if animal is None and cellview.animals:
            self._block_cell_temporarily(cell_coords, duration=7)
            if self.tracking_cell == cell_coords:
                self.tracking_cell = None
        return animal

    def _prune_unavailable_animals(self, cellview: CellView) -> None:
        """Drop unavailable animals that left the cell so we can reconsider later."""
        if not self.unavailable_animals:
            return
        self.unavailable_animals.intersection_update(cellview.animals)

    def _maybe_release_for_priority(self, cellview: CellView) -> Optional[Action]:
        """Free flock space when a rarer animal is available in the current cell."""
        if not self.is_flock_full():
            return None

        candidate, candidate_score = self._best_animal_in_cell(cellview)
        if candidate is None or candidate_score is None:
            return None

        worst_animal = max(self.flock, key=lambda a: self._score_animal(a))
        if self._score_animal(worst_animal) > candidate_score:
            return Release(worst_animal)

        return None

    def _select_move_target(self) -> tuple[float, float]:
        """Decide which coordinate to move towards this turn."""
        if self._tracking_target_active() and self.tracking_cell:
            return (float(self.tracking_cell[0]), float(self.tracking_cell[1]))

        if (
            self.patrol_target is None
            or _distance(*self.position, *self.patrol_target) < 0.5
        ):
            self._pick_new_patrol_target()

        if self.tracking_cell:
            return (float(self.tracking_cell[0]), float(self.tracking_cell[1]))

        if self.patrol_target:
            return self.patrol_target

        return self.ark_position

    def _random_safe_step(self) -> tuple[float, float]:
        """Fallback jitter to keep helpers moving even when stuck."""
        if self.kind == Kind.Noah:
            return self.position

        for _ in range(20):
            angle = random.uniform(0, math.tau)
            distance = random.uniform(0.4, c.MAX_DISTANCE_KM * 0.95)
            dx = math.cos(angle) * distance
            dy = math.sin(angle) * distance
            candidate = (self.position[0] + dx, self.position[1] + dy)
            if self.can_move_to(*candidate):
                return candidate

        return self.move_towards(*(self._random_point_in_safe_area()))

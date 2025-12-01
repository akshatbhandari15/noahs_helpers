"""Player7: fresh helper redesign with stable pursuit and clear phases."""

from __future__ import annotations
import math
from collections import deque

from core.action import Action, Move, Obtain, Release
from core.message import Message
from core.player import Player
from core.snapshots import HelperSurroundingsSnapshot
from core.views.player_view import Kind
import core.constants as c


class Player7(Player):
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

        # Tunables consolidated into config for easy tuning
        self.config = {
            "linger_turns": 3,
            "block_after_fail": 3,  # Retry failed cells faster
            "pursuit_hyst": 1.5,
            "pursuit_lock": 2,
            "recent_len": 10,
            "eta_buffer": 10,
            "recent_bias": 1.3,
            "swap_threshold": 1.5,
            "stuck_threshold": 3,
            "give_up_turns": 7,
        }

        # Rarity and territory
        self.rarity = self._compute_rarity()
        self.territory = self._compute_territory()

        # Linear formation state (optional for coordinated sweeps)
        self._formation_spacing = c.MAX_SIGHT_KM * 0.8  # Stay within sight

        # Deterministic sweep state within territory
        self._sweep_index: int | None = None

        # State
        self.turn = 0
        self.is_raining = False
        self._rain_started_at: int | None = None
        self.time_elapsed = 0
        self.last_snapshot: HelperSurroundingsSnapshot | None = None

        # Knowledge
        self.ark_status: dict[int, dict] = {}
        self.known: dict[tuple[int, int, int], dict] = {}

        # Simple statistics for experiments/reporting
        self.stats = {
            "cells_in_territory": set(),
            "cells_out_territory": set(),
            "animals_collected": set(),
        }

        # Behavior
        self._intend_obtain = False
        self._linger_until = 0
        self._blocked: dict[tuple[int, int], int] = {}
        self._recent: deque[tuple[int, int]] = deque(maxlen=self.config["recent_len"])
        self._cell_cache: dict[tuple[int, int], tuple[float, int]] = {}
        self._explored: set[tuple[int, int]] = set()  # Track explored

        # Pursuit
        self._tgt_cell: tuple[int, int] | None = None
        self._tgt_score: float = -1.0
        self._tgt_expires = 0
        self._lock_until = 0
        self._last_dist: float | None = None
        self._stuck = 0
        # Track chase attempts per cell to avoid infinite chasing
        self._chase_attempts: dict[tuple[int, int], int] = {}
        # Cells we have explicitly decided to ignore (not worth chasing)
        self._ignored_cells: set[tuple[int, int]] = set()
        self._prev_flock_size = 0  # Track for failed obtain detection

        # Animals that proved invalid (caught by someone else / unreachable).
        # Once added here, they are never targeted again.
        self._ignored_animals: set = set()

        # Messaging: track what others are carrying and claiming
        self._seen_carrying: dict[tuple[int, int], int] = {}
        self._claimed: dict[tuple[int, int], int] = {}

        # Communication system from comms_player
        self.priorities: set[tuple[int, int]] = set()
        self.messages_sent: set[int] = set()
        self.messages_to_send: list[int] = []
        self.last_seen_ark_animals: set[tuple[int, int]] = set()

        # Waiting position near ark for when multiple helpers return
        # Spread helpers in a circle around ark
        self._waiting_position = self._compute_waiting_position()

    def check_surroundings(self, snapshot: HelperSurroundingsSnapshot) -> int:
        self.last_snapshot = snapshot
        self._update_state(snapshot)

        # Initialize priorities on first turn
        if self.turn == 1 and self.kind != Kind.Noah:
            for letter in self.species_populations.keys():
                sid = ord(letter) - ord("a")
                self.priorities.add((sid, 0))  # Male
                self.priorities.add((sid, 1))  # Female

        # Process ark view for communication
        if snapshot.ark_view:
            import heapq

            current_ark = {
                (a.species_id, a.gender.value) for a in snapshot.ark_view.animals
            }
            new_animals = current_ark - self.last_seen_ark_animals

            for sid, gender in new_animals:
                self.priorities.discard((sid, gender))
                # Ark message: bits 3-7=species, bit 2=gender, bit 1=ARK
                msg = (sid << 3) | (gender << 2) | 0b00000010
                if msg not in self.messages_sent:
                    heapq.heappush(self.messages_to_send, msg)
                    self.messages_sent.add(msg)

            self.last_seen_ark_animals = current_ark

        # Send queued message or default
        if self.messages_to_send:
            import heapq

            return heapq.heappop(self.messages_to_send)

        return self._encode_message()

    def get_action(self, messages: list[Message]) -> Action | None:
        if self.kind == Kind.Noah:
            return None

        self._process_messages(messages)

        # Debug summary each turn
        # try:
        #     flock_summary = [
        #         (a.species_id, getattr(a.gender, "name", str(a.gender)))
        #         for a in self.flock
        #     ]
        # except Exception:
        #     flock_summary = []
        # print(
        #     f"[P7] t={self.turn} pos={self.position} flock={flock_summary}, territory={self.territory}"
        # )

        # If in ark with empty flock
        if self.is_in_ark() and len(self.flock) == 0:
            # Clear behaviors
            self._linger_until = 0
            self._tgt_cell = None
            self._stuck = 0

            dist_to_ark = self._dist_to_ark()

            # If raining, calculate urgency
            if self.is_raining and self._rain_started_at is not None:
                elapsed = self.time_elapsed - self._rain_started_at
                left = c.START_RAIN - elapsed
                turns_to_ark = math.ceil(dist_to_ark / c.MAX_DISTANCE_KM)

                # If time is tight (less than 50 turns buffer), go direct
                if turns_to_ark + 50 >= left:
                    return None

            t = self.territory
            return self._move_to((t["cx"], t["cy"]))

        if self._should_return():
            # When returning, prioritize getting to ark
            if self.is_in_ark():
                return None

            # Clear all active behaviors when returning to ark
            self._linger_until = 0
            self._tgt_cell = None
            self._stuck = 0

            dist_to_ark = self._dist_to_ark()

            # If raining, calculate urgency
            if self.is_raining and self._rain_started_at is not None:
                elapsed = self.time_elapsed - self._rain_started_at
                left = c.START_RAIN - elapsed
                turns_to_ark = math.ceil(dist_to_ark / c.MAX_DISTANCE_KM)

                # If time is tight (less than 50 turns buffer), go direct
                if turns_to_ark + 50 >= left:
                    return self._move_to(self.ark_position)

                # If we have animals, go directly to ark
                if len(self.flock) > 0:
                    return self._move_to(self.ark_position)

            # If very close to ark, go to exact position
            if dist_to_ark <= 5:
                return self._move_to(self.ark_position)

            # If raining, go to ark (not waiting position)
            if self.is_raining:
                return self._move_to(self.ark_position)

            # Otherwise move toward waiting position near ark
            # This spreads helpers around ark when many return
            return self._move_to(self.ark_position)

        if self._should_offload():
            if self.is_in_ark():
                # In ark with full flock - offload everything
                if len(self.flock) > 0:
                    for animal in self.flock:
                        return Release(animal)
                return None
            # Clear all active behaviors when offloading
            self._linger_until = 0
            self._tgt_cell = None
            self._stuck = 0
            return self._move_to(self.ark_position)

        if self.turn < self._linger_until:
            a = self._best_here()
            if a is not None:
                self._intend_obtain = True
                return Obtain(a)

        comp = self._best_visible_completer()
        here_val = self._best_value_here()
        if comp is not None:
            pos, val = comp
            if val > here_val * 1.5:
                if self.is_flock_full():
                    rls = self._lowest_in_flock()
                    if rls is not None:
                        return Release(rls)
                return self._move_to(pos)

        a = self._best_here()
        if a is not None:
            if self.is_flock_full():
                r = self._choose_release(a)
                if r is not None:
                    return Release(r)
            else:
                self._intend_obtain = True
                # Broadcast that we're obtaining this animal
                import heapq

                self.priorities.discard((a.species_id, a.gender.value))
                msg = (a.species_id << 3) | (a.gender.value << 2) | 0b00000001
                if msg not in self.messages_sent:
                    heapq.heappush(self.messages_to_send, msg)
                    self.messages_sent.add(msg)
                return Obtain(a)

        mv = self._pursue_best_cell()
        if mv is not None:
            return mv

        # If flock is full and no valuable targets, return to ark
        if self.is_flock_full():
            if not self.is_in_ark():
                self._tgt_cell = None
                self._stuck = 0
                return self._move_to(self.ark_position)
            # In ark with full flock - offload
            if len(self.flock) > 0:
                for animal in self.flock:
                    return Release(animal)
            return None

        return self._explore()

    # -------- State & messages --------

    def _update_state(self, snap: HelperSurroundingsSnapshot) -> None:
        self.turn += 1
        self.time_elapsed = snap.time_elapsed

        # Save flock size before processing turn (for chase detection)
        self._prev_flock_size = len(self.flock)

        self.is_raining = snap.is_raining
        if self.is_raining and self._rain_started_at is None:
            self._rain_started_at = snap.time_elapsed

        self.position = snap.position
        self._recent.append((int(self.position[0]), int(self.position[1])))

        # Mark current position and all visible cells as explored
        curr_cell = (int(self.position[0]), int(self.position[1]))
        self._explored.add(curr_cell)

        # Track unique cells this helper has visited inside vs outside territory
        t = self.territory
        if (
            t["min_x"] <= curr_cell[0] <= t["max_x"]
            and t["min_y"] <= curr_cell[1] <= t["max_y"]
        ):
            self.stats["cells_in_territory"].add(curr_cell)
        else:
            self.stats["cells_out_territory"].add(curr_cell)

        for cv in snap.sight:
            self._explored.add((cv.x, cv.y))

        prev_flock = set(self.flock)
        self.flock = snap.flock.copy()

        # Track unique animals collected by this helper
        new_animals = set(self.flock) - prev_flock
        if new_animals:
            self.stats["animals_collected"].update(new_animals)

        expired = [p for p, t in self._blocked.items() if t <= self.turn]
        for p in expired:
            del self._blocked[p]

        # Periodic cleanup: if too many blocked cells, clear old ones
        if len(self._blocked) > 20:
            cutoff = self.turn - 10
            old = [p for p, t in self._blocked.items() if t < cutoff]
            for p in old:
                del self._blocked[p]

        if self._intend_obtain:
            # If flock grew since last turn, we successfully obtained something
            if len(self.flock) > len(prev_flock):
                self._linger_until = self.turn + self.config["linger_turns"]
            else:
                # Treat this cell as failed for a while to avoid re-trying
                cx, cy = int(self.position[0]), int(self.position[1])
                self._blocked[(cx, cy)] = self.turn + self.config["block_after_fail"]
        self._intend_obtain = False

        if snap.ark_view is not None:
            self._update_ark(snap.ark_view.animals)

        for cv in snap.sight:
            for an in cv.animals:
                key = (cv.x, cv.y, an.species_id)
                self.known[key] = {
                    "sid": an.species_id,
                    "gender": an.gender,
                    "pos": (cv.x, cv.y),
                    "seen": self.turn,
                }

    def _encode_message(self) -> int:
        if not self.flock:
            return 0
        # Encode: bits 0-4=species, bit 5=gender, bit 6=have, bit 7=claim
        best = max(self.flock, key=lambda a: self._value(a.species_id, a.gender))
        sid = best.species_id & 0x1F
        msg = sid
        from core.animal import Gender

        if best.gender == Gender.Female:
            msg |= 1 << 5
        msg |= 1 << 6  # Have this animal
        msg |= 1 << 7  # Claiming (avoid duplicates)
        return msg

    def _process_messages(self, messages: list[Message]) -> None:
        from core.animal import Gender
        import heapq

        # Expire old claims/sightings
        to_rm = [k for k, v in self._claimed.items() if v < self.turn - 20]
        for k in to_rm:
            del self._claimed[k]
        to_rm = [k for k, v in self._seen_carrying.items() if v < self.turn - 20]
        for k in to_rm:
            del self._seen_carrying[k]

        for m in messages:
            b = m.contents

            # Decode message
            from_ark = bool(b & 0b00000010)
            from_local = bool(b & 0b00000001)
            gender = (b & 0b00000100) >> 2
            sid = (b & 0b11111000) >> 3

            # Handle ark messages (release if we have it)
            if from_ark:
                for a in self.flock:
                    if a.species_id == sid and a.gender.value == gender:
                        # Mark for immediate release in get_action
                        self.priorities.discard((sid, gender))
                        break

            # Handle local helper messages (update priorities)
            if from_local:
                self.priorities.discard((sid, gender))

            # Forward message to neighbors
            if self.last_snapshot:
                neighbor_ids = {
                    h.id
                    for cv in self.last_snapshot.sight
                    for h in cv.helpers
                    if h.id != self.id
                }
                if b not in self.messages_sent and any(
                    n != m.from_helper.id for n in neighbor_ids
                ):
                    heapq.heappush(self.messages_to_send, b)
                    self.messages_sent.add(b)

            # Legacy protocol support
            female = (b >> 5) & 1
            have = (b >> 6) & 1
            claiming = (b >> 7) & 1
            g = Gender.Female if female else Gender.Male
            if have:
                self._seen_carrying[(sid, g.value)] = self.turn
            if claiming:
                self._claimed[(sid, g.value)] = self.turn

    # -------- Decision helpers --------

    def _should_return(self) -> bool:
        # Calculate distance to ark and turns needed
        dist_to_ark = self._dist_to_ark()
        turns_to_ark = math.ceil(dist_to_ark / c.MAX_DISTANCE_KM)

        if self.is_raining and self._rain_started_at is not None:
            # Rain has started - must reach ark well before T
            elapsed = self.time_elapsed - self._rain_started_at
            left = c.START_RAIN - elapsed

            # Add generous buffer to account for obstacles, detours, etc.
            # Need to be back at least 20 turns before T (T-20)
            buffer = 20

            # Return if we don't have enough time left
            if turns_to_ark + buffer >= left:
                return True

            # If we have animals, return even earlier to be safe
            if len(self.flock) > 0 and turns_to_ark + buffer + 10 >= left:
                return True

        # After 7 days (1008 turns): ensure we can get back if rain starts
        # We need to be within START_RAIN distance minus large buffer
        if self.time_elapsed >= 1008:
            # Stay within safe distance with 150 turn buffer
            max_safe_distance = (c.START_RAIN - 150) * c.MAX_DISTANCE_KM
            if dist_to_ark > max_safe_distance:
                return True

        return False

    def _should_offload(self) -> bool:
        if self.is_flock_full():
            return True
        if len(self.flock) >= 3:
            vals = [self._value(a.species_id, a.gender) for a in self.flock]
            vals.sort(reverse=True)
            return vals[0] >= 90 and vals[1] >= 80
        return False

    def _best_here(self):
        snap = self.last_snapshot
        if snap is None:
            return None
        cx, cy = int(self.position[0]), int(self.position[1])
        exp = self._blocked.get((cx, cy))
        if exp is not None and exp > self.turn:
            return None
        animals = None
        for cv in snap.sight:
            if cv.x == cx and cv.y == cy:
                animals = list(cv.animals)
                break
        if not animals:
            return None
        best = None
        best_val = -1.0
        best_comp = None
        best_comp_val = -1.0
        for a in animals:
            # Permanently ignored animals (already failed/caught by others)
            if a in self._ignored_animals:
                continue
            if a in self.flock:
                # print(f"[P7] best_here skip: in flock sid={a.species_id} g={a.gender}")
                continue
            # Skip exact species+gender duplicates (we already carry one)
            if any(
                f.species_id == a.species_id and f.gender == a.gender
                for f in self.flock
            ):
                # print(
                #     f"[P7] best_here skip: dup type in flock sid={a.species_id} g={a.gender}"
                # )
                continue
            # Skip animals already in the ark
            if self._is_in_ark(a.species_id, a.gender):
                # print(f"[P7] best_here skip: in ark sid={a.species_id} g={a.gender}")
                continue
            # Skip claimed targets from other helpers
            if (a.species_id, a.gender.value) in self._claimed:
                # print(f"[P7] best_here skip: claimed sid={a.species_id} g={a.gender}")
                continue

            # Prioritize animals in priority set
            is_priority = (a.species_id, a.gender.value) in self.priorities
            val = self._value(a.species_id, a.gender)

            if is_priority:
                val *= 1.5  # Boost priority animals

            if self._would_complete(a.species_id, a.gender):
                if val > best_comp_val:
                    best_comp_val = val
                    best_comp = a
            if val > best_val:
                best_val = val
                best = a
        return best_comp or best

    def _best_value_here(self) -> float:
        snap = self.last_snapshot
        if snap is None:
            return 0.0
        cx, cy = int(self.position[0]), int(self.position[1])
        best = 0.0
        for cv in snap.sight:
            if cv.x == cx and cv.y == cy:
                for a in cv.animals:
                    best = max(best, self._value(a.species_id, a.gender))
                break
        return best

    def _lowest_in_flock(self):
        worst = None
        worst_val = float("inf")
        for a in self.flock:
            v = self._value(a.species_id, a.gender)
            if v < worst_val:
                worst_val = v
                worst = a
        return worst

    def _choose_release(self, target):
        if not self._would_complete(target.species_id, target.gender):
            return None
        worst = self._lowest_in_flock()
        if worst is None:
            return None
        tv = self._value(target.species_id, target.gender)
        wv = self._value(worst.species_id, worst.gender)
        # Only swap if target is significantly better
        return worst if tv > wv * self.config["swap_threshold"] else None

    def _best_visible_completer(self):
        snap = self.last_snapshot
        if snap is None:
            return None
        cx, cy = int(self.position[0]), int(self.position[1])
        best_pos = None
        best_val = -1.0
        for cv in snap.sight:
            if (cv.x, cv.y) == (cx, cy):
                continue
            exp = self._blocked.get((cv.x, cv.y))
            if exp is not None and exp > self.turn:
                continue
            cell_best = -1.0
            for a in cv.animals:
                if a in self._ignored_animals:
                    continue
                # Skip duplicates in flock
                if any(
                    f.species_id == a.species_id and f.gender == a.gender
                    for f in self.flock
                ):
                    # print(
                    #     f"[P7] completer skip: dup type in flock sid={a.species_id} g={a.gender} at={(cv.x, cv.y)}"
                    # )
                    continue
                # Skip animals already in the ark
                if self._is_in_ark(a.species_id, a.gender):
                    # print(
                    #     f"[P7] completer skip: in ark sid={a.species_id} g={a.gender} at={(cv.x, cv.y)}"
                    # )
                    continue
                if self._would_complete(a.species_id, a.gender):
                    cell_best = max(cell_best, self._value(a.species_id, a.gender))
            if cell_best < 0:
                continue
            if cell_best > best_val:
                best_val = cell_best
                best_pos = (cv.x, cv.y)
        if best_pos is None:
            return None
        return best_pos, best_val

    def _pursue_best_cell(self) -> Move | None:
        snap = self.last_snapshot
        if snap is None:
            return None
        curr = (int(self.position[0]), int(self.position[1]))

        # Clean up old chase attempts
        if self.turn % 50 == 0:
            self._chase_attempts.clear()
            self._ignored_cells.clear()

        # Check if we have an active target
        if (
            self._tgt_cell is not None
            and self._tgt_expires > self.turn
            and self._blocked.get(self._tgt_cell, 0) <= self.turn
        ):
            if self._tgt_cell == curr:
                # Reached target - check if we obtained an animal
                current_flock_size = len(self.flock)
                if self._prev_flock_size == current_flock_size:
                    # Flock didn't grow - animal likely moved away
                    # Any animals still reported in that cell are now
                    # treated as invalid targets for this helper.
                    for cv in snap.sight:
                        if (cv.x, cv.y) == self._tgt_cell:
                            for a in cv.animals:
                                self._ignored_animals.add(a)
                            break
                    cell_key = self._tgt_cell
                    self._chase_attempts[cell_key] = (
                        self._chase_attempts.get(cell_key, 0) + 1
                    )
                    # After 2 failed attempts at same cell, block it
                    if self._chase_attempts[cell_key] >= 2:
                        self._blocked[self._tgt_cell] = self.turn + 20
                        # Also mark this cell as ignored so we don't retarget it
                        self._ignored_cells.add(self._tgt_cell)
                        self._tgt_cell = None
                        self._last_dist = None
                        self._stuck = 0
                        self._recent.clear()
                        return None
                else:
                    # Success! Clear chase attempts for this cell
                    if self._tgt_cell in self._chase_attempts:
                        del self._chase_attempts[self._tgt_cell]
                # Clear target and continue
                self._tgt_cell = None
                self._last_dist = None
                self._stuck = 0
            else:
                # Check if making progress toward target
                tx, ty = self._tgt_cell
                dx = tx - self.position[0]
                dy = ty - self.position[1]
                d = max(0.0, math.hypot(dx, dy))

                if self._last_dist is not None:
                    if d >= self._last_dist - 1e-6:
                        self._stuck += 1
                    else:
                        self._stuck = 0

                self._last_dist = d

                if self._stuck >= self.config["stuck_threshold"]:
                    # Give up on this target
                    self._blocked[self._tgt_cell] = (
                        self.turn + self.config["give_up_turns"]
                    )
                    # Mark as ignored so we don't immediately reselect it
                    self._ignored_cells.add(self._tgt_cell)
                    self._tgt_cell = None
                    self._last_dist = None
                    self._stuck = 0
                    self._recent.clear()
                else:
                    # Continue toward target only if still valuable
                    target_val = 0.0
                    for cv in snap.sight:
                        num_helpers = len(cv.helpers)
                        if num_helpers > 0:
                            continue
                        if (cv.x, cv.y) == self._tgt_cell:
                            for a in cv.animals:
                                if a in self._ignored_animals:
                                    continue
                                if not any(
                                    f.species_id == a.species_id
                                    and f.gender == a.gender
                                    for f in self.flock
                                ):
                                    # Skip if already in ark
                                    if not self._is_in_ark(a.species_id, a.gender):
                                        target_val += self._value(
                                            a.species_id, a.gender
                                        )
                            break
                    # Only continue if target still has value
                    if target_val >= 5:
                        return self._move_to(self._tgt_cell)
                    else:
                        # Target no longer valuable
                        self._tgt_cell = None
                        self._stuck = 0

        # Find best animal to target
        best_cell = None
        best_score = -1.0

        for cv in snap.sight:
            tx, ty = cv.x, cv.y
            if (tx, ty) == curr:
                continue

            num_helpers = len(cv.helpers)
            if num_helpers > 0:
                continue

            # Skip cells we've explicitly decided to ignore
            if (tx, ty) in self._ignored_cells:
                # print(f"[P7] pursue skip ignored cell={(tx, ty)}")
                continue
            exp = self._blocked.get((tx, ty))
            if exp is not None and exp > self.turn:
                continue

            # Evaluate each animal in this cell
            for a in cv.animals:
                if a in self._ignored_animals:
                    continue
                if any(
                    f.species_id == a.species_id and f.gender == a.gender
                    for f in self.flock
                ):
                    # print(
                    #     f"[P7] pursue skip: dup type in flock sid={a.species_id} g={a.gender} at={(tx, ty)}"
                    # )
                    continue
                # Skip animals already in the ark
                if self._is_in_ark(a.species_id, a.gender):
                    # print(
                    #     f"[P7] pursue skip: in ark sid={a.species_id} g={a.gender} at={(tx, ty)}"
                    # )
                    continue
                if (a.species_id, a.gender.value) in self._claimed:
                    # print(
                    #     f"[P7] pursue skip: claimed sid={a.species_id} g={a.gender} at={(tx, ty)}"
                    # )
                    continue

                animal_val = self._value(a.species_id, a.gender)
                if animal_val <= 0:
                    # print(
                    #     f"[P7] pursue skip: low value={animal_val} sid={a.species_id} g={a.gender} at={(tx, ty)}"
                    # )
                    continue

                dx = tx - self.position[0]
                dy = ty - self.position[1]
                dist = max(1.0, math.hypot(dx, dy))
                score = animal_val / dist

                # Penalize recent cells
                if (tx, ty) in self._recent:
                    score *= 0.5

                # Strong bonus for very close animals (likely obtainable)
                if dist <= 2.0:
                    score *= 1.5
                # Bonus for cells we can reach in 1 turn
                elif dist <= c.MAX_DISTANCE_KM:
                    score *= 1.3

                if score > best_score:
                    best_score = score
                    best_cell = (tx, ty)

        # if best_cell is not None:
        #     print(
        #         f"[P7] pursue best_cell={best_cell} best_score={best_score} pos={self.position}"
        #     )

        if best_cell is None:
            # No valuable cells, clear history
            if len(self._recent) > 5:
                self._recent.clear()
            return None

        # If the best score is too low, treat this as a conscious decision
        # to not chase anything right now and continue exploration instead.
        if best_score < 1:
            if best_cell is not None:
                # print(
                #     f"[P7] pursue ignore low-score cell={best_cell} score={best_score}"
                # )
                self._ignored_cells.add(best_cell)
            return None

        # Set new target
        self._tgt_cell = best_cell
        self._tgt_score = best_score
        self._tgt_expires = self.turn + 5  # Shorter expiry
        self._lock_until = self.turn + 1  # Minimal lock
        dx = best_cell[0] - self.position[0]
        dy = best_cell[1] - self.position[1]
        self._last_dist = max(0.0, math.hypot(dx, dy))
        self._stuck = 0
        print(f"[P7] pursue set target cell={best_cell} score={best_score}")
        return self._move_to(best_cell)

    # -------- Scoring --------

    def _is_in_ark(self, sid: int, gender) -> bool:
        """Check if this species+gender (or species as a whole) is already in the ark.

        For unknown gender, we treat the species as "in ark" if both genders
        are already present. This prevents chasing distant animals of a
        fully-saved species whose gender we cannot yet see.
        """
        from core.animal import Gender

        info = self.ark_status.get(sid)
        if info is None:
            return False

        # If gender is unknown, species is effectively in the ark only if both
        # genders have been saved.
        if gender is None or gender == Gender.Unknown:
            in_ark = info.get(Gender.Male, False) and info.get(Gender.Female, False)
            # if in_ark:
            # print(
            #     f"[P7] _is_in_ark: skip unknown-gender of fully-saved species sid={sid}"
            # )
            return in_ark

        return info.get(gender, False)

    def _would_complete(self, sid: int, gender) -> bool:
        from core.animal import Gender

        info = self.ark_status.get(sid, {Gender.Male: False, Gender.Female: False})
        if gender == Gender.Male:
            return info[Gender.Female] and not info[Gender.Male]
        if gender == Gender.Female:
            return info[Gender.Male] and not info[Gender.Female]
        return False

    def _value(self, sid: int, gender) -> float:
        from core.animal import Gender

        base = self.rarity.get(sid, 1.0)
        info = self.ark_status.get(sid, {Gender.Male: False, Gender.Female: False})
        has_m = info[Gender.Male]
        has_f = info[Gender.Female]

        # If we already carry both genders of this species in our own flock,
        # treat all further animals of this species as worthless, regardless
        # of their gender. This prevents chasing already-complete species.
        have_m_in_flock = any(
            f.species_id == sid and f.gender == Gender.Male for f in self.flock
        )
        have_f_in_flock = any(
            f.species_id == sid and f.gender == Gender.Female for f in self.flock
        )
        if have_m_in_flock and have_f_in_flock:
            return 0.0

        # Unknown gender: decide based on ark completion for this species.
        if gender is None or gender == Gender.Unknown:
            # Species already fully saved: no value.
            if has_m and has_f:
                return 0.0
            # Species not on ark at all: moderately high value.
            if not has_m and not has_f:
                return base * 60
            # Exactly one gender present: could complete the species.
            return base * 90

        # If we already carry this species+gender in flock, make it worthless
        # so pursuit scoring and selection ignore duplicates.
        if any(f.species_id == sid and f.gender == gender for f in self.flock):
            return 0.0

        if (gender == Gender.Male and has_f and not has_m) or (
            gender == Gender.Female and has_m and not has_f
        ):
            boost = 1.0
            if (sid, gender.value ^ 1) in self._seen_carrying:
                boost = 1.2
            return base * 100 * boost
        if not has_m and not has_f:
            return base * 80
        return base * 10

    def _update_ark(self, animals) -> None:
        from core.animal import Gender

        self.ark_status = {}
        for a in animals:
            if a.species_id not in self.ark_status:
                self.ark_status[a.species_id] = {
                    Gender.Male: False,
                    Gender.Female: False,
                }
            if a.gender != Gender.Unknown:
                self.ark_status[a.species_id][a.gender] = True

    # -------- Movement & exploration --------

    def _move_to(self, pos: tuple[float, float]) -> Move:
        nx, ny = self.move_towards(pos[0], pos[1])
        # Clamp to valid field boundaries
        nx = max(0, min(nx, c.X - 1))
        ny = max(0, min(ny, c.Y - 1))

        # When moving toward the ark, snap exactly onto ark position if
        # we are within EPS. This ensures helpers actually count as being
        # in the ark at the end of the game.
        ark_x, ark_y = self.ark_position
        if abs(nx - ark_x) <= c.EPS and abs(ny - ark_y) <= c.EPS:
            nx = float(ark_x)
            ny = float(ark_y)

        return Move(nx, ny)

    def _explore(self) -> Move:
        """Deterministic serpentine sweep within biased-outward territory."""

        # Late-game safety: don't explore beyond safe return distance
        if self.time_elapsed >= 1008:
            dist_to_ark = self._dist_to_ark()
            max_safe_distance = (c.START_RAIN - 150) * c.MAX_DISTANCE_KM
            if dist_to_ark > max_safe_distance:
                return self._move_to(self.ark_position)

        t = self.territory
        min_x, max_x = t["min_x"], t["max_x"]
        min_y, max_y = t["min_y"], t["max_y"]

        if min_x > max_x or min_y > max_y:
            return self._move_to(self.ark_position)

        # Use a small interior margin to avoid edges
        width = max(1, max_x - min_x)
        height = max(1, max_y - min_y)
        margin = min(3, width // 4, height // 4)

        sweep_min_x = min_x + margin
        sweep_max_x = max_x - margin
        sweep_min_y = min_y + margin
        sweep_max_y = max_y - margin

        if sweep_min_x > sweep_max_x or sweep_min_y > sweep_max_y:
            sweep_min_x, sweep_max_x = min_x, max_x
            sweep_min_y, sweep_max_y = min_y, max_y

        # Discrete serpentine waypoints across territory
        width_i = max(1, sweep_max_x - sweep_min_x + 1)
        height_i = max(1, sweep_max_y - sweep_min_y + 1)
        total_cells = width_i * height_i

        if self._sweep_index is None:
            # Stagger starting index based on helper id to further decorrelate
            self._sweep_index = (
                self.id * (total_cells // max(1, self.num_helpers))
            ) % total_cells

        idx = self._sweep_index % total_cells
        row = idx // width_i
        col = idx % width_i

        # Serpentine: even rows L->R, odd rows R->L
        if row % 2 == 0:
            x_tgt = sweep_min_x + col
        else:
            x_tgt = sweep_max_x - col
        y_tgt = sweep_min_y + row

        # Advance index when we are close enough to current waypoint
        cur_x, cur_y = self.position
        if abs(cur_x - x_tgt) <= 0.5 and abs(cur_y - y_tgt) <= 0.5:
            self._sweep_index = (self._sweep_index + 1) % total_cells

        return self._move_to((x_tgt, y_tgt))

    # -------- Setup helpers --------

    def _compute_waiting_position(self) -> tuple[float, float]:
        """Compute a waiting position near ark for this helper.
        Spreads helpers in a circle to avoid crowding."""
        ark_x, ark_y = self.ark_position

        # Place helpers in a circle around ark
        # Radius of 3-5km keeps them close but spread out
        radius = 4.0
        angle = (2 * math.pi * self.id) / max(1, self.num_helpers)

        wait_x = ark_x + radius * math.cos(angle)
        wait_y = ark_y + radius * math.sin(angle)

        # Clamp to valid field bounds
        wait_x = max(0, min(wait_x, c.X - 1))
        wait_y = max(0, min(wait_y, c.Y - 1))

        return (wait_x, wait_y)

    def _compute_rarity(self) -> dict[int, float]:
        if not self.species_populations:
            return {}
        mx = max(self.species_populations.values())
        out: dict[int, float] = {}
        for letter, pop in self.species_populations.items():
            sid = ord(letter) - ord("a")
            out[sid] = (mx / pop) if pop > 0 else (mx * 10.0)
        return out

    def _compute_territory(self) -> dict[str, int]:
        # Aspect-aware grid so territories are closer to square even when
        # c.X != c.Y. Each helper gets roughly equal area, but we
        # bias territories outward from the ark. With many helpers,
        # the inner region near the ark is covered mostly by transit
        # (fan-out and returns), so systematic sweeps focus further out.
        if self.num_helpers <= 0:
            return {
                "min_x": 0,
                "max_x": c.X - 1,
                "min_y": 0,
                "max_y": c.Y - 1,
                "cx": c.X // 2,
                "cy": c.Y // 2,
            }

        # Choose number of columns proportional to field aspect ratio.
        cols = max(1, math.ceil(math.sqrt(self.num_helpers * c.X / c.Y)))
        rows = max(1, math.ceil(self.num_helpers / cols))

        cell_width = c.X / cols
        cell_height = c.Y / rows

        row = self.id // cols
        col = self.id % cols
        sx = col * cell_width
        sy = row * cell_height

        min_x = int(sx)
        max_x = int(min(sx + cell_width, c.X - 1))
        min_y = int(sy)
        max_y = int(min(sy + cell_height, c.Y - 1))

        # Bias rectangle outward from ark depending on number of helpers.
        # More helpers -> stronger outward push so inner region near ark
        # is left to natural transit coverage.
        ark_x, ark_y = self.ark_position
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2

        # Direction from ark to territory center
        dx = cx - ark_x
        dy = cy - ark_y
        dist = math.hypot(dx, dy)

        # Target minimum radius for territory centers
        # Base radius plus a term that grows with sqrt of helpers
        base_radius = min(c.X, c.Y) * 0.15
        helper_boost = math.sqrt(max(1, self.num_helpers)) * (min(c.X, c.Y) * 0.02)
        target_radius = base_radius + helper_boost

        if dist > 1e-6 and dist < target_radius:
            scale = target_radius / dist
            new_cx = ark_x + dx * scale
            new_cy = ark_y + dy * scale
        else:
            new_cx, new_cy = float(cx), float(cy)

        # Recenter rectangle around new center, preserving size
        half_w = (max_x - min_x) / 2.0
        half_h = (max_y - min_y) / 2.0

        min_x = int(max(0, min(c.X - 1, new_cx - half_w)))
        max_x = int(max(0, min(c.X - 1, new_cx + half_w)))
        min_y = int(max(0, min(c.Y - 1, new_cy - half_h)))
        max_y = int(max(0, min(c.Y - 1, new_cy + half_h)))

        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "cx": int((min_x + max_x) / 2),
            "cy": int((min_y + max_y) / 2),
        }

    # -------- Utils --------

    def _dist_to_ark(self) -> float:
        dx = self.ark_position[0] - self.position[0]
        dy = self.ark_position[1] - self.position[1]
        return math.hypot(dx, dy)

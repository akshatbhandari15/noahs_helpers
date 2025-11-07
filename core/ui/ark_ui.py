from typing import Literal
import pygame

from core.animal import Animal, Gender
from core.ark import Ark
from core.engine import Engine
import core.constants as c
from core.player import Player


def coords_to_px(x: float, y: float) -> tuple[float, float]:
    x_px = c.LANDSCAPE_WEST_PX + (c.LANDSCAPE_EAST_PX - c.LANDSCAPE_WEST_PX) * x / c.X
    y_px = (
        c.LANDSCAPE_NORTH_PX + (c.LANDSCAPE_SOUTH_PX - c.LANDSCAPE_NORTH_PX) * y / c.Y
    )
    return x_px, y_px


def km_to_px(km: float) -> float:
    return c.LANDSCAPE_HEIGHT * km / c.Y


def is_hovered_circle(
    mouse_pos: tuple[int, int], center: tuple[float, float], radius: float
) -> bool:
    mx, my = mouse_pos
    cx, cy = center
    return (mx - cx) ** 2 + (my - cy) ** 2 <= radius**2


class ArkUI:
    def __init__(self, engine: Engine) -> None:
        pygame.init()

        self.engine = engine
        self.running = True
        self.paused = True
        self.clock = pygame.time.Clock()

        self.turn = 0

        self.screen = pygame.display.set_mode((c.SCREEN_WIDTH, c.SCREEN_HEIGHT))
        self.bg_color = c.BG_COLOR
        self.big_font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 28)

        self.debug_mode = False

        self.drawn_objects: dict[
            tuple[tuple[float, float], float], Ark | Player | Animal
        ] = {}

    def write_at(
        self,
        font: pygame.font.Font,
        line: str,
        coord: tuple[int, int],
        align: Literal["left", "center", "right"] = "center",
        color=(0, 0, 0),
    ):
        text = font.render(line, True, color)

        # get rectangle to center the text
        match align:
            case x if x == "left":
                rect = text.get_rect()
                rect.midleft = coord
            case x if x == "center":
                rect = text.get_rect(center=coord)
            case x if x == "right":
                rect = text.get_rect()
                rect.midright = coord
            case _:
                raise Exception(f"invalid value for `align`: {align}")

        self.screen.blit(text, rect)

    def draw_grid(self):
        """Draw garden boundaries and grid."""
        border_rect = pygame.Rect(
            c.MARGIN_X, c.MARGIN_Y, c.LANDSCAPE_WIDTH, c.LANDSCAPE_HEIGHT
        )
        color = c.DAMP_GRASS_COLOR if self.engine.is_raining() else c.GRASS_COLOR
        pygame.draw.rect(self.screen, color, border_rect)  # fill
        pygame.draw.rect(self.screen, (0, 0, 0), border_rect, 2)  # border

        for i in range(c.NUM_GRID_LINES + 1):
            x = c.MARGIN_X + int(c.LANDSCAPE_WIDTH * i / c.NUM_GRID_LINES)

            if i not in [0, c.NUM_GRID_LINES]:
                # only draw lines inside the grid
                pygame.draw.line(
                    self.screen,
                    c.GRIDLINE_COLOR,
                    (x, c.MARGIN_Y),
                    (x, c.MARGIN_Y + c.LANDSCAPE_HEIGHT),
                )

            line = "X"
            if i:
                val = c.X * i / c.NUM_GRID_LINES
                line = f"{int(val)}" if val.is_integer() else f"{val:.1f}"
            self.write_at(self.small_font, line, (x, c.MARGIN_Y - 20))

        for i in range(c.NUM_GRID_LINES + 1):
            y = c.MARGIN_Y + int(c.LANDSCAPE_HEIGHT * i / c.NUM_GRID_LINES)

            if i not in [0, c.NUM_GRID_LINES]:
                pygame.draw.line(
                    self.screen,
                    c.GRIDLINE_COLOR,
                    (c.MARGIN_X, y),
                    (c.MARGIN_X + c.LANDSCAPE_WIDTH, y),
                )

            line = "Y"
            if i:
                val = c.Y * i / c.NUM_GRID_LINES
                line = f"{int(val)}" if val.is_integer() else f"{val:.1f}"
            self.write_at(self.small_font, line, (c.MARGIN_X - 10, y), align="right")

    def draw_ark(self):
        ark_x, ark_y = self.engine.ark.position
        ark_center = coords_to_px(ark_x, ark_y)

        ark_img_orig = pygame.image.load("sprites/ark.png").convert_alpha()
        ark_img = pygame.transform.scale(
            ark_img_orig, (2.5 * c.ARK_RADIUS, 2.5 * c.ARK_RADIUS)
        )
        ark_rect = ark_img.get_rect(center=ark_center)
        self.screen.blit(ark_img, ark_rect)

        key = (ark_center, c.ARK_RADIUS)
        self.drawn_objects[key] = self.engine.ark

    def render_hover_view(self, title: str):
        cw = int(c.SCREEN_WIDTH / 2)
        ch = int(c.SCREEN_HEIGHT / 2)

        left = cw - int(c.HOVERED_WIDTH / 2)
        top = ch - int(c.HOVERED_HEIGHT / 2)

        rect = pygame.Rect(
            left,
            top,
            c.HOVERED_WIDTH,
            c.HOVERED_HEIGHT,
        )
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)

        pygame.draw.rect(
            overlay, (*c.BG_COLOR, 240), overlay.get_rect(), border_radius=10
        )
        self.screen.blit(overlay, rect.topleft)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, width=1, border_radius=10)

        self.write_at(self.big_font, title, (cw, ch - int(c.HOVERED_HEIGHT / 2) + 15))

        return left, top

    def draw_hovered_ark(self, pos: tuple[int, int]):
        left, top = self.render_hover_view("ARK")

        margined_x = left + c.HOVERED_MARGIN_X
        self.write_at(
            self.small_font,
            f"pos: {pos}",
            (margined_x, top + c.MARGIN_Y),
            align="left",
        )

        species_in_ark = self.engine.ark.get_species()

        y = top + c.MARGIN_Y * 2
        for sid, (has_male, has_female) in species_in_ark.items():
            self.write_at(self.small_font, f"{sid}: ", (margined_x, y), align="left")

            side = 40
            offset = 40

            male_rect = pygame.Rect(margined_x + offset, y - int(side / 2), side, side)
            female_rect = pygame.Rect(
                margined_x + offset + 5 + side, y - int(side / 2), side, side
            )
            if has_male:
                pygame.draw.rect(self.screen, c.MALE_ANIMAL_COLOR, male_rect)
            else:
                pygame.draw.rect(self.screen, c.MALE_ANIMAL_COLOR, male_rect, 2)

            if has_female:
                pygame.draw.rect(self.screen, c.FEMALE_ANIMAL_COLOR, female_rect)
            else:
                pygame.draw.rect(self.screen, c.FEMALE_ANIMAL_COLOR, female_rect, 2)

            y += 50

    def draw_helpers(self):
        for helper in self.engine.helpers:
            helper_x, helper_y = helper.position
            helper_center = coords_to_px(helper_x, helper_y)

            pygame.draw.circle(
                self.screen, c.HELPER_COLOR, helper_center, c.HELPER_RADIUS
            )
            self.drawn_objects[(helper_center, c.HELPER_RADIUS)] = helper

    def draw_hovered_helper(
        self, id: int, position: tuple[float, float], flock: set[Animal]
    ):
        left, top = self.render_hover_view(f"Helper {id}")

        y = top + c.MARGIN_Y

        margined_x = left + c.HOVERED_MARGIN_X
        self.write_at(
            self.small_font,
            f"pos: ({position[0]:.2f}, {position[1]:.2f})",
            (margined_x, y),
            align="left",
        )

        y += c.MARGIN_Y

        self.write_at(
            self.small_font,
            "Flock",
            (margined_x, y),
            align="left",
        )

        flist = list(flock) + [None] * (c.MAX_FLOCK_SIZE - len(flock))
        for i in range(c.MAX_FLOCK_SIZE):
            y += 30
            self.write_at(
                self.small_font, f"{i}: {flist[i]}", (margined_x + 10, y), align="left"
            )

        y += c.MARGIN_Y

        last_msg = self.engine.last_messages[id]
        if last_msg is not None:
            self.write_at(
                self.small_font,
                f"Last msg: 0b{last_msg:08b} = {last_msg}",
                (margined_x, y),
                align="left",
            )

    def draw_animals(self):
        for animal, cell in self.engine.free_animals.items():
            animal_center = coords_to_px(cell.x, cell.y)

            color = (
                c.MALE_ANIMAL_COLOR
                if animal.gender == Gender.Male
                else c.FEMALE_ANIMAL_COLOR
            )

            animal_rect = pygame.Rect(
                animal_center[0] - c.ANIMAL_RADIUS / 2,
                animal_center[1] - c.ANIMAL_RADIUS / 2,
                c.ANIMAL_RADIUS,
                c.ANIMAL_RADIUS,
            )
            pygame.draw.rect(self.screen, color, animal_rect, c.ANIMAL_RADIUS)

            self.drawn_objects[(animal_center, c.ANIMAL_RADIUS)] = animal

    def draw_hovered_animal(self, sid: int, gender: Gender, pos: tuple[int, int]):
        left, top = self.render_hover_view("Animal")

        y = top + c.MARGIN_Y

        margined_x = left + c.HOVERED_MARGIN_X
        self.write_at(
            self.small_font,
            f"pos: {pos}",
            (margined_x, y),
            align="left",
        )

        y += c.MARGIN_Y

        self.write_at(
            self.small_font,
            f"species_id: {sid}",
            (margined_x, y),
            align="left",
        )

        y += c.MARGIN_Y

        color = c.MALE_ANIMAL_COLOR if gender == Gender.Male else c.FEMALE_ANIMAL_COLOR

        self.write_at(
            self.small_font,
            f"gender: {gender}",
            (margined_x, y),
            align="left",
            color=color,
        )

    def draw_if_hovered(self):
        pos = pygame.mouse.get_pos()

        best_obj = None
        smallest_radius = -1
        for (center, radius), obj in self.drawn_objects.items():
            if is_hovered_circle(pos, center, radius):
                if best_obj is None or radius < smallest_radius:
                    best_obj = obj
                    smallest_radius = radius

        match best_obj:
            case Ark(position=p):
                self.draw_hovered_ark(p)
            case Player(id=id, position=p, flock=f):
                self.draw_hovered_helper(id, p, f)
            case Animal(species_id=sid, gender=g):
                cell = self.engine.free_animals[best_obj]
                self.draw_hovered_animal(sid, g, (cell.x, cell.y))

    def draw_objects(self):
        self.drawn_objects.clear()
        self.draw_ark()
        self.draw_helpers()
        self.draw_animals()

    def draw_info_panel(self):
        info_pane_x = c.LANDSCAPE_EAST_PX + c.MARGIN_X
        info_pane_y = c.MARGIN_Y

        info_lines = [
            f"{self.engine.helpers[0]}",
            "",
            f"Turn: {self.engine.time_elapsed}/{self.engine.time}",
            f"Score: {self.engine.ark.get_score()}",
            f"Helpers: {len(self.engine.helpers)}",
            f"is_raining: {self.engine.is_raining()}",
            f"{'DEBUG ON' if self.debug_mode else 'DEBUG OFF'}",  # NEW: Show debug status
        ]

        y = 0
        for i, line in enumerate(info_lines):
            y = info_pane_y + i * 30
            if line:  # Skip empty debug line when not in debug mode
                self.write_at(self.big_font, line, (info_pane_x, y), align="left")

        y += 40

        self.write_at(self.big_font, "Animals", (info_pane_x, y), align="left")
        for sid, (num_male, num_female) in self.engine.species_stats.items():
            y += 30
            self.write_at(
                self.big_font,
                f"{sid}:",
                (info_pane_x + 20, y),
                align="left",
            )
            self.write_at(
                self.big_font,
                f"{num_male}M",
                (info_pane_x + 60, y),
                align="left",
                color=c.MALE_ANIMAL_COLOR,
            )
            self.write_at(
                self.big_font,
                f"{num_female}F",
                (info_pane_x + 110, y),
                align="left",
                color=c.FEMALE_ANIMAL_COLOR,
            )

    def draw_debug_helper_screens(self):
        grid = pygame.Rect(
            c.MARGIN_X, c.MARGIN_Y, c.LANDSCAPE_WIDTH, c.LANDSCAPE_HEIGHT
        )

        mask = pygame.Surface((grid.w, grid.h), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 50))

        for helper in self.engine.helpers:
            grid_center = coords_to_px(*helper.position)
            center = grid_center[0] - grid.x, grid_center[1] - grid.y
            radius = km_to_px(c.MAX_SIGHT_KM)

            pygame.draw.circle(mask, (0, 0, 0, 0), center, radius)

        for helper in self.engine.helpers:
            grid_center = coords_to_px(*helper.position)
            center = grid_center[0] - grid.x, grid_center[1] - grid.y
            radius = km_to_px(c.MAX_SIGHT_KM)

            pygame.draw.circle(mask, (0, 0, 0, 255), center, radius, width=1)

        self.screen.blit(mask, grid.topleft)

    def draw_debug_info(self):
        if not self.debug_mode:
            return

        self.draw_debug_helper_screens()

    def step_simulation(self):
        """Run one turn of simulation."""
        if self.engine.time_elapsed < self.engine.time:
            self.engine.run_turn()
            self.turn += 1
        else:
            self.paused = True

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT and self.paused:
                    self.step_simulation()
                elif event.key == pygame.K_d:  # NEW: Toggle debug mode
                    self.debug_mode = not self.debug_mode

    def run(self) -> dict:
        while self.running:
            self.screen.fill(self.bg_color)
            self.draw_grid()
            self.draw_objects()
            self.draw_info_panel()
            self.draw_debug_info()
            self.draw_if_hovered()

            self.handle_events()

            if not self.paused:
                self.step_simulation()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

        return {}

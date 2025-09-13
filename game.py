from __future__ import annotations

import random
from typing import ClassVar, Optional

import pygame

# Type aliases
Board = list[list[Optional[str]]]

# Config
WIDTH: int = 480
HEIGHT: int = 640
SCORING: dict[int, int] = {1: 100, 2: 200, 3: 500, 4: 800}
CELL_SIZE: int = 25
FPS: int = 60

# Colors
BLACK: tuple[int, int, int] = (0, 0, 0)
WHITE: tuple[int, int, int] = (255, 255, 255)
RED: tuple[int, int, int] = (252, 91, 122)


class Tetromino:
    """
    Tetromino with position (x,y), type, and rotation.

    This class encapsulates the properties and behaviors of a Tetris game piece,
    including its position on the board, type (shape), current rotation, and
    provides methods to retrieve its layout and to rotate it in a clock-wise
    direction.

    Parameters
    ----------
    x : int
        The horizontal position of the tetromino on the board.
    y : int
        The vertical position of the tetromino on the board.
    """

    types: ClassVar[list[str]] = ["I", "J", "L", "O", "S", "T", "Z"]
    colors: ClassVar[dict[str, tuple[int, int, int]]] = {
        "I": (0, 255, 255),
        "J": (0, 0, 255),
        "L": (255, 128, 0),
        "O": (255, 255, 0),
        "S": (0, 255, 0),
        "T": (128, 0, 128),
        "Z": (255, 0, 0),
    }
    figures: ClassVar[dict[str, list[list[int]]]] = {
        "I": [[1, 5, 9, 13], [4, 5, 6, 7]],
        "J": [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        "L": [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        "O": [[1, 2, 5, 6]],
        "S": [[6, 7, 9, 10], [1, 5, 6, 10]],
        "T": [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        "Z": [[4, 5, 9, 10], [2, 6, 5, 9]],
    }  # 4x4 matrix

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.type = random.choice(Tetromino.types)
        self.rotation = 0

    def shape(self) -> list[int]:
        """Return the current rotation shape indices for the tetromino."""
        return self.figures[self.type][self.rotation]

    def rotate(self) -> None:
        """Rotate the tetromino in a clock-wise direction to the next rotation state."""
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])

    def clone(self) -> Tetromino:
        """Create and return a copy of this Tetromino, including its position, type, and rotation."""
        new_piece = Tetromino(self.x, self.y)
        new_piece.type = self.type
        new_piece.rotation = self.rotation
        return new_piece


class Tetris:
    """
    Tetris game that handles the game state, tetromino spawning, movement,
    line clearing, score tracking, rendering and event handling.

    Parameters
    ----------
    rows : int, optional
        Number of rows in the game board. Default: ``20``.
    cols : int, optional
        Number of columns in the game board. Default: ``10``.
    """

    def __init__(self, rows: int = 20, cols: int = 10) -> None:
        self.rows = rows
        self.cols = cols

        self.total_lines_cleared: int = 0
        self.score: int = 0
        self.level: int = 1
        self.last_fall_time: int = pygame.time.get_ticks()
        self.running: bool = True
        self.game_over: bool = False

        self.font: pygame.font.Font = pygame.font.Font("arial.ttf", 25)
        self.display: pygame.Surface = pygame.display.set_mode(
            (WIDTH, HEIGHT), flags=(pygame.RESIZABLE | pygame.SCALED)
        )
        pygame.display.set_caption("Tetris")
        self.clock: pygame.time.Clock = pygame.time.Clock()

        self.board: Board = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.curr_tetromino: Tetromino = Tetromino(4, 0)
        self.next_tetromino: Tetromino = Tetromino(4, 0)

    def is_running(self) -> bool:
        return self.running

    def play_step(self) -> None:
        """Handle one iteration of the main game loop."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN and not self.game_over:
                if event.key == pygame.K_LEFT:
                    self._move_sideways(-1)
                elif event.key == pygame.K_RIGHT:
                    self._move_sideways(1)
                elif event.key == pygame.K_UP:
                    self._rotate()
                elif event.key == pygame.K_DOWN:
                    self._soft_drop()
                elif event.key == pygame.K_SPACE:
                    self._hard_drop()

            elif event.type == pygame.KEYDOWN and self.game_over:
                if event.key == pygame.K_r:
                    self._reset()
                elif event.key == pygame.K_q:
                    self.running = False

        self._auto_drop()
        self._update_display()
        self.clock.tick(FPS)

    def _reset(self) -> None:
        """Reset the game state to its initial values."""
        self.total_lines_cleared = 0
        self.score = 0
        self.level = 1
        self.last_fall_time = pygame.time.get_ticks()
        self.running = True
        self.game_over = False
        self.board = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.curr_tetromino = Tetromino(4, 0)
        self.next_tetromino = Tetromino(4, 0)

    def _new_tetromino(self) -> None:
        """Set the next tetromino as the current one and randomly choose the next."""
        self.curr_tetromino = self.next_tetromino
        self.next_tetromino = Tetromino(4, 0)

    def _get_fall_interval(self) -> int:
        """Return the current fall interval in milliseconds."""
        base_speed = 1000
        speedup_per_level = 50
        max_speed = 100

        return max(max_speed, base_speed - (self.level - 1) * speedup_per_level)

    def _auto_drop(self) -> None:
        """Automatically drop the current tetromino if enough time has passed."""
        curr_time = pygame.time.get_ticks()
        if (
            curr_time - self.last_fall_time > self._get_fall_interval()
            and not self.game_over
        ):
            self._soft_drop()
            self.last_fall_time = curr_time

    def _freeze(self) -> None:
        """Place the current tetromino on the board and spawn a new one."""
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.curr_tetromino.shape():
                    self.board[i + self.curr_tetromino.y][
                        j + self.curr_tetromino.x
                    ] = self.curr_tetromino.type
        self._remove_line()
        self._new_tetromino()
        if self._intersects():
            self.game_over = True

    def _remove_line(self) -> None:
        """Check for and remove full lines, updating score and level."""
        lines_cleared = 0
        for y in range(self.rows):
            if all(cell is not None for cell in self.board[y]):
                del self.board[y]
                self.board.insert(0, [None for _ in range(self.cols)])
                lines_cleared += 1

        if lines_cleared > 0:
            self.score += SCORING.get(lines_cleared, 0) * self.level
            self.total_lines_cleared += lines_cleared
            if self.total_lines_cleared // 10 > self.level - 1:
                self.level += 1

    def _move_sideways(self, dx: int) -> None:
        self.curr_tetromino.x += dx
        if self._intersects():
            self.curr_tetromino.x -= dx

    def _rotate(self) -> None:
        rotation = self.curr_tetromino.rotation
        self.curr_tetromino.rotate()
        if self._intersects():
            self.curr_tetromino.rotation = rotation

    def _soft_drop(self) -> None:
        self.curr_tetromino.y += 1
        if self._intersects():
            self.curr_tetromino.y -= 1
            self._freeze()

    def _hard_drop(self) -> None:
        while not self._intersects():
            self.curr_tetromino.y += 1
        self.curr_tetromino.y -= 1
        self._freeze()

    def _intersects(self, tetromino: Tetromino | None = None) -> bool:
        """
        Check if a given tetromino intersects with the boundaries of the
        game board or other placed pieces.

        Parameters
        ----------
        tetromino : Tetromino | None
            Tetromino that should be checked for intersections.
        """
        piece = tetromino if tetromino is not None else self.curr_tetromino
        for i in range(4):
            for j in range(4):
                if i * 4 + j in piece.shape():
                    if (
                        i + piece.y > self.rows - 1
                        or j + piece.x > self.cols - 1
                        or j + piece.x < 0
                        or self.board[i + piece.y][j + piece.x] is not None
                    ):
                        return True
        return False

    def _update_display(self) -> None:
        """Redraw the game screen."""
        self.display.fill(BLACK)

        curr_width, curr_height = self.display.get_size()
        grid_width = self.cols * CELL_SIZE
        grid_height = self.rows * CELL_SIZE

        left = (curr_width - grid_width) // 2
        top = (curr_height - grid_height) // 2
        right = left + grid_width
        bottom = top + grid_height

        self._draw_grid(left, top, right, bottom)
        self._draw_board(left, top)

        if self.curr_tetromino:
            self._draw_tetromino(left, top)
            self._draw_ghost(left, top)

        if self.next_tetromino:
            self._draw_next_tetromino(left, top, right, grid_height)

        self._draw_score(left, top)

        if self.game_over:
            self._draw_game_over(left, top, grid_width, grid_height)

        pygame.display.flip()

    def _draw_grid(self, left: int, top: int, right: int, bottom: int) -> None:
        """
        Draw the Tetris grid lines on the main display.

        Parameters
        ----------
        left : int
            The x-coordinate of the left edge of the game grid.
        top : int
            The y-coordinate of the top edge of the game grid.
        right : int
            The x-coordinate of the right edge of the game grid.
        bottom : int
            The y-coordinate of the bottom edge of the game grid.
        """
        for i in range(self.rows + 1):
            y = top + i * CELL_SIZE
            pygame.draw.line(self.display, WHITE, (left, y), (right, y))
        for j in range(self.cols + 1):
            x = left + j * CELL_SIZE
            pygame.draw.line(self.display, WHITE, (x, top), (x, bottom))

    def _draw_board(self, left: int, top: int) -> None:
        """
        Draw all already placed tetrominoes on the game board.

        Parameters
        ----------
        left : int
            The x-coordinate of the left edge of the game grid.
        top : int
            The y-coordinate of the top edge of the game grid.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.board[i][j]
                if cell is not None:
                    col = Tetromino.colors[cell]
                    x = left + j * CELL_SIZE
                    y = top + i * CELL_SIZE

                    pygame.draw.rect(
                        self.display, col, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                    )
                    pygame.draw.rect(
                        self.display,
                        WHITE,
                        pygame.Rect(x, y, CELL_SIZE, CELL_SIZE),
                        1,
                    )

    def _draw_tetromino(self, left: int, top: int) -> None:
        """
        Draw the current tetromino at its position on the board.

        Parameters
        ----------
        left : int
            The x-coordinate of the left edge of the game grid.
        top : int
            The y-coordinate of the top edge of the game grid.
        """
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.curr_tetromino.shape():
                    col = Tetromino.colors[self.curr_tetromino.type]
                    x = left + (self.curr_tetromino.x + j) * CELL_SIZE
                    y = top + (self.curr_tetromino.y + i) * CELL_SIZE

                    pygame.draw.rect(
                        self.display, col, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                    )
                    pygame.draw.rect(
                        self.display,
                        WHITE,
                        pygame.Rect(x, y, CELL_SIZE, CELL_SIZE),
                        1,
                    )

    def _draw_ghost(self, left: int, top: int) -> None:
        """
        Draw a translucent preview of where the current tetromino will land.

        Parameters
        ----------
        left : int
            The x-coordinate of the left edge of the game grid.
        top : int
            The y-coordinate of the top edge of the game grid.
        """
        ghost = self.curr_tetromino.clone()
        while not self._intersects(ghost):
            ghost.y += 1
        ghost.y -= 1

        ghost_surf = pygame.Surface(
            (CELL_SIZE * self.cols, CELL_SIZE * self.rows), pygame.SRCALPHA
        )
        for i in range(4):
            for j in range(4):
                if i * 4 + j in ghost.shape():
                    col = (*Tetromino.colors[ghost.type], 96)
                    x = (ghost.x + j) * CELL_SIZE
                    y = (ghost.y + i) * CELL_SIZE

                    pygame.draw.rect(
                        ghost_surf, col, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                    )
                    pygame.draw.rect(
                        ghost_surf,
                        WHITE,
                        pygame.Rect(x, y, CELL_SIZE, CELL_SIZE),
                        1,
                    )

        self.display.blit(ghost_surf, (left, top))

    def _draw_next_tetromino(
        self, left: int, top: int, right: int, grid_height: int
    ) -> None:
        """
        Draw the next tetromino in the preview area next to the main grid.

        Parameters
        ----------
        left : int
            The x-coordinate of the left edge of the game grid.
        top : int
            The y-coordinate of the top edge of the game grid.
        right : int
            The x-coordinate of the right edge of the game grid.
        grid_heigt: int
            The height of the main game grid.
        """
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.next_tetromino.shape():
                    col = Tetromino.colors[self.next_tetromino.type]
                    x = right + 0.1 * left + j * CELL_SIZE
                    y = top + 0.4 * grid_height + i * CELL_SIZE

                    pygame.draw.rect(
                        self.display, col, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                    )
                    pygame.draw.rect(
                        self.display,
                        WHITE,
                        pygame.Rect(x, y, CELL_SIZE, CELL_SIZE),
                        1,
                    )

    def _draw_score(self, left: int, top: int) -> None:
        """
        Draw the current level and score above the game grid.

        Parameters
        ----------
        left : int
            The x-coordinate of the left edge of the game grid.
        top : int
            The y-coordinate of the top edge of the game grid.
        """
        text = self.font.render(
            f"Level: {self.level}  ---  Score: {self.score}", True, WHITE
        )
        self.display.blit(text, [left, top - 1.25 * CELL_SIZE])

    def _draw_game_over(
        self, left: int, top: int, grid_width: int, grid_height: int
    ) -> None:
        """
        Draw the "Game Over" overlay message on the game display.

        Parameters
        ----------
        left : int
            The x-coordinate of the left edge of the game grid.
        top : int
            The y-coordinate of the top edge of the game grid.
        grid_width : int
            The width of the main game grid.
        grid_heigt: int
            The height of the main game grid.
        """
        rect = pygame.Rect(
            (
                left // 2,
                top + grid_height * 0.25,
                grid_width + left,
                grid_height * 0.5,
            )
        )
        pygame.draw.rect(self.display, BLACK, rect)
        pygame.draw.rect(self.display, RED, rect, 2)

        over = self.font.render("Game Over", True, WHITE)
        msg1 = self.font.render("Press r to restart", True, RED)
        msg2 = self.font.render("Press q to quit", True, RED)

        self.display.blit(over, (rect.centerx - over.get_width() / 2, rect.y + 20))
        self.display.blit(msg1, (rect.centerx - msg1.get_width() / 2, rect.y + 80))
        self.display.blit(msg2, (rect.centerx - msg2.get_width() / 2, rect.y + 110))


def main():
    pygame.init()

    game = Tetris()
    while game.is_running():
        game.play_step()

    pygame.quit()


if __name__ == "__main__":
    main()

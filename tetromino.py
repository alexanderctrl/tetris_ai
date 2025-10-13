from __future__ import annotations

import random
from typing import ClassVar


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

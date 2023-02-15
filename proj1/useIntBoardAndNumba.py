# try numba
import numpy as np
from numba import njit
import random
import math


COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
infinity = math.inf
direction = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
random.seed(0)


# @njit(cache=True)
def get_candidate_list(color, pieces_b, pieces_w):
    my = pieces_b
    op = pieces_w
    if color == COLOR_WHITE:
        my = pieces_w
        op = pieces_b

    candidate_list = []
    for i in range(8):
        for j in range(8):
            if (int(my) & int(1 << (i * 8 + j))) ^ (int(op) & int(1 << (i * 8 + j))) == 0:  # one available position
                for dx, dy in direction:
                    x = i + dx
                    y = j + dy
                    flag = False
                    while 0 <= x <= 7 and 0 <= y <= 7 and (int(op) & int(1 << (x * 8 + y))) != 0:
                        x += dx
                        y += dy
                        flag = True
                    if flag and 0 <= x <= 7 and 0 <= y <= 7 and (int(my) & int(1 << (x * 8 + y))) != 0:
                        candidate_list.append((i, j))
    return candidate_list

# @njit(cache=True)
def move_one_step(move, color, pieces_b, pieces_w):
    my = pieces_b
    op = pieces_w
    rev = False
    if color == COLOR_WHITE:
        my = pieces_w
        op = pieces_b
        rev = True
    my |= (1 << (move[0] * 8 + move[1]))
    chain = []
    for dx, dy in direction:
        x = move[0] + dx
        y = move[1] + dy
        subchain = []
        while 0 <= x <= 7 and 0 <= y <= 7 and (op & (1 << (x * 8 + y))) != 0:
            x += dx
            y += dy
            subchain.append((x, y))
        if subchain and 0 <= x <= 7 and 0 <= y <= 7 and (my & (1 << (x * 8 + y))) != 0:
            chain.extend(subchain)
    for c in chain:
        my |= (1 << (c[0] * 8 + c[1]))
        op ^= (1 << (c[0] * 8 + c[1]))
    if rev:
        return op, my
    else:
        return my, op

@njit(cache=True)
# update the weighted_grid according to the current state
def update_weighted_grid(color, pieces_b, pieces_w):
    return np.array([[1, 8, 3, 7, 7, 3, 8, 1],
                     [8, 3, 2, 5, 5, 2, 3, 8],
                     [3, 2, 6, 6, 6, 6, 2, 3],
                     [7, 5, 6, 4, 4, 6, 5, 7],
                     [7, 5, 6, 4, 4, 6, 5, 7],
                     [3, 2, 6, 6, 6, 6, 2, 3],
                     [8, 3, 2, 5, 5, 2, 3, 8],
                     [1, 8, 3, 7, 7, 3, 8, 1]])

@njit(cache=True)
def evaluation_func(color, pieces_b, pieces_w, round, mobility):  # unbounded piece  number of reverse-able piece
    val = 0
    my = pieces_b
    op = pieces_w
    if color == COLOR_WHITE:
        my = pieces_w
        op = pieces_b
    weighted_grid = update_weighted_grid(color, pieces_b, pieces_w)
    piece_num = 0
    for i in range(8):
        for j in range(8):
            if (int(my) & int(1 << (i * 8 + j))) != 0:
                piece_num += 1
                val += weighted_grid[i][j]
    if round < 15:
        return -piece_num + val
    elif round < 40:
        return -piece_num + mobility * 2 + val
    else:
        return val - piece_num

# @njit(cache=True)
# return v and a tuple(x,y) denoting the best move
def h_alphabeta_search(color, op_color, round, pieces_b, pieces_w, cutoff_depth):
    # @njit(cache=True)
    def max_value(pieces_b, pieces_w, alpha, beta, depth):
        moves = get_candidate_list(color, pieces_b, pieces_w)
        if not moves:
            return evaluation_func(color, pieces_b, pieces_w, depth + round, 0), None
        elif depth > cutoff_depth:
            return evaluation_func(color, pieces_b, pieces_w, depth + round, len(moves)), None

        v, move = -infinity, None
        for m in moves:
            new_b, new_w = move_one_step(m, color, pieces_b, pieces_w)
            v2, _ = min_value(new_b, new_w, alpha, beta, depth + 1)
            if v2 > v:
                v, move = v2, m
            if v2 >= beta:
                return v, move
            if v > alpha:
                alpha = v
        return v, move

    # @njit(cache=True)
    def min_value(pieces_b, pieces_w, alpha, beta, depth):
        moves = get_candidate_list(op_color, pieces_b, pieces_w)
        if not moves:
            return evaluation_func(op_color, pieces_b, pieces_w, depth + round, 0), None
        elif depth > cutoff_depth:
            return evaluation_func(op_color, pieces_b, pieces_w, depth + round, len(moves)), None

        v, move = +infinity, None
        for m in moves:
            new_b, new_w = move_one_step(m, op_color, pieces_b, pieces_w)
            v2, _ = max_value(new_b, new_w, alpha, beta, depth + 1)
            if v2 < v:
                v, move = v2, m
            if v2 <= alpha:
                return v, move
            if v < beta:
                beta = v
        return v, move

    return max_value(pieces_b, pieces_w, -infinity, +infinity, 0)

class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.color = color
        self.op_color = -color

        self.round = 0  # the current game round   round+depth:the real round in alphaBetaSearch

        self.chessboard_size = chessboard_size
        self.time_out = time_out
        self.candidate_list = []

    def go(self, chessboard):
        self.round = self.round + 1
        self.candidate_list.clear()

        # int64
        pieces_b = 0  # if [i][j]has a black piece: pieces_b&(1<<((i*8)+j)) == 1  0:empty
        pieces_w = 0  # if [i][j]has a white piece: pieces_w&(1<<((i*8)+j)) == 1  0:empty
        # if [i][j] is empty: (pieces_b&(1<<((i*8)+j)))^(pieces_w&(1<<((i*8)+j))) == 0
        for i in range(7,-1,-1):
            for j in range(7,-1,-1):
                if chessboard[i][j] == COLOR_BLACK:
                    pieces_b |= 1
                elif chessboard[i][j] == COLOR_WHITE:
                    pieces_w |= 1
                pieces_b <<= 1
                pieces_w <<= 1
        pieces_b >>= 1
        pieces_w >>= 1

        # get all pos in candidate_list
        self.candidate_list = get_candidate_list(self.color, pieces_b, pieces_w)
        # call search to get the best one and append it
        _, move = h_alphabeta_search(self.color,self.op_color,self.round, pieces_b, pieces_w, 3)
        if move is not None:
            self.candidate_list.append(move)

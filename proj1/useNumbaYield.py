from typing import Tuple, List

import numpy as np
# from numba import njit
import random
# import time
import math

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
infinity = math.inf
direction = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
random.seed(0)

# @njit(Cache=True)
def get_candidate_list(color, chessboard:np.ndarray)->Tuple:
    op_color = -color
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == COLOR_NONE:  # one available position
                for dx, dy in direction:
                    x = i + dx
                    y = j + dy
                    flag = False
                    while 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == op_color:
                        x += dx
                        y += dy
                        flag = True
                    if flag and 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == color:
                        yield i, j

# @njit(cache=True)
def move_one_step(move, color, newboard:np.ndarray):
    op_color = -color
    newboard[move[0]][move[1]] = color
    chain = []
    for dx, dy in direction:
        x = move[0] + dx
        y = move[1] + dy
        subchain = []
        while 0 <= x <= 7 and 0 <= y <= 7 and newboard[x][y] == op_color:
            x += dx
            y += dy
            subchain.append((x, y))
        if subchain and 0 <= x <= 7 and 0 <= y <= 7 and newboard[x][y] == color:
            chain.extend(subchain)
    for c in chain:
        newboard[c[0]][c[1]] = color

# @njit(cache=True)
def get_stability(color, chessboard:np.ndarray):
    my_stability = 0
    op_stability = 0
    for corner in ((0, 0, 1, 1), (0, 7, 1, -1), (7, 7, -1, -1), (7, 0, -1, 1)):
        if chessboard[corner[0]][corner[1]] == 0:
            continue
        elif chessboard[corner[0]][corner[1]] == color:
            my_stability += 1
            for i in range(1,8 * corner[2]):
                if chessboard[i][corner[1]] == color:
                    my_stability += 1
                else:
                    break
            for j in range(1,8 * corner[3]):
                if chessboard[corner[0]][j] == color:
                    my_stability += 1
                else:
                    break
        else:
            op_stability += 1
            for i in range(1, 8 * corner[2]):
                if chessboard[i][corner[1]] == -color:
                    op_stability += 1
                else:
                    break
            for j in range(1, 8 * corner[3]):
                if chessboard[corner[0]][j] == -color:
                    op_stability += 1
                else:
                    break
    return my_stability,op_stability


# @njit(cache=True)
# the current evaluation according to this.color chessboard state
def evaluation_func(color, chessboard: np.ndarray, round, end):
    mobility_w = np.array([[1, -1, 1, -1], [2, -2, 2, -2], [1, -1, 4, -4]])
    stability_w = np.array([[-4, 4], [-2, 2], [-1, 1]])
    if end:
        return infinity if np.sum(chessboard) * (-color) > 0 else -infinity

    # position ################################################################
    position_weight = np.array([[1, 8, 3, 7, 7, 3, 8, 1],
                                [8, 3, 2, 5, 5, 2, 3, 8],
                                [3, 2, 6, 6, 6, 6, 2, 3],
                                [7, 5, 6, 4, 4, 6, 5, 7],
                                [7, 5, 6, 4, 4, 6, 5, 7],
                                [3, 2, 6, 6, 6, 6, 2, 3],
                                [8, 3, 2, 5, 5, 2, 3, 8],
                                [1, 8, 3, 7, 7, 3, 8, 1]])

    # mobility ################################################################
    my_candidate_len = 0
    op_candidate_len = 0
    for (x,y) in get_candidate_list(color, chessboard):
        my_candidate_len += 1
    for (x,y) in get_candidate_list(-color, chessboard):
        op_candidate_len += 1

    empty_indexes = np.where(chessboard == COLOR_NONE)
    potentials = np.zeros(chessboard.shape)
    for dx,dy in ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)):
        for point in zip(empty_indexes[0] + dx, empty_indexes[1] + dy):
            if 0 <= point[0] <= 7 and 0 <= point[1] <= 7:
                potentials[point] = 1
    my_potential = np.sum((chessboard == color) & potentials)
    op_potential = np.sum((chessboard == -color) & potentials)

    # stability################################################################
    my_stability,op_stability = get_stability(color,chessboard)

    if round < 15:
        position = np.sum(np.multiply(chessboard, position_weight)) * (-color)
        mobility = mobility_w[0][0] * my_candidate_len + mobility_w[0][1] * op_candidate_len + mobility_w[0][2] * my_potential + \
                   mobility_w[0][3] * op_potential
        stability = stability_w[0][0] * my_stability + stability_w[0][1] * op_stability
    elif round < 40:
        position = np.sum(np.multiply(chessboard, position_weight)) * (-color)
        mobility = mobility_w[1][0] * my_candidate_len + mobility_w[1][1] * op_candidate_len + mobility_w[1][2] * my_potential + \
                   mobility_w[1][3] * op_potential
        stability = stability_w[1][0] * my_stability + stability_w[1][1] * op_stability
    else:
        position = np.sum(np.multiply(chessboard, position_weight)) * (-color)
        mobility = mobility_w[2][0] * my_candidate_len + mobility_w[2][1] * op_candidate_len + mobility_w[2][2] * my_potential + \
                   mobility_w[2][3] * op_potential
        stability = stability_w[2][0] * my_stability + stability_w[2][1] * op_stability
    return position+mobility+stability



# return v and a tuple(x,y) denoting the best move
def h_alphabeta_search(chessboard, cutoff_depth, color, round):
    def max_value(board, alpha, beta, depth):
        if depth > cutoff_depth:
            return evaluation_func(color, board, depth + round, False), None
        v, move = -infinity, None
        try:
            for (x,y) in get_candidate_list(color, board):
                newboard = board.copy()
                move_one_step((x,y), color, newboard)
                v2, _ = min_value(newboard, alpha, beta, depth + 1)
                if v2 > v:
                    v, move = v2, (x,y)
                if v2 >= beta:
                    return v, move
                if v > alpha:
                    alpha = v
        except Exception:
            print('?')
            return evaluation_func(color, board, depth + round, True), None

        return v, move

    def min_value(board, alpha, beta, depth):
        if depth > cutoff_depth:
            return evaluation_func(-color, board, depth + round, False), None
        v, move = +infinity, None
        try:
            for (x,y) in get_candidate_list(-color, board):
                newboard = board.copy()
                move_one_step((x,y), -color, newboard)
                v2, _ = max_value(newboard, alpha, beta, depth + 1)
                if v2 < v:
                    v, move = v2, (x,y)
                if v2 <= alpha:
                    return v, move
                if v < beta:
                    beta = v
        except Exception:
            print('?')
            return evaluation_func(-color, board, depth + round, True), None

        return v, move

    return max_value(chessboard, -infinity, +infinity, 0)


class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.color = color
        self.round = 0  # the current game round   round+depth:the real round in alphaBetaSearch
        self.candidate_list = []

        self.chessboard_size = chessboard_size
        self.time_out = time_out

    def go(self, chessboard):
        self.round = self.round + 1
        self.candidate_list.clear()
        # get all pos in candidate_list
        try:
            for (x, y) in get_candidate_list(self.color, chessboard):
                self.candidate_list.append((x, y))
        except Exception:
            return
        # call search to get the best one and append it
        _, move = h_alphabeta_search(chessboard, 3, self.color, self.round)
        if move is not None:
            self.candidate_list.append(move)

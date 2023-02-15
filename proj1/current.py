import time
import numpy as np
from numba import njit
import math

position_weight = np.array([[1, 8, 3, 7, 7, 3, 8, 1],
                                    [8, 3, 2, 5, 5, 2, 3, 8],
                                    [3, 2, 6, 6, 6, 6, 2, 3],
                                    [7, 5, 6, 4, 4, 6, 5, 7],
                                    [7, 5, 6, 4, 4, 6, 5, 7],
                                    [3, 2, 6, 6, 6, 6, 2, 3],
                                    [8, 3, 2, 5, 5, 2, 3, 8],
                                    [1, 8, 3, 7, 7, 3, 8, 1]])
@njit(cache=True)
def get_candidate_list(color, chessboard: np.ndarray):
    op_color = -color
    candidate_list =  set()
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == 0:  # one available position
                for dx, dy in ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)):
                    x = i + dx
                    y = j + dy
                    flag = False
                    while 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == op_color:
                        x += dx
                        y += dy
                        flag = True
                    if flag and 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == color:
                        candidate_list.add((i, j))
    return list(candidate_list)


@njit(cache=True)
def move_one_step(move, color, newboard: np.ndarray):
    op_color = -color
    newboard[move] = color
    for dx, dy in ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)):
        x = move[0] + dx
        y = move[1] + dy
        chain = []
        while 0 <= x <= 7 and 0 <= y <= 7 and newboard[x][y] == op_color:
            chain.append((x,y))
            x += dx
            y += dy
        if chain and 0 <= x <= 7 and 0 <= y <= 7 and newboard[x][y] == color:
            for c in chain:
                newboard[c] = color


#
@njit(cache=True)
def get_stability(color, chessboard: np.ndarray):
    stability = [0, 0, 0]
    for corner in ((0, 0, 1, 1), (0, 7, 1, -1), (7, 7, -1, -1), (7, 0, -1, 1)):
        p = chessboard[corner[0],corner[1]]
        if p == 0:
            continue
        else:
            i = corner[0]
            j = corner[1]
            while 0 <= i <= 7:
                if chessboard[i][j] == p:
                    stability[p] += 1
                else:
                    break
                i += corner[2]

            i = corner[0]
            while 0 <= j <= 7:
                if chessboard[i][j] == p:
                    stability[p] += 1
                else:
                    break
                j += corner[3]
            stability[p] -= 1
    return stability[color],stability[-color]


class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out,mobility_w,stability_w):
        self.color = color
        if color:
            self.round = 0  # the current game round   round+depth:the real round in alphaBetaSearch
        else:
            self.round = 1
        self.candidate_list = []
        self.mobility_w = mobility_w
        self.stability_w = stability_w
        self.chessboard_size = chessboard_size
        self.time_out = time_out

    def go(self, chessboard):
        start = time.time()
        self.candidate_list.clear()
        # get all pos in candidate_list
        self.candidate_list = get_candidate_list(self.color, chessboard)
        # call search to get the best one and append it
        _, move = self.h_alphabeta_search(chessboard, 2, self.color, self.round)
        if move is not None:
            self.candidate_list.append(move)
        self.round += 2
        # print('time: ',time.time() - start)

    def evaluation_func(self,color, chessboard: np.ndarray, round, my_candidate_len, op_candidate_len):  # , op_candidate_len
        # mobility_w = np.array([[0.5, -0.5, 0.5, -0.5], [0.7, -0.7, 0.7, -0.7], [0.5, -0.5, 0.5, -0.5]])
        # stability_w = np.array([[-2.0, 2.0], [-1.0, 1.0], [-0.5, 0.5]])

        if not my_candidate_len and not op_candidate_len:
            return 1e5 if np.sum(chessboard) * (-color) > 0 else -1e5

        empty_indexes = np.where(chessboard == 0)
        potentials = np.zeros(chessboard.shape)
        for dx, dy in ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)):
            for point in zip(empty_indexes[0] + dx, empty_indexes[1] + dy):
                if 0 <= point[0] <= 7 and 0 <= point[1] <= 7:
                    potentials[point] = 1
        my_potential = np.sum(np.multiply((chessboard == color), potentials))
        op_potential = np.sum(np.multiply((chessboard == -color), potentials))

        # stability################################################################
        my_stability, op_stability = get_stability(color, chessboard)
        position = np.sum(np.multiply(chessboard, position_weight)) * color

        if round < 18:
            position /=2
            mobility = self.mobility_w[0][0] * my_candidate_len + self.mobility_w[0][1] * op_candidate_len + self.mobility_w[0][
                2] * my_potential + \
                       self.mobility_w[0][3] * op_potential
            stability = self.stability_w[0][0] * my_stability + self.stability_w[0][1] * op_stability
        elif round < 40:
            mobility = self.mobility_w[1][0] * my_candidate_len + self.mobility_w[1][1] * op_candidate_len + self.mobility_w[1][
                2] * my_potential + \
                       self.mobility_w[1][3] * op_potential
            stability = self.stability_w[1][0] * my_stability + self.stability_w[1][1] * op_stability
        else:
            mobility = self.mobility_w[2][0] * my_candidate_len + self.mobility_w[2][1] * op_candidate_len + self.mobility_w[2][
                2] * my_potential + \
                       self.mobility_w[2][3] * op_potential
            stability = self.stability_w[2][0] * my_stability + self.stability_w[2][1] * op_stability
        return position + stability + mobility

    def h_alphabeta_search(self,chessboard, cutoff_depth, color, round):
        def max_value(board, alpha, beta, depth):
            moves = get_candidate_list(color, board)
            op_moves = get_candidate_list(-color, board)
            if depth > cutoff_depth or not moves:
                return self.evaluation_func(color, board, depth + round, len(moves), len(op_moves)), None  #

            v, move = -math.inf, None
            for m in moves:
                newboard = board.copy()
                move_one_step(m, color, newboard)
                v2, _ = min_value(newboard, alpha, beta, depth + 1)
                if v2 > v:
                    v, move = v2, m
                if v >= beta:
                    return v, move
                if v > alpha:
                    alpha = v
            return v, move

        def min_value(board, alpha, beta, depth):
            moves = get_candidate_list(color, board)
            op_moves = get_candidate_list(-color, board)  # this.move
            if depth > cutoff_depth or not op_moves:
                return self.evaluation_func(color, board, depth + round, len(moves), len(op_moves)), None  #

            v, move = math.inf, None
            for m in op_moves:
                newboard = board.copy()
                move_one_step(m, -color, newboard)
                v2, _ = max_value(newboard, alpha, beta, depth + 1)
                if v2 < v:
                    v, move = v2, m
                if v <= alpha:
                    return v, move
                if v < beta:
                    beta = v
            return v, move

        return max_value(chessboard, -math.inf, math.inf, 0)

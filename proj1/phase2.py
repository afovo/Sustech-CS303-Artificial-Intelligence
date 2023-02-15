import numpy as np
import random
import time
import math

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
infinity = math.inf
direction = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
random.seed(0)


# don't change the class name


class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.color = color
        self.op_color = COLOR_BLACK
        if color == COLOR_BLACK:
            self.op_color = COLOR_WHITE

        self.weighted_grid = np.array([[1, 8, 3, 7, 7, 3, 8, 1],
                                       [8, 3, 2, 5, 5, 2, 3, 8],
                                       [3, 2, 6, 6, 6, 6, 2, 3],
                                       [7, 5, 6, 4, 4, 6, 5, 7],
                                       [7, 5, 6, 4, 4, 6, 5, 7],
                                       [3, 2, 6, 6, 6, 6, 2, 3],
                                       [8, 3, 2, 5, 5, 2, 3, 8],
                                       [1, 8, 3, 7, 7, 3, 8, 1]])
        self.mobility = 0 # len(candidate_list)
        self.round = 0  # the current game round   round+depth:the real round in alphaBetaSearch
        self.pieceNumber = 0  # pieceNumber+#of pieces in max: real piece number
        # self.unboundedPiece = 0


        self.chessboard_size = chessboard_size
        self.time_out = time_out
        self.candidate_list = []

    def get_candidate_list(self, chessboard):
        candidate_list = []
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] == COLOR_NONE:  # one available position
                    for dx, dy in direction:
                        x = i + dx
                        y = j + dy
                        flag = False
                        while 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == self.op_color:
                            x += dx
                            y += dy
                            flag = True
                        if flag and 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == self.color:
                            candidate_list.append((i, j))
        return candidate_list

    def move_one_step(self, move, newboard):
        newboard[move[0]][move[1]] = self.color
        chain = []
        for dx, dy in direction:
            x = move[0] + dx
            y = move[1] + dy
            subchain = []
            while 0 <= x <= 7 and 0 <= y <= 7 and newboard[x][y] == self.op_color:
                x += dx
                y += dy
                subchain.append((x, y))
            if subchain and 0 <= x <= 7 and 0 <= y <= 7 and newboard[x][y] == self.color:
                chain.extend(subchain)
        for c in chain:
            newboard[c[0]][c[1]] = self.color

    # update the weighted_grid according to the current state
    def update_weighted_grid(self, chessboard):
        return None

    # the current evaluation according to chessboard
    def evaluation_func(self, chessboard):
        val = 0
        self.update_weighted_grid(chessboard)
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] == self.color:
                    val += self.weighted_grid[i][j]
        return val

    # return v and a tuple(x,y) denoting the best move
    def h_alphabeta_search(self, chessboard, cutoff_depth):
        def max_value(board, alpha, beta, depth):
            if depth > cutoff_depth:
                return self.evaluation_func(board), None

            moves = self.get_candidate_list(chessboard)
            if not moves:
                return self.evaluation_func(board), None

            newboard = board.copy()
            v, move = -infinity, None
            for m in moves:
                self.move_one_step(m, newboard)
                v2, _ = min_value(newboard, alpha, beta, depth + 1)
                if v2 > v:
                    v, move = v2, m
                if v2 <= beta:
                    return v, move
                if v > alpha:
                    alpha = v
            return v, move

        def min_value(board, alpha, beta, depth):
            if depth > cutoff_depth:
                return self.evaluation_func(board), None

            moves = self.get_candidate_list(chessboard)
            if not moves:
                return self.evaluation_func(board), None

            newboard = board.copy()
            v, move = +infinity, None
            for m in moves:
                self.move_one_step(m, newboard)
                v2, _ = max_value(newboard, alpha, beta, depth + 1)
                if v2 < v:
                    v, move = v2, m
                if v2 >= alpha:
                    return v, move
                if v < beta:
                    beta = v
            return v, move

        return max_value(chessboard, -infinity, +infinity, 0)

    def go(self, chessboard):
        self.round = self.round + 1
        self.candidate_list.clear()
        # get all pos in candidate_list
        self.candidate_list = self.get_candidate_list(chessboard)
        # call search to get the best one and append it
        move = self.h_alphabeta_search(chessboard, 20)[1]
        if move is not None:
            self.candidate_list.append(move)

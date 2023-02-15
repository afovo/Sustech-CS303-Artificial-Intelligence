import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)


# don't change the class name
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as
        # your decision.
        self.candidate_list = []

        # The input is the current chessboard. Chessboard is a numpy array.

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # start = time.time()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        # empty = np.where(chessboard == COLOR_NONE)
        # empty = list(zip(empty[0], empty[1]))
        # ==============Find new pos========================================
        direction = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        op_color = COLOR_BLACK
        if self.color == COLOR_BLACK:
            op_color = COLOR_WHITE

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
                        if flag and 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == self.color:
                            # moved, still in board and reach my piece
                            self.candidate_list.append((i, j))
        # Make sure that the position of your decision on the chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chessboard
        # You need to add all the positions which are valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # candidate_list example: [(3,3),(4,4),(4,4)]
        # we will pick the last element of the candidate_list as the position you choose.
        # In above example, we will pick (4,4) as your decision.
        # If there is no valid position, you must return an empty list
        # run_time = (time.time() - start)

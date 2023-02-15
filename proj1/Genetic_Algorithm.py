import time

import numpy as np
import random

from current import AI
from current15 import AI1
from current16 import AI2
from current14 import AI1 as AI3
from numba import njit

POPULATION_SIZE = 2
ROUND = 10

@njit(cache=True)
def move_one_step(color, move, chessboard):
    op_color = -color
    chessboard[move] = color
    for dx, dy in ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)):
        x = move[0] + dx
        y = move[1] + dy
        chain = []
        flag = False
        while 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == op_color:
            chain.append((x, y))
            x += dx
            y += dy
            flag = True
        if flag and 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == color:
            for c in chain:
                chessboard[c] = color

# @njit(cache=True)
def playOnce(b: AI, w: AI):
    b.color = -1
    w.color = 1
    chessboard = np.zeros((8, 8), dtype=np.int8)
    chessboard[(3, 3)] = 1
    chessboard[(4, 4)] = 1
    chessboard[(3, 4)] = -1
    chessboard[(4, 3)] = -1
    while True:
        b.go(chessboard)
        if b.candidate_list:
            move_one_step(-1,b.candidate_list[-1],chessboard)
        w.go(chessboard)
        if w.candidate_list:
            move_one_step(1,w.candidate_list[-1],chessboard)
        if not b.candidate_list and not w.candidate_list:
            break
    if np.sum(chessboard) > 0:  # white>black
        return 5, -5
    else:
        return -5, 5

# @njit(cache=True)
def reproduce(mobility_w1, mobility_w2,stability_w1,stability_w2):
    cm = random.randint(1,3)
    mobility_w = np.hstack([mobility_w1[:,:cm], mobility_w2[:,cm:]])
    stability_w = np.hstack([stability_w2[:, :1], stability_w1[:,1:]])
    return mobility_w,stability_w


def main():
    # origin0 = AI(8, 0, 5, np.array(
    #     [[0.5, -0.5, 0.5, -0.5], [0.7, -0.7, 0.7, -0.7], [0.5, -0.5, 0.5, -0.5]]), np.array(
    #     [[-2.0, 2.0], [-1.0, 1.0], [-0.5, 0.5]]))
    # origin = AI(8,0,5,np.array(
    #     [[8.71, -10.03, 7.126001, -7.41365776],
    #      [16.10918033, -17.42505795, 10.70645469, -10.43457629],
    #      [13.7, -13.63, 20.07, -20.27]]),np.array(
    #     [[-20., 20.],
    #      [-30.58338063, 34.95252024],
    #      [-20., 20.]]
    # ))
    # byHand = AI(8, 0, 5, np.array(  ## Son2
    #     [[8.71, -10.03, 7.126001, -5.23],
    #      [18.10918033, -17.42505795, 15.70645469, -10.23],
    #      [13.7, -13.63, 20.07, -20.27]]), np.array(
    #     [[-2.33, 20.],
    #      [-30., 34.95252024],
    #      [-20., 20.]]))
    #
    # champion1 = AI(8, 0, 5, np.array(  ## Son2
    #     [[8.71, -10.03, 7.126001, -5.23],
    #      [16.10918033, -17.42505795, 10.70645469, -10.23],
    #      [13.7, -13.63, 20.07, -20.27]]), np.array(
    #     [[-2.33, 20.],
    #      [-30., 34.95252024],
    #      [-20., 20.]]))
    # champion2 = AI(8, 0, 5, np.array( ## Son2
    #     [[8.71, -10.03, 7.126001, -5.23],
    #      [16.10918033, -17.42505795, 10.70645469, -10.23],
    #      [13.7, -13.63, 20.07, -20.27]]), np.array(
    #     [[2.33, 20.],
    #      [-20., 34.95252024],
    #      [-20., 20.]]))
    #
    # son1 = AI(8, 0, 5, np.array( ## Son1
    #     [[8.71, -10.03, 7.126001, -5.23],
    #      [16.10918033, -17.42505795, 10.70645469, -10.23],
    #      [13.7, -13.63, 20.07, -20.27]]),np.array(
    #     [[2.33, 20.],
    #      [-20., 34.95252024],
    #      [-20., 20.]]))
    # son2 = AI(8, 0, 5, np.array(  ## Son1
    #     [[8.71, -10.03, 7.126001, -5.23],
    #      [16.10918033, -17.42505795, 10.70645469, -10.23],
    #      [13.7, -13.63, 20.07, -20.27]]), np.array(
    #     [[-2.33, 20.],
    #      [-30., 34.95252024],
    #      [-20., 20.]]))
    # domain1 = AI(8, 0, 5, np.array(
    #     [[8.71, -10.03, 7.126001, -7.41365776],
    #      [16.10918033, -17.42505795, 10.70645469, -10.43457629],
    #      [13.7, -13.63, 20.07, -20.27]]),
    #              np.array([[-20., 20.],
    #                        [-30.58338063, 34.95252024],
    #                        [-20., 20.]]))
    # domain0 = AI(8, 0, 5, np.array(
    #     [[0.5, -0.5, 0.5, -0.5],
    #      [1, -1, 0.7, -0.7],
    #      [1, -1, 2, -2]]),
    #             np.array([[-2, 2],
    #      [-2.5, 2.5],
    #      [-2, 2]]))
    # domain1 = AI(8, 0, 5, 2*np.array(
    #     [[0.5, -0.5, 0.5, -0.5],
    #      [1, -1, 0.7, -0.7],
    #      [1, -1, 2, -2]]),
    #              2*np.array([[-2, 2],
    #                        [-2.5, 2.5],
    #                        [-2, 2]]))
    # domain2 = AI(8, 0, 5, np.array(
    #     [[2, -2, 2, -2],
    #      [4, -4, 2.8, -2.8],
    #      [4, -4, 8, -8]]),
    #              np.array([[-8, 8],
    #                        [-10, 10],
    #                        [-8, 8]]))
    # domain3 = AI(8, 0, 5, np.array(
    #     [[5, -5, 5, -5],
    #      [10, -10, 7, -7],
    #      [10, -10, 20, -20]]),
    #              np.array([[-20, 20],
    #                        [-25, 25],
    #                        [-20, 20]]))
    # domain4 = AI(8, 0, 5, np.array(
    #     [[7, -7, 7, -7],
    #      [12, -12, 9, -9],
    #      [12, -12, 22, -22]]),
    #              np.array([[-22, 22],
    #                        [-27, 27],
    #                        [-22, 22]]))
    # domain5 = AI(8, 0, 5, 2 * np.array(  #40
    #     [[5, -5, 5, -5],
    #      [10, -10, 7, -7],
    #      [10, -10, 20, -20]]),
    #              2 * np.array([[-20, 20],
    #                                            [-25, 25],
    #                                            [-20, 20]]))
    # domain5 = AI(8, 0, 5, 2.5 * np.array(
    #     [[5, -5, 5, -5],
    #      [10, -10, 7, -7],
    #      [10, -10, 20, -20]]),
    #              2 * np.array([[-20, 20],
    #                            [-25, 25],
    #                            [-20, 20]]))
    current = AI(8, 0, 5, np.array(  # 20
        [[8.71, -10.03, 7.126001, -7.41365776],
         [16.10918033, -17.42505795, 10.70645469, -10.43457629],
         [13.7, -13.63, 20.07, -20.27]]),
                 np.array([[-20., 20.],
         [-30.58338063, 34.95252024],
         [-20., 20.]]))
    current1 = AI(8, 0, 5, np.array(  # 20
        [[19.25, -19.25, 19.25, -19.25],
         [38.5, -38.5, 26.95, -26.95],
         [38.5, -38.5, 77., -77.]]),
                 np.array([[-76., 76.],
                        [-95., 95.],
                        [-76., 76.]]))
    # domain = AI(8, 0, 5, 3 * np.array(   #20
    #     [[5, -5, 5, -5],
    #      [10, -10, 7, -7],
    #      [10, -10, 20, -20]]),
    #              3 * np.array([[-20, 20],
    #                            [-25, 25],
    #                            [-20, 20]]))
    #
    # domain1 = AI(8, 0, 5, 3.1 * np.array(  # 20
    #     [[5, -5, 5, -5],
    #      [10, -10, 7, -7],
    #      [10, -10, 20, -20]]),
    #              3 * np.array([[-20, 20],
    #                            [-25, 25],
    #                            [-20, 20]]))
    # domain2 = AI(8, 0, 5, 3.2 * np.array(  # 20
    #     [[5, -5, 5, -5],
    #      [10, -10, 7, -7],
    #      [10, -10, 20, -20]]),
    #               3 * np.array([[-20, 20],
    #                             [-25, 25],
    #                             [-20, 20]]))
    # domain3 = AI(8, 0, 5, 3.3 * np.array(  # 20
    #     [[5, -5, 5, -5],
    #      [10, -10, 7, -7],
    #      [10, -10, 20, -20]]),
    #              3 * np.array([[-20, 20],
    #                            [-25, 25],
    #                            [-20, 20]]))
    # domain4 = AI(8, 0, 5, 3.4 * np.array(  # 20
    #     [[5, -5, 5, -5],
    #      [10, -10, 7, -7],
    #      [10, -10, 20, -20]]),
    #              3.4 * np.array([[-20, 20],
    #                            [-25, 25],
    #                            [-20, 20]]))
    # domain5 = AI(8, 0, 5, 3.5 * np.array( #20
    #     [[5, -5, 5, -5],
    #      [10, -10, 7, -7],
    #      [10, -10, 20, -20]]),
    #              3.5 * np.array([[-20, 20],
    #                            [-25, 25],
    #                            [-20, 20]]))
    # domain6 = AI(8, 0, 5, 3.6 * np.array(  # 20
    #     [[5, -5, 5, -5],
    #      [10, -10, 7, -7],
    #      [10, -10, 20, -20]]),
    #              3.6 * np.array([[-20, 20],
    #                            [-25, 25],
    #                            [-20, 20]]))
    newc = AI1(8, 0, 5, np.array(  # 20
        [[8.71, -10.03, 7.126001, -7.41365776],
         [16.10918033, -17.42505795, 10.70645469, -10.43457629],
         [13.7, -13.63, 20.07, -20.27]]),
                 np.array([[-20., 20.],
         [-30.58338063, 34.95252024],
         [-20., 20.]]))

    newcc = AI2(8, 0, 5, np.array(  # 20
        [[8.71, -10.03, 7.126001, -7.41365776],
         [16.10918033, -17.42505795, 10.70645469, -10.43457629],
         [13.7, -13.63, 20.07, -20.27]]),
               np.array([[-20., 20.],
                         [-30.58338063, 34.95252024],
                         [-20., 20.]]))

    newccc = AI3(8, 0, 5, np.array(  # 20
        [[8.71, -10.03, 7.126001, -7.41365776],
         [16.10918033, -17.42505795, 10.70645469, -10.43457629],
         [13.7, -13.63, 20.07, -20.27]]),
                np.array([[-20., 20.],
                          [-30.58338063, 34.95252024],
                          [-20., 20.]]))
    domain7 = AI(8, 0, 5, 3.7 * np.array(  # 10
        [[5, -5, 5, -5],
         [10, -10, 7, -7],
         [10, -10, 20, -20]]),
                  3.7 * np.array([[-20, 20],
                                [-25, 25],
                                [-20, 20]]))
    domain75 = AI(8, 0, 5, 3.75 * np.array(  # 10
        [[5, -5, 5, -5],
         [10, -10, 7, -7],
         [10, -10, 20, -20]]),
                 3.75 * np.array([[-20, 20],
                                 [-25, 25],
                                 [-20, 20]]))
    domain8 = AI(8, 0, 5, 3.8 * np.array(  # 10
        [[5, -5, 5, -5],
         [10, -10, 7, -7],
         [10, -10, 20, -20]]),
                 3.8 * np.array([[-20, 20],
                                 [-25, 25],
                                 [-20, 20]]))
    domain85 = AI(8, 0, 5, 3.85 * np.array(  # 10
        [[5, -5, 5, -5],
         [10, -10, 7, -7],
         [10, -10, 20, -20]]),
                 3.8 * np.array([[-20, 20],
                                 [-25, 25],
                                 [-20, 20]]))
    domain9 = AI(8, 0, 5, 3.9 * np.array(  # 20
        [[5, -5, 5, -5],
         [10, -10, 7, -7],
         [10, -10, 20, -20]]),
                 3.9 * np.array([[-20, 20],
                                 [-25, 25],
                                 [-20, 20]]))
    son = AI(8, 0, 5, np.array(  # 20
        [[18.75, -18.75, 19., -19.],
         [37.5, -37.5, 26.6, -26.6],
         [37.5, -37.5, 76., -76.]]),
                 np.array([[-76.  ,  75.  ],
       [-95.  ,  93.75],
       [-76.  ,  75.  ]]))
    population = {current:0,current1:0,domain8:0,domain85:0,domain9:0,son:0,domain7:0,domain75:0,newc:0}#,domain8:0,domain85:0,domain9:0,son:0   domain0:0,domain1:0,domain2:0,domain3:0,domain4:0, ,champion2:0,champion3:0,son1:0,son2:0
    # for i in range(POPULATION_SIZE):
    #     # def __init__(self, chessboard_size, color, time_out, mobility_w, stability_w):
    #     j = 10+i*10
    #     # ai = AI(8, 0, 5, np.array(
    #     #     [[random.uniform(-j, j), random.uniform(-j, j), random.uniform(-j, j), random.uniform(-j, j)],
    #     #      [random.uniform(-j, j), random.uniform(-j, j), random.uniform(-j, j), random.uniform(-j, j)],
    #     #      [random.uniform(-j, j), random.uniform(-j, j), random.uniform(-j, j), random.uniform(-j, j)]]),
    #     #         np.array([[random.uniform(-j, j), random.uniform(-j, j)], [random.uniform(-j, j), random.uniform(-j, j)],
    #     #                   [random.uniform(-j, j), random.uniform(-j, j)]]))
    #     ai = AI(8, 0, 5, np.array(
    #         [[random.uniform(0, j), -random.uniform(0, j), random.uniform(0, j), -random.uniform(0, j)],
    #          [random.uniform(0, j), -random.uniform(0, j), random.uniform(0, j), -random.uniform(0, j)],
    #          [random.uniform(0, j), -random.uniform(0, j), random.uniform(0, j), -random.uniform(0, j)]]),
    #             np.array([[-random.uniform(0, j), random.uniform(0, j)], [-random.uniform(0, j), random.uniform(0, j)],
    #                       [-random.uniform(0, j), random.uniform(0, j)]]))
    #     population.update({ai: 0})

    # for r in range(ROUND):
    new_population = {}
    start = time.time()
    for p1 in population:
        for p2 in population:
            if p1 != p2:
                s1, s2 = playOnce(p1, p2)
                population[p1] += s1
                population[p2] += s2

    s_population = sorted(population.items(), key=lambda x: x[1], reverse=True)
    print(s_population)
    i = 0
    for ai in s_population:
        i+=1
        print(i,': ',ai[0].mobility_w,ai[0].stability_w,ai[1])
    # print('gold:', repr(s_population[0][0].mobility_w), repr(s_population[0][0].stability_w))
    # print('silver:', repr(s_population[1][0].mobility_w), repr(s_population[1][0].stability_w))
    # print('medal:', repr(s_population[1][0].mobility_w), repr(s_population[1][0].stability_w))
    # print('worst:', repr(s_population[-1][0].mobility_w), repr(s_population[-1][0].stability_w))
    m,s = reproduce(s_population[0][0].mobility_w,s_population[1][0].mobility_w,s_population[0][0].stability_w,s_population[1][0].stability_w)
    print('son:',repr(m),repr(s))
    # new_population.update({s_population[0][0]:s_population[0][1]})
    # print(new_population)
    print(time.time()-start)
    return None

if __name__ == "__main__":
    main()

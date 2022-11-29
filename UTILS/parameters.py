import os

import torch


class parameters:
    def __init__(self):
        self.m_t = 0.5
        self.t_b = 0.1
        self.t_f = 0.5
        self.t_p = 0.5


# test
if __name__ == "__main__":
    para = parameters()
    print(para.m_t)

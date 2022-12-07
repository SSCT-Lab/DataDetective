import os

import torch


class parameters:
    def __init__(self):
        self.fault_type = {
            'no fault': 0,
            'class fault': 1,
            'location fault': 2,
            'redundancy fault': 3,
            'missing fault': 4,
        }
        self.fault_ratio = 0.05
        self.m_t = 0.5
        self.t_b = 0.1
        self.t_f = 0.5
        self.t_p = 0.5


# test
if __name__ == "__main__":
    para = parameters()
    print(para.m_t)

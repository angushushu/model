from enum import Enum, unique


@unique
class Type(Enum):
    ru = 0  # representation unit
    au = 1  # action unit
    r = 2   # representation
    a = 3   # action
    c1 = 4  # connection 1
    c2 = 5  # connection 2 (w/ action)
    c3 = 6  # connection 3 (w/o action, maybe useful)
    s = 7 # state for SAG
    g = 8 # goal for SAG

    @classmethod
    def str_to_int(cls, t):
        mapping = {'ru':0, 'au':1, 'r':2, 'a': 3, 'c1': 4, 'c2': 5, 'c3': 6, 's': 7, 'g': 8}
        return mapping[t]

    @classmethod
    def int_to_str(cls, t):
        mapping = {0:'ru', 1:'au', 2:'r', 3: 'a', 4: 'c1', 5: 'c2', 6: 'c3', 7: 'g', 8: 'g'}
        return mapping[t]

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

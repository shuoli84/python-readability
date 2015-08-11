from collections import namedtuple

Range = namedtuple("Range", ["min", "max"])


class Settings(object):
    title_length = Range(min=5, max=150)

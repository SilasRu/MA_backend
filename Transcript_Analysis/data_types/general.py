from enum import Enum, auto, unique


@unique
class Output_type(Enum):
    # HTML = auto()
    SENTENCE = auto()
    WORD = auto()

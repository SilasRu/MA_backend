from operator import attrgetter
from collections import namedtuple


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))


class ItemsCount(object):
    def __init__(self, value):
        self._value = value

    def __call__(self, sequence):
        if isinstance(self._value, (bytes, str,)):
            if self._value.endswith("%"):
                total_count = len(sequence)
                percentage = int(self._value[:-1])
                # at least one sentence should be chosen
                count = max(1, total_count*percentage // 100)
                return sequence[:count]
            else:
                return sequence[:int(self._value)]
        elif isinstance(self._value, (int, float)):
            return sequence[:int(self._value)]
        else:
            ValueError("Unsuported value of items count '%s'." % self._value)

    def __repr__(self):
        return to_string("<ItemsCount: %r>" % self._value)


class BaseSummarizer(object):

    def __call__(self, document, sentences_count):
        raise NotImplementedError(
            "This method should be overriden in subclass")

    @staticmethod
    def normalize_word(word):
        return word.lower()

    @staticmethod
    def _get_best_sentences(sentences, count, rating, *args, **kwargs):
        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            def rate(s): return rating[s]

        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs))
                 for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        if not isinstance(count, ItemsCount):
            count = ItemsCount(count)
        infos = count(infos)
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))

        return tuple(i.sentence for i in infos)

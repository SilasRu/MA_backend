from fast_autocomplete import AutoComplete
from collections import Counter

from Transcript_Analysis.utils.utils import Utils


class Meeting_Autocomplete:

    def __init__(self, text: str) -> None:
        text = text
        text = Utils.remove_punct(text)

        vocab = Counter(set(text.split()))

        self.autocomplete = AutoComplete(words={
            word: {} for word in vocab.keys()
        })
        for word, count in vocab.items():
            self.autocomplete.update_count_of_word(word, count)

    def search(self, query: str, size_of_results: int = 5):
        return self.autocomplete.search(
            word=query,
            size=size_of_results
        )

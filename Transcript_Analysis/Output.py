import json


class Keyword:
    def __init__(self, start_time: int, end_time: int, text: str, speaker: str, weight: int) -> None:
        self.start_time: int = start_time
        self.end_time: int = end_time
        self.text: str = text
        self.speaker: str = speaker
        self.weight: int = weight

    def get_json_output(self) -> json:
        return json.dumps({
            'start_time': self.start_time,
            'end_time': self.end_time,
            'text': self.text,
            'speaker': self.speaker,
            'weight': self.weight
        })


def main() -> None:
    test_keyword = Keyword(
        start_time=0,
        end_time=100,
        text='this is a test',
        speaker='Zhivar',
        weight=20
    )
    print(test_keyword.get_json_output())


if __name__ == "__main__":
    main()

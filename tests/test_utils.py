import unittest

from tiny_llm.utils import read_text


class TestReadText(unittest.TestCase):
    def test_read_text(self):
        filename = "tiny_llm/input_data/mock_text.txt"
        raw_text = read_text(filename=filename)
        print(raw_text)

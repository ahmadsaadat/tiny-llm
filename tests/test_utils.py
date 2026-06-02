import unittest

from tiny_llm.utils import prepare_text, read_text


class TestReadText(unittest.TestCase):
    FILENAME = "tiny_llm/input_data/mock_text.txt"
    EXPECTED = """hello 1\nhello 2\nhello 3"""

    def test_read_text(self):
        assert read_text(filename=self.FILENAME) == self.EXPECTED


class TestPrepareText(unittest.TestCase):
    RAW_TEXT = """hello 1\nhello 2\nhello 3"""
    EXPECTED = ["hello 1 <END>", "hello 2 <END>", "hello 3 <END>"]

    def test_prepare_text(self):
        assert prepare_text(raw_text=self.RAW_TEXT) == self.EXPECTED

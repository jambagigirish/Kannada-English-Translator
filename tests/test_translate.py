from models.translator import Translator


def test_basic_translation():
tr = Translator()
out = tr.translate("ನಮಸ್ಕಾರ")
assert isinstance(out, str)
assert len(out) > 0

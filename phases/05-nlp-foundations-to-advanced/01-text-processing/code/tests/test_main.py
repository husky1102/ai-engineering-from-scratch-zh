import unittest

from main import demo_pos_tagger, lemmatize, preprocess, stem_step_1a, tokenize


class TextProcessingTest(unittest.TestCase):
    def test_tokenize_splits_words_numbers_and_punctuation(self):
        self.assertEqual(tokenize("Cats ran at 3pm."), ["Cats", "ran", "at", "3", "pm", "."])

    def test_stem_step_1a_handles_plural_suffixes(self):
        cases = {
            "classes": "class",
            "ponies": "poni",
            "glass": "glass",
            "cats": "cat",
            "s": "s",
        }
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(stem_step_1a(word), expected)

    def test_lemmatize_uses_pos_sensitive_table(self):
        self.assertEqual(lemmatize("better", "ADJ"), "good")
        self.assertEqual(lemmatize("better", "NOUN"), "better")
        self.assertEqual(lemmatize("were", "VERB"), "be")

    def test_lemmatize_falls_back_by_part_of_speech(self):
        self.assertEqual(lemmatize("walking", "VERB"), "walk")
        self.assertEqual(lemmatize("dogs", "NOUN"), "dog")
        self.assertEqual(lemmatize("LOUD", "ADJ"), "loud")

    def test_demo_pos_tagger_marks_known_verbs_and_adjectives(self):
        self.assertEqual(
            demo_pos_tagger(["running", "better", "cats"]),
            [("running", "VERB"), ("better", "ADJ"), ("cats", "NOUN")],
        )

    def test_preprocess_returns_tokens_stems_and_lemmas(self):
        result = preprocess("The cats were running.", pos_tagger=demo_pos_tagger)

        self.assertEqual(result["tokens"], ["The", "cats", "were", "running", "."])
        self.assertEqual(result["stems"], ["the", "cat", "were", "running", "."])
        self.assertEqual(result["lemmas"], ["the", "cat", "be", "run", "."])


if __name__ == "__main__":
    unittest.main()

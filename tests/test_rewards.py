import os
import sys
import unittest


PACKAGE_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(PACKAGE_SRC))

from open_r1.vlm_modules.qwen_module import QwenVLModule


def completion(text):
    return [[{"content": text}]]


class RewardTests(unittest.TestCase):
    def test_format_reward_is_binary(self):
        valid = '<answer>[{"bbox_2d":[1,2,3,4],"text_content":"A"}]</answer>'
        invalid = '<answer>[{"bbox_2d":[1,2,3,4]}]</answer>'
        self.assertEqual(QwenVLModule.format_reward_spotting(completion(valid), [[]]), [1.0])
        self.assertEqual(QwenVLModule.format_reward_spotting(completion(invalid), [[]]), [0.0])

    def test_text_reward_uses_duplicate_counts(self):
        text = '<answer>[{"bbox_2d":[1,2,3,4],"text_content":"sale"}]</answer>'
        solution = [[
            {"bbox_2d": [1, 2, 3, 4], "text_content": "sale"},
            {"bbox_2d": [5, 6, 7, 8], "text_content": "sale"},
        ]]
        reward = QwenVLModule.content_reward(completion(text), solution)[0]
        self.assertAlmostEqual(reward, 2 / 3)

    def test_detection_precision_and_recall_are_separate(self):
        text = '<answer>[{"bbox_2d":[0,0,10,10],"text_content":"A"},{"bbox_2d":[20,20,30,30],"text_content":"B"}]</answer>'
        solution = [[{"bbox_2d": [0, 0, 10, 10], "text_content": "A"}]]
        self.assertEqual(QwenVLModule.precision_reward(completion(text), solution), [0.5])
        self.assertEqual(QwenVLModule.recall_reward(completion(text), solution), [1.0])


if __name__ == "__main__":
    unittest.main()

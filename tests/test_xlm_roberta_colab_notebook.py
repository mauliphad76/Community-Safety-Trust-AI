import ast
import json
from pathlib import Path
import unittest


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "fake_news"
    / "xlm_roberta_colab.ipynb"
)


class XlmRobertaColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.code_cells = [
            "".join(cell["source"])
            for cell in cls.cells
            if cell["cell_type"] == "code"
        ]

    def test_notebook_is_valid_and_code_cells_parse(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for source in self.code_cells:
            python_source = "\n".join(
                line for line in source.splitlines() if not line.startswith("%")
            )
            ast.parse(python_source)

    def test_training_uses_validation_selection_and_early_stopping(self):
        all_code = "\n".join(self.code_cells)
        self.assertIn("FacebookAI/xlm-roberta-base", all_code)
        self.assertIn("MAX_LENGTH = 256", all_code)
        self.assertIn("learning_rate=2e-5", all_code)
        self.assertIn("optim='adamw_torch'", all_code)
        self.assertIn("metric_for_best_model='f1_fake'", all_code)
        self.assertIn("EarlyStoppingCallback(early_stopping_patience=2)", all_code)
        self.assertIn("eval_dataset=validation_dataset", all_code)

    def test_test_data_is_read_only_after_training(self):
        joined = "\n".join(self.code_cells)
        self.assertLess(joined.index("trainer.train()"), joined.index("read_split(language, 'test')"))


if __name__ == "__main__":
    unittest.main()

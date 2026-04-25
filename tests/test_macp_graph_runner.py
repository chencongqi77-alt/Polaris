import unittest
from app.graph.main_graph import MacpGraphRunner, MacpState

class TestMacpGraphRunner(unittest.TestCase):

    def test_initialization(self):
        runner = MacpGraphRunner()
        self.assertIsNotNone(runner.tm)
        self.assertIsNotNone(runner.sg)
        self.assertIsNotNone(runner.eva)
        self.assertIsNotNone(runner.graph)

    def test_build_graph(self):
        runner = MacpGraphRunner()
        self.assertIsNotNone(runner.graph)

    def test_run_flow(self):
        state = MacpState(user_request="Build a machine learning intro class")
        runner = MacpGraphRunner()
        result = runner.run(state)
        
        self.assertIsNotNone(result.selected_candidate_id)
        self.assertGreater(len(result.evaluations), 0)
        self.assertIsInstance(result.approved, bool)
        self.assertIn("trace", result.metadata)
        self.assertEqual(result.metadata["trace"][0], "TM")

if __name__ == "__main__":
    unittest.main()
import ast
import inspect
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def active_main_calls(path: Path):
    tree = ast.parse(path.read_text())
    main_blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Name) and child.id == "__name__" for child in ast.walk(node.test))
    ]
    block = main_blocks[-1]
    calls = {}
    assignments = {}
    for node in ast.walk(block):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            calls[name] = {keyword.arg: ast.unparse(keyword.value) for keyword in node.keywords}
    return assignments, calls


class SingleConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assignments, cls.calls = active_main_calls(
            REPO_ROOT / "Finetuning" / "finetune_script2.py"
        )

    def test_single_environment(self):
        self.assertEqual(self.assignments["env_name"], "cube")
        self.assertEqual(self.assignments["specific_env"], "single-play")
        self.assertEqual(self.assignments["task_id"], 4)
        self.assertEqual(self.assignments["finetune_buffer_cutoff_length"], 100)
        self.assertEqual(self.assignments["train_buffer_cutoff_length"], 200)

    def test_reward_config(self):
        expected = {
            "hidden_layers": "4", "hidden_dim": "512", "batch_size": "256",
            "num_steps": "30000", "lr": "0.005", "min_lr": "0.0005",
            "sigma": "4.0", "target_reward": "500.0", "train_goal": "None",
            "task_id": "task_id",
        }
        self.assertEqual(self.calls["Train_Reward_Config"], expected)

    def test_finetuning_config_has_no_extra_runtime_knobs(self):
        config = self.calls["FinetuningConfig"]
        self.assertNotIn("rollout_every", config)
        expected_values = {
            "offline": "True", "critic": "True", "update_critic": "True",
            "kernel": "True", "update_kernel": "False", "buffer_size": "200000",
            "finetune_steps": "90", "finetune_rounds": "30", "diffusion_steps": "10",
            "karras_percent": "0.1", "Loss_Clip_percent": "0.0",
            "finetune_batch_size": "32", "finetune_batch_per_sample": "8",
            "finetune_lr": "2e-05", "initial_lam": "0.05", "eta_lam": "0.5",
            "gradient_accumulate_every": "1", "update_lambda_every": "1",
            "reward_scaling_factor": "150", "MaxEnt": "False",
            "Entropy_Scaling_Factor": "0.5", "rollout_length": "4000",
            "rollout_num_envs": "8", "continual_rollout": "True",
            "chunk_size": "31", "num_rollout_processes": "8",
        }
        for name, expected in expected_values.items():
            self.assertEqual(config[name], expected, name)

    def test_imported_dataclass_matches_script(self):
        sys.path.insert(0, str(REPO_ROOT / "Finetuning"))
        from Finetune_Backbone3 import FinetuningConfig

        signature = inspect.signature(FinetuningConfig)
        self.assertNotIn("rollout_every", signature.parameters)


if __name__ == "__main__":
    unittest.main()

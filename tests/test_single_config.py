import ast
import unittest
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


class SingleConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = OmegaConf.load(REPO_ROOT / "Finetuning" / "conf" / "cube_single.yaml")

    def test_single_environment(self):
        env = self.config.scripts.finetune_script2
        self.assertEqual(env.dataset_name, "cube")
        self.assertEqual(env.specific_dataset, "single-play")
        self.assertEqual(env.task_id, 4)
        settings = self.config.scripts.finetune_script2.settings
        self.assertIsNone(settings.finetune_buffer_cutoff_length)
        self.assertIsNone(settings.train_buffer_cutoff_length)
        self.assertEqual(
            self.config.scripts.train_reward_script.target_reward,
            1.0,
        )

    def test_configuration_is_grouped_by_entrypoint(self):
        expected_scripts = {
            "pretrain_script4",
            "train_reward_script",
            "train_kernel_script",
            "train_critic_script",
            "train_critic_script2",
            "finetune_script2",
        }
        self.assertEqual(set(self.config.scripts.keys()), expected_scripts)
        for legacy_name in (
            "planner_pretrain",
            "reward_pretrain",
            "kernel_pretrain",
            "critic_pretrain",
            "critic_warmup",
            "critic_training",
            "finetuning",
        ):
            self.assertNotIn(legacy_name, self.config)

    def test_symlog_critic_pipeline(self):
        scripts = self.config.scripts
        self.assertEqual(scripts.train_critic_script.new_step, -1)
        self.assertEqual(scripts.train_critic_script.value_scale, 1.0)
        self.assertEqual(scripts.train_critic_script.num_steps, 2000)
        for bootstrap_parameter in ("lam", "horizon", "tau"):
            self.assertNotIn(
                bootstrap_parameter,
                scripts.train_critic_script,
            )
        self.assertEqual(scripts.train_critic_script2.old_critic_checkpoint, -1)
        self.assertEqual(scripts.train_critic_script2.kernel.oversample, 40)
        self.assertEqual(scripts.finetune_script2.kernel_model.oversample, 30)
        self.assertEqual(scripts.finetune_script2.critic_update.rho, 0.0)
        self.assertEqual(scripts.finetune_script2.critic_update.resample_every, 1)

    def test_finetuning_parameters_are_preserved(self):
        config = self.config.scripts.finetune_script2.settings
        expected = {
            "offline": True,
            "critic": True,
            "update_critic": True,
            "kernel": False,
            "update_kernel": False,
            "buffer_size": 200000,
            "finetune_steps": 90,
            "finetune_rounds": 30,
            "diffusion_steps": 10,
            "karras_percent": 0.1,
            "loss_clip_percent": 0.2,
            "finetune_batch_size": 256,
            "finetune_batch_per_sample": 4,
            "finetune_lr": 2e-5,
            "initial_lam": 0.0,
            "eta_lam": 0.05,
            "gradient_accumulate_every": 1,
            "update_lambda_every": 1,
            "reward_scaling_factor": 500,
            "max_ent": False,
            "entropy_scaling_factor": 0.5,
            "rollout_length": 4000,
            "rollout_num_envs": 8,
            "continual_rollout": True,
            "chunk_size": 31,
            "num_rollout_processes": 8,
        }
        for name, expected_value in expected.items():
            self.assertEqual(config[name], expected_value, name)

    def test_critic_dataclass_has_planner7_controls(self):
        source = (REPO_ROOT / "Finetuning" / "Finetune_Backbone3.py").read_text()
        tree = ast.parse(source)
        critic_config = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Train_Critic_Config"
        )
        parameters = {
            node.target.id for node in critic_config.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        for name in ("rho", "resample_every", "log_every"):
            self.assertIn(name, parameters)

    def test_finetune_yaml_matches_dataclass_fields(self):
        def annotated_fields(relative_path, class_name):
            tree = ast.parse((REPO_ROOT / relative_path).read_text())
            class_node = next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            return {
                node.target.id for node in class_node.body
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
            }

        stage = self.config.scripts.finetune_script2
        am_runtime = {
            "step_start_ema",
            "ema_decay",
            "save_freq",
            "save_model_freq",
            "log_freq",
        }
        checks = (
            (
                set(stage.adjoint_matching.keys()) - am_runtime,
                annotated_fields(
                    "Finetuning/acc_adjoint_matching.py",
                    "Acc_AdjointMatchingConfig",
                ),
            ),
            (
                set(stage.reward_objective.keys()),
                annotated_fields("Finetuning/traj_reward4.py", "RewardConfig"),
            ),
            (
                set(stage.reward_model.keys()),
                annotated_fields("Finetuning/Finetune_Backbone3.py", "Train_Reward_Config"),
            ),
            (
                set(stage.kernel_model.keys()) - {"lambda_reg"} | {"λ_reg"},
                annotated_fields("Finetuning/Finetune_Backbone3.py", "Train_Kernel_Config"),
            ),
            (
                set(stage.critic_update.keys()),
                annotated_fields("Finetuning/Finetune_Backbone3.py", "Train_Critic_Config"),
            ),
        )
        for supplied, accepted in checks:
            self.assertTrue(supplied <= accepted, supplied - accepted)

    def test_reward_model_keeps_state_action_inputs(self):
        for relative_path in (
            "Pretrain/Rewards/Reward_Backbone.py",
            "Finetuning/utils.py",
            "Finetuning/traj_reward4.py",
        ):
            tree = ast.parse((REPO_ROOT / relative_path).read_text())
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "reward_net"
                    or isinstance(node.func, ast.Attribute)
                    and node.func.attr == "reward_net"
                )
            ]
            self.assertTrue(calls, relative_path)
            self.assertTrue(
                all(len(call.args) == 2 for call in calls),
                relative_path,
            )

    def test_cube_trajectories_use_first_success_and_validation_split(self):
        source = (REPO_ROOT / "Pretrain" / "Dataset.py").read_text()
        self.assertIn('split: str = "train"', source)
        self.assertIn('episode_rewards == 0', source)

    def test_critic_one_is_monte_carlo_only(self):
        source = (REPO_ROOT / "Finetuning" / "utils.py").read_text()
        start = source.index("def train_critic_with_reward(")
        stop = source.index("def test_critic_with_reward(", start)
        function_source = source[start:stop]
        self.assertIn("obtain_monte_carlo_data", function_source)
        self.assertNotIn("target_critic", function_source)
        self.assertNotIn("td_lambda", function_source)


if __name__ == "__main__":
    unittest.main()

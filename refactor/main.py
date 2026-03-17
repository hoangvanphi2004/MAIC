
import argparse
import json
from pathlib import Path
import torch
from train import run_training


def load_train_config(config_path: str) -> dict:
	path = Path(config_path)
	if not path.is_absolute():
		path = Path(__file__).resolve().parent / path

	if not path.is_file():
		raise FileNotFoundError(f'Config file not found: {path}')

	with path.open('r', encoding='utf-8') as f:
		cfg = json.load(f)

	if not isinstance(cfg, dict):
		raise ValueError(f'Config file must be a JSON object: {path}')

	return cfg


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train multi-agent algorithms for MultiGrid with configurable observation/state extraction.')
	parser.add_argument('--config', type=str, default='train_config.json', help='Path to single JSON config containing all training settings.')
	args = parser.parse_args()

	train_cfg = load_train_config(args.config)

	run_training(**train_cfg)
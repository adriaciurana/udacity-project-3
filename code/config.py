import os
import torch
class Config:
	"""
		PATHS
	"""
	CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
	PROJECT_PATH = os.path.dirname(CURRENT_PATH)
	
	ENVS_PATH = os.path.join(PROJECT_PATH, "envs")
	CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "checkpoints")
	MODELS_PATH = os.path.join(PROJECT_PATH, "models")
	
	TENNIS_ENV_PATH = os.path.join(ENVS_PATH, "tennis", "Tennis.x86_64")
	CHECKPOINT_TENNIS_PATH = os.path.join(CHECKPOINT_PATH, "tennis")
	MODEL_TENNIS_PATH = os.path.join(MODELS_PATH, "tennis")

	"""
		TORCH CONFIG
	"""
	DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	"""
		 TRAINING PARAMETERS
	"""
	ENABLE_TRAIN = True
	SELECTED_ENV = 'tennis'

	BUFFER_SIZE = int(1e9)  # replay buffer size
	BATCH_SIZE = 1024       # minibatch size
	GAMMA = 1 #0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR_ACTOR = 1e-4         # learning rate of the actor 
	LR_CRITIC = 1e-3        # learning rate of the critic
	WEIGHT_DECAY = 0        # L2 weight decay
	LEARN_INTERVAL = 10     # Interval of learning, for continuous put None or 1
	LEARN_STEPS = 50     # Interval of learning, for continuous put None or 1
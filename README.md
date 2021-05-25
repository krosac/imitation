# Imitation Learning (GAIL, AIRL)
forked from [https://github.com/HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation) (based on [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html))

Rewards in the course of adversarial training are not explicitly tracked in [AdversarialTrainer](https://github.com/krosac/imitation/blob/master/imitation/src/imitation/algorithms/adversarial.py).
So I manually evaluate the reward from generator samples at [imitation/src/imitation/algorithms/adversarial.py#L251](https://github.com/krosac/imitation/blob/master/imitation/src/imitation/algorithms/adversarial.py#L251). 
**The reward evaluation needs to be further checked!**

## Run
```
cd imitation
pip install -e .
python3 examples/quickstart_airl.py --env Pendulum-v0 --expert A2C --policy MlpPolicy --expert_timestamps 3e6 --il_algo airl --il_rounds 2000 -v
```
The expert will be trained first, followed by the adversarial training with expert trajectories.
Under ``imitation/``, visualize the reward during expert training with tensorboard.
```
tensorboard --logdir <expert>_<env>_tensorboard/<run>_<id>/
```
Visualize the reward during adversarial training.
```
tensorboard --logdir output/summary/
```

## Results

##### Pendulum-v0


##### Mountaincar-v0

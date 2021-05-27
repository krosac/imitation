# Imitation Learning (reproduce GAIL, AIRL)
forked from [https://github.com/HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation) (based on [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html))

Expert training hyperparameters: [https://github.com/araffin/rl-baselines-zoo/tree/master/hyperparams](https://github.com/araffin/rl-baselines-zoo/tree/master/hyperparams)

Rewards in the course of adversarial training are not explicitly tracked in [AdversarialTrainer](https://github.com/krosac/imitation/blob/master/imitation/src/imitation/algorithms/adversarial.py).
So I manually evaluate the reward from generator samples at [imitation/src/imitation/algorithms/adversarial.py#L251](https://github.com/krosac/imitation/blob/master/imitation/src/imitation/algorithms/adversarial.py#L251). 
**The reward evaluation needs to be further checked!**

## Run
```
cd imitation
```
Update the code of python package if you customize ``imitation/src/*``. 
```
pip install -e .
```
Run the quickstart script.
```
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

## Initial Results

##### Pendulum-v0
|![](https://github.com/krosac/imitation/blob/master/images/pendulum_ppo.PNG) | ![]()|
|:--:| :--:|
|*Expert(PPO('MlpPolicy', util.make_vec_env('Pendulum-v0', n_envs=8),
                    n_steps=2048,
                    batch_size=32,
                    gae_lambda=0.95,
                    gamma=0.99,
                    n_epochs=10,
                    ent_coef=0.0,
                    learning_rate=3e-4,
                    clip_range=0.2))* |*AIRL*|

##### Mountaincar-v0
|![](https://github.com/krosac/imitation/blob/master/images/mountaincar_a2c.PNG)|![](https://github.com/krosac/imitation/blob/master/images/mountaincar_airl.PNG)|
|:--:|:--:|
|*Expert(A2C)*|*AIRL*|

# imitation
Course project reproducing GAIL, AIRL results

## AIRL on Pendulum-v0

#### Build
```
git clone https://github.com/rll/rllab.git
git clone https://github.com/justinjfu/inverse_rl.git
```
Fix a typo at [https://github.com/rll/rllab/blob/master/rllab/sampler/stateful_pool.py#L3](https://github.com/rll/rllab/blob/master/rllab/sampler/stateful_pool.py#L3) from ````from joblib.pool import MemmapingPool```` to ``from joblib.pool import MemmappingPool``.
Update ``PYTHONPATH`` and install some unmentioned denpendencies.
```
export PYTHONPATH=/path/to/rllab:$PYTHONPATH
pip install path pyprind cached-property
export PYTHONPATH=/path/to/inverse_rl:$PYTHONPATH
```
Downgrade openai gym version for compatability. 
```
pip install gym==0.14
```

#### Run
[Pendulum Example](https://github.com/justinjfu/inverse_rl#examples) runs well with tf1.15.3 and python 3.7.5.
The results (trajectories and training stats) can be found under ``data/``.
```
cd inverse_irl
python scripts/pendulum_data_collect.py
python scripts/pendulum_irl.py
```
Visualize the average return
```
python3 gen_tensorboard_event.py --stats data/pendulum_irl/progress.csv --log_dir tensorboard/pendulum_irl
```

## AIRL on CartPole-v1 and LunarLander-v2

```
cd imitation
pip install -e .
```
Run the quickstart script.
```
python3 examples/quickstart_airl.py --env LunarLanderContinuous-v2 --expert PPO --policy MlpPolicy --expert_timestamps 2e6 --il_algo airl --il_rounds 2000 -v
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


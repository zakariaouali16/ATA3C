## Atari-A3C(CPU-only)
```shell
nohup python -u main.py 2>&1 &
```
### Requirements

We use `Pytorch 1.4.0` and `Python 3.6`.

### HyperParameters

|      Name       |                   Description                    |    Value    |
| :-------------: | :----------------------------------------------: | :---------: |
|   gae_lambda    |             lambda parameter for GAE             |    1.00     |
|      gamma      |           discount factor for rewards            |    0.99     |
|       lr        |                  learning rate                   |   0.0001    |
|  entropy_coef   |             entropy term coefficient             |    0.01     |
| value_loss_coef |              value loss coefficient              |     0.5     |
|  max_grad_norm  |         clamping the values of gradient          |     40      |
|  num_processes  | how many training processes to use(asynchronous) |     32      |
|    num_steps    |          number of forward steps in A3C          |     20      |
|    no_shared    |     use an optimizer without shared momentum     |    False    |
|    env_name     |             environment to train on              | Breakout-v0 |
|      Input      |                    Input size                    | [1, 80, 80] |

### Model

In this model, a CNN is used to extract the feature of images and a Long Short-Term Memory(LSTM)  is  used  to  process  temporal  dependencies.   Thereafter,  the  two  fully-connected  layerswhich sit at the top of the network will provide the probability distribution and value functionapproximation given the current state.

Instead of the experience replay used in DQN, the ATA3C asynchronously execute multiple agentsin parallel on multiple instances of the environment.  This parallelism decorrelates the agents’ datainto a more stationary process and make the model learn online, since at any given time-step, theparallel agents will be experiencing a variety of different states.



### Result (training)

#### Test score per 10 episodes as training goes on

> Suddenly goes down may because of the action 'fire' problem. (We don't use wrappers.)












```shell
# You can use the trained model to see the simulation, or use it as your baseline.
# Test the model and see the simulation.
python utils.py
```

### Note

```shell
# Do not change below code in main.py, it will cause some bugs in Ubuntu.
mp.set_start_method("spawn")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
```


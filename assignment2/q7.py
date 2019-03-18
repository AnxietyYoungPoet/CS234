import gym
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear

from configs.q7_config import config


class MyDQN(Linear):
  def get_q_values_op(self, state, scope, reuse=False):
    num_actions = self.env.action_space.n
    with tf.variable_scope(scope, reuse=reuse):
      normed = tf.layers.batch_normalization(state, scale=True)
      conv1 = layers.conv2d(normed, 32, 8, stride=4)
      conv2 = layers.conv2d(conv1, 64, 4, stride=2)
      conv3 = layers.conv2d(conv2, 64, 3, stride=1)
      aligned = layers.flatten(conv3)
      fc1 = layers.fully_connected(aligned, 512)
      fc2 = layers.fully_connected(aligned, 512)
      adv = layers.fully_connected(fc1, num_actions, activation_fn=None)
      state_value = layers.fully_connected(fc2, 1, activation_fn=None) 
      out = state_value + adv - tf.reduce_mean(adv, axis=1, keep_dims=True)
    return out



if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = MyDQN(env, config)
    model.run(exp_schedule, lr_schedule)

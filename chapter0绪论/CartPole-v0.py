import gym
env = gym.make("CartPole-v0") #构建实验环境
env.reset() #重置一个回合
for _ in range(1000):
    env.render() #渲染当前环境
    action = env.action_space.sample() #随机选择一个动作
    observation, reward, done, info = env.step(action) #用于提交动作，括号内是具体的动作
    print(observation)
env.close() #关闭环境
import random
import numpy as np

import gym

from sklearn.metrics import mean_squared_error as me
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier
from xgboost import XGBRegressor

seed = 31
random.seed(seed)
np.random.seed(seed)

# Machine Learning Model
class Model():
    GaussianProcessRegressor = "GPR"
    GaussianNaiveBayes = "GNB"
    RandomForestClassifier = "RFC"
    KNeighborsClassifier = "KNC"
    GaussianProcessClassifier = "GPC"
    GradientBoostingClassifier = "GBC"
    XGBClassifier = "XGBC"
    XGBRegressor = "XGBR"

    def __init__(self, type , shuffle = True, undersample = True, partialFit = True, debug = False) -> None:
        self.shuffle = shuffle
        self.undersample = undersample
        self.partialFit = partialFit
        self.debug = debug
        self.type = type
        self.isRegressor = False
        
        if self.type == self.GaussianNaiveBayes:
            self.model = GaussianNB()
        elif self.type == self.GaussianProcessRegressor:
            self.model = GaussianProcessRegressor()
            self.isRegressor = True
        elif self.type == self.XGBRegressor:
            self.model = XGBRegressor()
            self.isRegressor = True
        elif self.type == self.RandomForestClassifier:
            self.model = RandomForestClassifier()
        elif self.type == self.KNeighborsClassifier:
            self.model = KNeighborsClassifier()
        elif self.type == self.GaussianProcessClassifier:
            self.model = GaussianProcessClassifier()
        elif self.type == self.GradientBoostingClassifier:
            self.model = GradientBoostingClassifier()
        elif self.type == self.XGBClassifier:
            self.model = XGBClassifier()

        self.randomUnderSampler = RandomUnderSampler()
        self.lastAccuracy = 0
        self.didFit = False

    def update(self, states, actions):
        # (...,8) data shape
        X = np.array([value for value in states])
        # (...,1) label shape
        y = np.array([value[0] for value in actions])
        
        if self.debug:
            print("X:",X.shape)
            print("y:",y.shape)

        # undersample and shuffle
        if self.undersample == True and self.isRegressor == False:
            X, y = self.randomUnderSampler.fit_resample(X, y)
        if self.shuffle == True:
            X, y = shuffle(X, y)

        # dividing X, y into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

        # update model
        if self.partialFit == True:
            self.model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        else:
            self.model.fit(X_train, y_train)
        # save accuracy
        self.lastAccuracy = self.model.score(X_test, y_test)
        self.didFit = True

        return self.lastAccuracy

# Lookup memory collector and model trainer
class Memory():
    def __init__(self,minimumHorizon,stateResolutions,env:gym.Env,maximumHorizon=None,predictionLow=0,predictionHigh=1,minimumReward=-17,outputSize=2, wrapAction = False, useModel = True) -> None:
        self.maximumHorizon = maximumHorizon
        self.minimumHorizon = minimumHorizon
        self.DISCRETE_OS_SIZE = stateResolutions
        self.projection = None
        self.projectionStep = 0
        self.projectionLength = None
        self.env = env
        self.predictionLow = predictionLow
        self.predictionHigh = predictionHigh
        self.minimumReward = minimumReward
        self.outputSize = outputSize
        # save the output type
        self.isOutputContinuous = isinstance(predictionLow, float) if True else False
        self.wrapAction = wrapAction
        self.useModel = useModel
        self.attentionStartValue = 0.99
        self.attentionEndValue = 1.0
        self.input_space_high = env.observation_space.high
        self.input_space_low =  env.observation_space.low
        self.model : Model
        self.customRewardCallBack = None

    def setCustomStateHighLow(self, high, low):
        self.input_space_high = high
        self.input_space_low = low 

    def setAttention(self, start, end = 1.0):
        self.attentionStartValue = start
        self.attentionEndValue = end

    def setModel(self, newModel : Model):
        self.model = newModel

    def initialize(self):
        DISCRETE_OS_SIZE = self.DISCRETE_OS_SIZE
        self.discrete_os_win_size = (self.input_space_high - self.input_space_low) / [i-1 for i in DISCRETE_OS_SIZE]
        # discrete action-state table
        self.q_table = dict()
        # continuous state table
        self.c_table = dict()
        # reward table
        self.reward_table = dict()

    def get_discrete_state(self,state):
        ds = (state - self.input_space_low) / self.discrete_os_win_size
        return tuple(ds.astype(np.int32))

    def get_action(self,index, map ,c_index):
        if map and self.model.didFit:
            return self.get_model_action(c_index)

        if self.projection is None or self.projectionStep >= self.projectionLength:
            # get action from look up table if index exist
            if index in self.q_table:
                self.projection = self.q_table[index]
                self.projectionLength = len(self.projection)
            else:
                # get action from fitted model with ~%5 chance
                if self.model.didFit and random.randint(0,20) == 3:
                    self.projection = [self.get_model_action(c_index)]
                    self.projectionLength = 1
                else:
                    # get random action 
                    if self.isOutputContinuous == True:
                        self.projection = np.random.uniform(self.predictionLow,self.predictionHigh, size = (self.outputSize,))
                    else:
                        self.projection = np.random.randint(self.predictionLow,self.predictionHigh+1, size = (self.outputSize,))
                    self.projectionLength = self.outputSize
            self.projectionStep = 0

        action = self.projection[self.projectionStep]

        self.projectionStep += 1

        # distract agent 
        if self.projectionStep > random.randint(0,self.projectionLength):
            self.projection = None

        return action

    def cache(self,observation,continuousObservation,action,reward):
        self.actionCache.append(action)
        self.observationCache.append(observation)
        self.rewardCache.append(reward)
        self.continuousObservationCache.append(continuousObservation)

    def openCache(self):
        self.actionCache = []
        self.observationCache = []
        self.continuousObservationCache = []
        self.rewardCache = []
        self.indexCache = []

    def slice_by_values(self, lst, value1, value2):
        # Değerleri küçükten büyüğe sırala
        value1, value2 = min(value1, value2), max(value1, value2)

        # Değerlere karşılık gelen indeksleri bul
        start_index = int(value1 * len(lst))
        end_index = int(value2 * len(lst))

        # Dilimi alıp döndür
        sliced_list = lst[start_index:end_index]
        return sliced_list

    def train(self):
        observationLength = len(self.actionCache)
        if observationLength < 1 :
            return
        
        maxRange = min(self.maximumHorizon, observationLength) if self.maximumHorizon != None else observationLength
        for currentHorizon in range(self.minimumHorizon, maxRange,1):
            if currentHorizon == 0: continue
            for t in range(0,observationLength - currentHorizon,1):
                if t == 0: continue
                
                observation = self.observationCache[t]
                c_observation = self.continuousObservationCache[t]
                actionList = self.actionCache[t: currentHorizon + t]
                lastReward = self.rewardCache[t-1]
                
                horizonTarget = currentHorizon + t
                reward = (sum(self.slice_by_values(self.rewardCache[t:horizonTarget],self.attentionStartValue, self.attentionEndValue)) / currentHorizon)

                # if this sequence increases the reward
                if reward > lastReward:

                    tableDiff = self.reward_table.get(observation,self.minimumReward)
                    # and better than the old best
                    if reward > tableDiff:
                        # save action list
                        self.q_table[observation] = actionList
                        self.c_table[observation] = c_observation
                        self.reward_table[observation] = reward

    def fitModel(self):
        if self.useModel:
            self.model.update(self.c_table.values(), self.q_table.values())
 
    def get_model_action(self, x):
        x = np.array([x])
        action = self.model.model.predict(x)[0]
        return action

    def learn(self, maxReward = None,
                    MAX_TIME_FRAMES = 500,
                    MAX_EPISODES = 2000000,
                    SHOW_EVERY = 99,
                    STATS_EVERY = 99,

                    epsilon = 1,
                    EPSILON_THRESHOLD = 0.01,
                    epsilon_decay_value = 0.994,
                    targetReward = None,
                    render = True,
                    renderEveryStep = True
        ):

        ep_rewards = []
        real_rewards = []
        didFinish = False
        self.epsilon = epsilon
        haveCustomReward = self.customRewardCallBack != None if True else False

        print(f'{self.env.unwrapped.spec.id} environment starting to learn with {self.model.type}')

        for episode in range(MAX_EPISODES):
            real_episode_reward = 0
            episode_reward = 0
            observation = self.env.reset()
            done = False
            self.openCache()

            # fit naive bayes model from collected data with classic memory
            if episode % SHOW_EVERY == 0 and episode != 0:
                self.fitModel()

            for time_frame in range(MAX_TIME_FRAMES):
                # get dicrete observation
                discrete_ob = self.get_discrete_state(observation)  

                if episode % SHOW_EVERY == 0 and episode != 0 or didFinish:
                    # naive bayes model action
                    action = self.get_action(discrete_ob,self.useModel,c_index=observation)
                else:
                    # classic memory action
                    action = self.get_action(discrete_ob,False,c_index=observation)

                lastObservation = observation
                
                observation, r, done, _ = self.env.step(np.array([action]) if self.wrapAction else action)

                real_episode_reward += r

                if haveCustomReward:
                    r = self.customRewardCallBack(observation,action)
                
                episode_reward += r

                # cache S,A,R
                self.cache(observation=discrete_ob,continuousObservation=lastObservation,action=action,reward=r)

                # after solve, render every episode
                if render and episode % SHOW_EVERY == 0 and episode != 0 or didFinish or renderEveryStep:
                    self.env.render()

                if done:
                    if episode % SHOW_EVERY == 0 and episode != 0:
                        if maxReward != None and real_episode_reward >= maxReward:
                            print("ENVIRONMENT SOLVED AFTER",episode,"EPISODE")
                            didFinish = True
                    break

            if self.epsilon >= EPSILON_THRESHOLD:
                self.epsilon *= epsilon_decay_value

            ep_rewards.append(episode_reward)
            real_rewards.append(real_episode_reward)

            # log
            if didFinish:
                print(f'model evaluation reward: {real_episode_reward:>4.1f}, memory size: {self.q_table.__len__():>4.1f}, model accuracy: {self.model.lastAccuracy:.2f}')
            elif not episode % STATS_EVERY and episode != 0:
                average_real_reward = sum(real_rewards[-STATS_EVERY:]) / STATS_EVERY

                print(f'Episode: {episode:>5d}, average memory reward: {average_real_reward:>4.1f}, model evaluation reward: {real_episode_reward:>4.1f}, memory size: {self.q_table.__len__():>4.1f}, model accuracy: {self.model.lastAccuracy:.2f}, epsilon: {self.epsilon:.2f}')
            
            if targetReward is not None and real_episode_reward >= targetReward:
                print("target reward achieved. evaluating...")
                self.evaluate()
                return

            # train classic memory
            self.train()

        self.env.close()

    def evaluate(self, MAX_TIME_FRAMES = 500, EPISODES = 5, STATS_EVERY = 1):
        real_rewards = []

        for episode in range(EPISODES):
            real_episode_reward = 0
            observation = self.env.reset()
            done = False

            for _ in range(MAX_TIME_FRAMES):
                action = self.get_model_action(observation)

                observation, r, done, _ = self.env.step(np.array([action]) if self.wrapAction else action)

                real_episode_reward += r
                if done:
                    break

            real_rewards.append(real_episode_reward)
            if not episode % STATS_EVERY and episode != 0:
                average_real_reward = sum(real_rewards[-STATS_EVERY:]) / STATS_EVERY

                print(f'Evaluation Episode: {episode}, average evaluation reward: {average_real_reward:>4.1f}')
                
        self.env.close()
        
####################################################################################################################################################################################################################

def learn_pendulum(MAX_EPISODES):
    env = gym.make("Pendulum-v1")
    memory = Memory(maximumHorizon=12,minimumHorizon=1,stateResolutions=[21,21,65],predictionLow=-2,predictionHigh=2,env=env,minimumReward=-17, wrapAction=True,useModel=True)
    memory.setModel(Model(
            type = Model.GaussianProcessRegressor,
            partialFit = False,
    ))

    env.seed(seed)
    memory.initialize()
    memory.learn(render=True, renderEveryStep=False, MAX_EPISODES=MAX_EPISODES)

def learn_lunar_lander(MAX_EPISODES):
    env = gym.make("LunarLander-v2")
    memory = Memory(minimumHorizon=1,stateResolutions=[21,21,45,45,45,45,2,2],predictionLow=0,predictionHigh=3,env=env,minimumReward=-100,
                    useModel=True)
    memory.setModel(Model(
        type = Model.GaussianNaiveBayes,
        partialFit=True
    ))
    memory.setAttention(start = 0.0)
    memory.setCustomStateHighLow(np.array([1.5, 1.5, 5., 5., 3.14, 5.,1,1]), np.array([-1.5, -1.5, -5., -5., -3.14, -5.,0,0]))
    memory.customRewardCallBack = lambda observation,action: -me(np.array([0,0,0,0,0,0]), observation[:-2]) + (.001 if action < 1 else 0)     

    env.seed(seed)
    memory.initialize()
    memory.learn(render=True, renderEveryStep=False, MAX_EPISODES=MAX_EPISODES)

def learn_mountain_car(MAX_EPISODES):
    env = gym.make("MountainCarContinuous-v0")
    memory = Memory(maximumHorizon=90,minimumHorizon=1,stateResolutions=[21,65],predictionLow=-1,predictionHigh=1,env=env,minimumReward=-17, wrapAction=True)
    memory.setModel(Model(
        type = Model.GaussianNaiveBayes,
        partialFit=True
    ))
    memory.customRewardCallBack = lambda observation, _: -me(np.array([0.5, 0]), observation) 
    memory.setAttention(start = 0.5)

    env.seed(seed)
    memory.initialize()
    memory.learn(render=True, renderEveryStep=False, MAX_EPISODES=MAX_EPISODES)

def learn_cart_pole(MAX_EPISODES):
    env = gym.make("CartPole-v1")
    memory = Memory(maximumHorizon=25,minimumHorizon=20,stateResolutions=[65,65,65,65],predictionLow=0,predictionHigh=1,env=env)
    memory.setModel(Model(
            type = Model.XGBClassifier,
            partialFit=False
    ))
    memory.setCustomStateHighLow(np.array([4.8, 4, 0.4, 4]), np.array([-4.8, -4, -0.4, -4]))
    memory.customRewardCallBack = lambda observation, _: -me(np.array([0,0,0,0]), observation) 

    env.seed(seed)
    memory.initialize()
    memory.learn(render=True, renderEveryStep=False, MAX_EPISODES=MAX_EPISODES)

def learn_acrobot(MAX_EPISODES):
    env = gym.make('Acrobot-v1')
    memory = Memory(maximumHorizon=12,minimumHorizon=1,stateResolutions=[45,3,3,3,3,3],predictionLow=0,predictionHigh=2,env=env,minimumReward=-100, useModel=False)
    memory.setModel(Model(
        type = Model.XGBClassifier,
        partialFit=False
    ))
    memory.customRewardCallBack = lambda observation, _: -me(np.array([-0.5,  -1, -0.5, -1]), observation[:4]) 

    env.seed(seed)
    memory.initialize()
    memory.learn(render=True, renderEveryStep=False, MAX_EPISODES=MAX_EPISODES)

####################################################################################################################################################################################################################

learn_pendulum(MAX_EPISODES = 300)

learn_lunar_lander(MAX_EPISODES = 300)

learn_mountain_car(MAX_EPISODES = 300)

learn_cart_pole(MAX_EPISODES = 300)

learn_acrobot(MAX_EPISODES = 300)

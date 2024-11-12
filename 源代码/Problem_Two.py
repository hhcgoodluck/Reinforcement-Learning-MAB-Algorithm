"""Reinforcement Learning -- Bandit Algorithm"""
from typing import Any
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import signedinteger
from scipy.stats import norm

# 读取 ratings.dat 文件
ratings = pd.read_csv('ratings.dat', sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                      engine='python', encoding='latin1')

# 读取 movies.dat 文件
movies = pd.read_csv('movies.dat', sep='::', header=None, names=['MovieID', 'Title', 'Genres'], engine='python',
                     encoding='latin1')

# 将电影的类别字符串转换为类别列表
movies['Genres'] = movies['Genres'].str.split('|')


def map_movies_to_genres(movies_df) -> dict:
    """ 根据电影数据框中的类别信息，创建一个字典，将每个类别对应的电影ID存储到相应的列表中。
        movies_df: 包含电影信息的数据框，必须包含 'MovieID' 和 'Genres' 列
        genres_list: 电影类别的列表
        一个字典，键为类别，值为电影ID的列表
    """
    # 定义电影类别列表
    genres_list = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    # 初始化一个字典，用于存储每个类别对应的电影ID列表
    genre_to_movies = {genre: [] for genre in genres_list}

    # 遍历所有电影，根据电影的类别将其电影ID添加到对应的类别列表中
    for _, row in movies_df.iterrows():
        movie_id = row['MovieID']
        genres = row['Genres']

        for genre in genres:
            if genre in genre_to_movies:
                genre_to_movies[genre].append(movie_id)
    return genre_to_movies


def calculate_genre_stats(ratings_df, genre_to_movies: dict) -> dict:
    """计算每个电影类别的评分均值和方差"""
    # 定义电影类别列表
    genres_list = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    genre_stats = {genre: {'mean': 0, 'var': 0} for genre in genres_list}

    for genre, movie_ids in genre_to_movies.items():
        # 获取该类别的电影评分
        genre_ratings = ratings_df[ratings_df['MovieID'].isin(movie_ids)]['Rating']

        if not genre_ratings.empty:
            # 计算均值和方差
            genre_stats[genre]['mean'] = genre_ratings.mean()
            genre_stats[genre]['var'] = genre_ratings.var()

    return genre_stats


class Bandit:
    """Bandit类 遵循亚高斯分布"""

    def __init__(self, genre_stats: dict) -> None:
        """ 初始化 Bandit 类
            genre_stats: 每个电影类别的均值和方差的字典。
            genres_list: 电影类别列表。
        """
        genres_list = [
            "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

        self.genres_list = genres_list  # 电影类别的列表
        self.K = len(genres_list)  # 臂的数量，即电影类别的数量
        self.means = np.zeros(self.K)  # 每个臂的均值 初始值为K个0的numpy数组
        self.variances = np.zeros(self.K)  # 每个臂的方差 初始值为K个0的numpy数组
        self._initialize_bandit(genre_stats)  # 从 genre_stats 初始化均值和方差

    def _initialize_bandit(self, genre_stats: dict) -> None:
        """ 根据每个电影类别的统计数据初始化 Bandit 类的奖励分布
            genre_stats: 每个电影类别的均值和方差的字典
        """
        for i, genre in enumerate(self.genres_list):  # enumerate 用于同时获取列表项的索引和值
            if genre in genre_stats:
                self.means[i] = genre_stats[genre]['mean']
                self.variances[i] = genre_stats[genre]['var']

    def sample_reward(self, arm: int) -> int:
        """ 从指定的臂（index）的奖励分布中抽样 即一次随机试验 返回从亚高斯分布中抽取的奖励值
            arm: 选择的臂（电影类别的索引）
        """
        # 设置亚高斯分布的标准差评分范围从1到5，最大差异为4
        B = 4
        sigma = B // 2

        # 从均值和方差创建一个正态分布
        distribution = norm(loc=self.means[arm], scale=np.sqrt(self.variances[arm] + sigma))

        # 从分布中抽样
        return distribution.rvs()


class Solver:
    """ 多臂老虎机算法基本框架 """

    def __init__(self, bandit) -> None:
        self.bandit = bandit  # bandit 是上边刚创建的实例
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数记录的列表
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔
        self.rewards = [[] for _ in range(self.bandit.K)]  # 记录每个臂的奖励的数组

    def update_regret(self, k: int) -> None:
        """计算累积懊悔并保存,k为本次动作选择的拉杆的编号"""
        self.regret += self.bandit.means.max() - self.bandit.means[k]  # bandit.means 是一个数组
        self.regrets.append(self.regret)

    def update_rewards(self, k) -> None:
        """更新奖励记录"""
        reward = self.bandit.sample_reward(k)
        self.rewards[k].append(reward)

    def run_one_step(self) -> int:
        """返回当前动作索引（index）选择哪一根拉杆,由每个具体的策略实现"""
        raise NotImplementedError

    def run(self, num_steps: int) -> None:  # 在执行TS算法的run时 不需要屏蔽掉
        """运行一定次数,num_steps为总运行次数"""
        for _ in range(num_steps):
            k = self.run_one_step()  # 获得动作臂索引index
            self.counts[k] += 1
            self.actions.append(k)
            self.update_rewards(k)
            self.update_regret(k)


class ETC_Algorithm(Solver):
    def __init__(self, bandit, exploration_length: int) -> None:
        super(ETC_Algorithm, self).__init__(bandit)  # 继承Solver类属性
        self.exploration_length = exploration_length  # ETC算法属性 探索轮数 即参数m
        self.empirical_means = np.zeros(self.bandit.K)  # 初始化经验均值数组

    def run_one_step(self) -> int | signedinteger[Any] | Any:
        """ETC算法执行每一步前决定arm的index"""
        current_step = len(self.actions)  # 当前步数

        if current_step <= self.exploration_length:
            # 在探索阶段，均匀选择拉杆
            return current_step % self.bandit.K
        else:
            # 计算每个拉杆的经验平均值
            for k in range(self.bandit.K):
                if len(self.rewards[k]) > 0:
                    self.empirical_means[k] = np.mean(self.rewards[k])
                else:
                    # 如果没有奖励记录，则设为负无穷以避免选择
                    self.empirical_means[k] = -np.inf

            # 选择经验平均值最大的拉杆
            return np.argmax(self.empirical_means)


def run_etc_experiments(bandit, exploration_length, num_steps, num_experiments) -> list:
    """ 运行多次ETC算法实验并记录累积懊悔，并返回所有实验中每一步的累积懊悔列表
        bandit: Bandit 类实例
        exploration_length: 探索阶段的长度
        num_steps: 总步数
        num_experiments: 实验次数
    """
    all_regrets = []

    for _ in range(num_experiments):
        # 创建 ETC_Algorithm 实例并运行实验
        algorithm = ETC_Algorithm(bandit, exploration_length)
        algorithm.run(num_steps)
        # 将每次实验的累积懊悔记录到 all_regrets
        all_regrets.append(algorithm.regrets)

    return all_regrets


class UCB_Algorithm(Solver):
    def __init__(self, bandit):
        super(UCB_Algorithm, self).__init__(bandit)

    def run_one_step(self) -> int | signedinteger[Any]:
        """UCB算法执行每一步前决定arm的index"""
        n = 50000
        t = len(self.actions) + 1  # 在前K轮中算法选择每个臂一次
        if t <= self.bandit.K:  # 当前轮数round t小于或等于K 故返回t - 1以确保每个臂在初始轮中被选择一次
            return t - 1
        else:
            B = 4
            empirical_means = [np.mean(self.rewards[k]) if len(self.rewards[k]) > 0
                               else 0 for k in range(self.bandit.K)]  # 经验奖励均值
            ucb_values = [empirical_means[k] + B // 2 * np.sqrt(4 * np.log(n) / self.counts[k])
                          for k in range(self.bandit.K)]  # UCB算法公式
            return np.argmax(ucb_values)


def run_ucb_experiments(bandit, num_steps, num_experiments) -> list:
    """ 运行多次UCB算法实验并记录累积懊悔，并返回所有实验中每一步的累积懊悔列表
        bandit: Bandit 类实例
        num_steps: 总步数
        num_experiments: 实验次数
    """
    all_regrets = []

    for _ in range(num_experiments):
        algorithm = UCB_Algorithm(bandit)  # 创建 UCB_Algorithm 实例并运行实验
        algorithm.run(num_steps)
        all_regrets.append(algorithm.regrets)  # 将每次实验的累积懊悔记录到 all_regrets

    return all_regrets


class GaussianBandit:
    """GaussianBandit类，遵循高斯分布"""

    def __init__(self, genre_stats: dict) -> None:
        """ 初始化 GaussianBandit 类
            genre_stats: 每个电影类别的均值和方差的字典。
        """
        genres_list = [
            "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

        self.genres_list = genres_list  # 电影类别的列表
        self.K = len(genres_list)  # 臂的数量，即电影类别的数量
        self.means = np.zeros(self.K)  # 初始化每个臂的均值
        self.vars = np.zeros(self.K)  # 初始化每个臂的方差
        self._initialize_bandit(genre_stats)  # 从 genre_stats 初始化均值和方差

    def _initialize_bandit(self, genre_stats: dict) -> None:
        """ 根据每个电影类别的统计数据初始化 GaussianBandit 类的奖励分布
            genre_stats: 每个电影类别的均值和方差的字典
        """
        for i, genre in enumerate(self.genres_list):
            if genre in genre_stats:
                self.means[i] = genre_stats[genre]['mean']
                self.vars[i] = genre_stats[genre]['var']

    def sample_reward(self, arm: int) -> float:
        """ 从指定的臂（index）的奖励分布中抽样，即一次随机试验，返回从高斯分布中抽取的奖励值
            arm: 选择的臂（电影类别的索引）
        """
        return np.random.normal(self.means[arm], np.sqrt(self.vars[arm]))


class TS_Algorithm(Solver):
    def __init__(self, bandit: GaussianBandit, B: int = 4) -> None:
        super(TS_Algorithm, self).__init__(bandit)
        self.mu = np.zeros(self.bandit.K)  # 均值的初始值
        self.sigma_squared = np.full(self.bandit.K, (B ** 2) / 4)  # 方差的初始值
        self.counts = np.zeros(self.bandit.K)  # 记录每个臂的选择次数
        self.B = B  # 奖励的范围差

    def run_one_step(self) -> int | signedinteger[Any]:
        """TS 算法执行每一步前决定 arm 的 index"""
        if len(self.actions) < self.bandit.K:
            # 前 k 轮，每次选择一个不同的臂
            return len(self.actions)
        else:
            # 计算每个臂的采样值，并选择具有最大采样值的臂
            sampled_means = [np.random.normal(self.mu[k], np.sqrt(self.sigma_squared[k])) for k in range(self.bandit.K)]
            return np.argmax(sampled_means)

    def run(self, num_steps: int) -> None:
        """运行一定次数, num_steps为总运行次数"""
        for _ in range(num_steps):
            k = self.run_one_step()  # 获得动作臂索引 index
            self.counts[k] += 1
            self.actions.append(k)

            # 更新奖励和置信度
            reward = self.bandit.sample_reward(k)
            self.update_rewards(k)

            # 更新均值
            self.mu[k] = (self.mu[k] * (self.counts[k] - 1) + reward) / self.counts[k]
            # 更新方差
            self.sigma_squared[k] = (self.B ** 2 / 4) / self.counts[k]

            self.update_regret(k)


def run_ts_experiments(bandit: GaussianBandit, num_steps: int, num_experiments: int) -> list:
    """运行多次TS算法实验并记录累积懊悔，并返回所有实验中每一步的累积懊悔列表
        bandit: Bandit 类实例
        num_steps: 总步数
        num_experiments: 实验次数
    """
    all_regrets = []

    for _ in range(num_experiments):
        algorithm = TS_Algorithm(bandit)
        algorithm.run(num_steps)
        all_regrets.append(algorithm.regrets)

    return all_regrets


def plot_average_regret(bandit, horizon_values, genre_stats, num_experiments):
    """
    绘制三个算法在不同步长下的平均懊悔值曲线。
    bandit: Bandit 类实例
    horizon_values: 不同的总步数列表
    num_experiments: 每个步长下的实验次数
    """
    plt.figure(figsize=(12, 8))

    for n in horizon_values:
        # 探索阶段长度为总步长的10%
        exploration_length = int(0.1 * n)

        # 运行 ETC 算法实验
        etc_regrets = run_etc_experiments(bandit, exploration_length, n, num_experiments)
        avg_etc_regret = np.mean(etc_regrets, axis=0)

        # 运行 UCB 算法实验
        ucb_regrets = run_ucb_experiments(bandit, n, num_experiments)
        avg_ucb_regret = np.mean(ucb_regrets, axis=0)

        # 运行 TS 算法实验
        ts_bandit = GaussianBandit(genre_stats)  # TS算法需要高斯分布的奖励模型
        ts_regrets = run_ts_experiments(ts_bandit, n, num_experiments)
        avg_ts_regret = np.mean(ts_regrets, axis=0)

        # 绘制三种算法的平均懊悔曲线
        plt.plot(range(n), avg_etc_regret, label=f'ETC (n={n})')
        plt.plot(range(n), avg_ucb_regret, label=f'UCB (n={n})')
        plt.plot(range(n), avg_ts_regret, label=f'TS (n={n})')

        # 设置图例和标题
        plt.xlabel('Steps')
        plt.ylabel('Average Regret')
        plt.title(f'Average Regret for Different Algorithms (n={n})')
        plt.legend()

        # 保存图像
        plt.savefig('/Users/huohongcheng/desktop/regret.png')
        plt.clf()  # 清除当前图像，准备绘制下一个图像

    plt.show()

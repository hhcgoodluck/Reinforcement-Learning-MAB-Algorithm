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
            k = self.run_one_step()         # 获得动作臂索引index
            self.counts[k] += 1
            self.actions.append(k)
            self.update_rewards(k)
            self.update_regret(k)


class ETC_Algorithm(Solver):
    def __init__(self, bandit, exploration_length: int) -> None:
        super(ETC_Algorithm, self).__init__(bandit)     # 继承Solver类属性

        self.exploration_length = exploration_length    # ETC算法属性 探索阶段的总长度
        self.empirical_means = np.zeros(self.bandit.K)  # 初始化经验均值数组

    def run_one_step(self) -> int | signedinteger[Any] | Any:
        """ETC算法执行每一步前决定arm的index"""
        current_step = len(self.actions)                # 当前步数

        if current_step <= self.exploration_length:     # 探索阶段总的步数 exploration_length
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


def run_ETC_experiments(bandit: Bandit, num_steps: int, num_experiments: int, mk_values: list) -> dict:
    """ 运行ETC算法实验
        bandit:          Bandit对象
        num_steps:       总运行步数
        num_experiments: 实验总次数
        mk_values:       探索阶段总步数的列表
    """
    all_regrets = {value: [] for value in mk_values}

    for value in mk_values:
        for _ in range(num_experiments):
            algorithm = ETC_Algorithm(bandit, value)
            algorithm.run(num_steps)
            all_regrets[value].append(algorithm.regrets)

    return all_regrets


def plot_ETC_results(all_regrets, mk_values, num_steps):
    """ 绘制ETC算法的累积懊悔
        all_regrets: 不同探索长度的累积懊悔结果
        m_values: 探索长度的列表
        num_steps: 总运行步数
    """
    plt.figure(figsize=(10, 6))

    for value in mk_values:
        regrets = np.array(all_regrets[value])
        mean_regrets = np.mean(regrets, axis=0)
        std_regrets = np.std(regrets, axis=0)

        plt.plot(range(num_steps), mean_regrets, label=f'm * k = {value}')
        plt.fill_between(range(num_steps),
                         mean_regrets - std_regrets,
                         mean_regrets + std_regrets,
                         alpha=0.2)

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.title('Performance of ETC Algorithm for Different Exploration Lengths')
    plt.grid(True)
    plt.show()
    plt.savefig('/Users/huohongcheng/desktop/etc_regret_with_exploration.png')

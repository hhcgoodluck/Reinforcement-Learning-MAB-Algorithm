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


class Asy_UCB_Algorithm(Solver):
    def __init__(self, bandit: Bandit, para_l: int) -> None:
        super(Asy_UCB_Algorithm, self).__init__(bandit)
        self.para_l = para_l

    def run_one_step(self) -> int | signedinteger[Any]:
        """Asy-UCB算法执行每一步前决定arm的index"""
        n = 50000
        t = len(self.actions) + 1  # 在前K轮中算法选择每个臂一次
        if t <= self.bandit.K:     # 当前轮数round t小于或等于K 故返回t - 1以确保每个臂在初始轮中被选择一次
            return t - 1
        else:
            B = 4
            empirical_means = [np.mean(self.rewards[k]) if len(self.rewards[k]) > 0
                               else 0 for k in range(self.bandit.K)]  # 经验奖励均值
            Asy_ucb_values = [empirical_means[k] + B // 2 * np.sqrt(self.para_l * np.log(n) / self.counts[k])
                              for k in range(self.bandit.K)]          # Asy-UCB算法公式
            return np.argmax(Asy_ucb_values)


def run_UCB_experiments(bandit: Bandit, num_steps: int, num_experiments: int) -> list:
    """
    运行UCB算法实验
    """
    all_regrets = []

    for _ in range(num_experiments):
        algorithm = UCB_Algorithm(bandit)
        algorithm.run(num_steps)
        all_regrets.append(algorithm.regrets)

    return all_regrets


def run_Asy_UCB_experiments(bandit, num_steps, num_experiments, para_l: int) -> list:
    """
    运行Asy_UCB算法实验
    """
    all_regrets = []

    for _ in range(num_experiments):
        algorithm = Asy_UCB_Algorithm(bandit, para_l)
        algorithm.run(num_steps)
        all_regrets.append(algorithm.regrets)

    return all_regrets


def plot_ucb_vs_asy_ucb(bandit, num_steps, num_experiments):
    """运行 UCB 和 Asy-UCB 算法实验，并绘制结果"""
    # 运行 UCB 算法实验
    ucb_all_regrets = run_UCB_experiments(bandit, num_steps, num_experiments)
    ucb_mean_regrets = np.mean(ucb_all_regrets, axis=0)
    ucb_std_regrets = np.std(ucb_all_regrets, axis=0)

    # 运行 Asy-UCB 算法实验，使用不同的 l 值
    l_values = [1, 2, 4]
    asy_ucb_results = {}

    for para_l in l_values:
        asy_ucb_all_regrets = run_Asy_UCB_experiments(bandit, num_steps, num_experiments, para_l)
        asy_ucb_mean_regrets = np.mean(asy_ucb_all_regrets, axis=0)
        asy_ucb_std_regrets = np.std(asy_ucb_all_regrets, axis=0)
        asy_ucb_results[para_l] = (asy_ucb_mean_regrets, asy_ucb_std_regrets)

    # 绘制结果
    plt.figure(figsize=(12, 8))
    x = np.arange(num_steps)

    # 绘制 UCB 结果
    plt.plot(x, ucb_mean_regrets, label='UCB', color='blue')
    plt.fill_between(x, ucb_mean_regrets - ucb_std_regrets, ucb_mean_regrets + ucb_std_regrets, color='blue', alpha=0.1)

    # 绘制 Asy-UCB 结果
    colors = ['red', 'green', 'orange']
    for i, l in enumerate(l_values):
        mean_regrets, std_regrets = asy_ucb_results[l]
        plt.plot(x, mean_regrets, label=f'Asy-UCB l={l}', color=colors[i])
        plt.fill_between(x, mean_regrets - std_regrets, mean_regrets + std_regrets, color=colors[i], alpha=0.1)

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.title('Cumulative Regret Performance of UCB and Asy-UCB with Different l values')
    plt.show()
    plt.savefig('/Users/huohongcheng/desktop/ucb_vs_asy_ucb.png')   # 指定保存路径



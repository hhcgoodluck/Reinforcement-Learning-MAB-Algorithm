# 问题一测试代码
# ETC算法测试样例
"""
    # 初始化 Bandit 实例
    genre_to_movies = map_movies_to_genres(movies)
    genre_stats = calculate_genre_stats(ratings, genre_to_movies)
    bandit = Bandit(genre_stats)

    # 设置实验参数
    exploration_length = 5000  # 探索阶段的长度
    num_steps = 50000  # 总步数
    num_experiments = 10  # 实验次数 绘制累积懊悔曲线
    num_experiments = 100 # 实验次数 绘制平均累计懊悔曲线

    # 运行多次 ETC 算法实验并记录累积懊悔
    etc_regrets = run_etc_experiments(bandit, exploration_length, num_steps, num_experiments)

    # 绘制累积懊悔曲线
    plot_etc_regret_curves(etc_regrets, 'ETC Algorithm', num_steps, num_experiments)

    # 绘制平均累计懊悔曲线
    plot_etc_regret_with_error_bars(etc_regrets, 'ETC Algorithm', num_steps)
"""

# UCB算法测试样例
"""
    # 初始化 Bandit 实例
    genre_to_movies = map_movies_to_genres(movies)
    genre_stats = calculate_genre_stats(ratings, genre_to_movies)
    bandit = Bandit(genre_stats)
    
    # 设置实验参数
    num_steps = 50000  # 总步数
    num_experiments = 10  # 实验次数 绘制累积懊悔曲线
    num_experiments = 100 # 实验次数 绘制平均累计懊悔曲线
    
    # 运行多次 UCB 算法实验并记录累积懊悔
    ucb_regrets = run_ucb_experiments(bandit, num_steps, num_experiments)
    
    # 绘制累积懊悔曲线
    plot_ucb_regret_curves(ucb_regrets, 'UCB Algorithm', num_steps, num_experiments)
    
    # 绘制平均累计懊悔曲线
    plot_ucb_regret_with_error_bars(ucb_regrets, 'UCB Algorithm', num_steps)
"""

# TS算法测试样例
"""
    # 初始化 Bandit 实例
    genre_to_movies = map_movies_to_genres(movies)
    genre_stats = calculate_genre_stats(ratings, genre_to_movies)
    bandit = GaussianBandit(genre_stats)
    
    # 设置实验参数
    num_steps = 50000  # 总步数
    num_experiments = 10   # 实验次数 绘制累积懊悔曲线
    num_experiments = 100  # 实验次数 绘制平均累计懊悔曲线
    
    # 运行多次 TS 算法实验并记录累积懊悔
    ts_regrets = run_ts_experiments(bandit, num_steps, num_experiments)
    
    # 绘制累积懊悔曲线
    plot_ts_regret_curves(ts_regrets, 'TS Algorithm', num_steps, num_experiments)
    
    # 绘制平均累计懊悔曲线
    plot_ts_regret_with_error_bars(ts_regrets, 'TS Algorithm', num_steps)
"""

# 问题三测试代码
# 关于不同探索长度的ETC算法表现
"""
    # 准备数据
    genre_to_movies = map_movies_to_genres(movies)
    genre_stats = calculate_genre_stats(ratings, genre_to_movies)
    bandit = Bandit(genre_stats)
    
    # 设置参数
    num_steps = 50000
    mk_values = [50, 500, 2000, 5000, 10000]  # 每个探索阶段的总长度
    mk_values = [5, 50, 100, 500]             # 小数据集测试
    num_experiments = 100
    
    # 运行ETC实验
    all_regrets = run_ETC_experiments(bandit, num_steps, num_experiments, mk_values)
    
    # 绘制结果
    plot_ETC_results(all_regrets, mk_values, num_steps)
"""

# 问题四测试代码
""" # 初始化Bandit实例
    genre_to_movies = map_movies_to_genres(movies)
    genre_stats = calculate_genre_stats(ratings, genre_to_movies)
    bandit = Bandit(genre_stats)
    
    # 设置实验参数并运行绘图函数
    num_steps = 50000
    num_experiments = 100
    
    # 运行实验并绘制结果
    plot_ucb_vs_asy_ucb(bandit, num_steps, num_experiments)
"""

# 问题二测试代码
"""
    genre_to_movies = map_movies_to_genres(movies)
    genre_stats = calculate_genre_stats(ratings, genre_to_movies)
    bandit = Bandit(genre_stats)

    # 不同步长值
    horizon_values = [500, 5000, 50000, 500000, 5000000] # 还是一次一个来算了
    
    # 运行实验并绘制图像
    plot_average_regret(bandit, horizon_values, genre_stats, num_experiments=100)
"""

# 问题五测试代码
"""
    genre_to_movies = map_movies_to_genres(movies)
    genre_stats = calculate_genre_stats(ratings, genre_to_movies)
    
    bandit = Bandit(genre_stats)
    
    num_steps = 1000000
    num_experiments = 100
    exploration_length = 100000 
    
    # 运行并绘制比较图
    plot_comparison(genre_stats, num_steps, exploration_length, num_experiments)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D

from ipdb import set_trace
"""
log_dir: base_log/env/alg/xxx/seed_i/process.csv
"""

"""
function need to be modified

LEGEND_ORDER
envs_name
get_alg_name
env_name_to_figure_name
get_plot_data_from_single_experiment
get_x_limit
"""

colors = ["orangered", "lightseagreen", "cornflowerblue", "orchid", "gray", "darkseagreen", "goldenrod", "darkorange", "mediumorchid", "darkturquoise" ]

SMALL_SIZE = 15    #30
MEDIUM_SIZE = 20   #35
BIGGER_SIZE = 25   #40
LINEWIDTH = 3      #4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

pass_alg_name = ["SACFP", "SACRB"]
# pass_alg_name = ["SFP", "SRB"]

LEGEND_ORDER = {
    "surprise_vision": 0,
    "sac":1,
    "surprise": 2,
    "rnd": 3,
    "src": 4,
    "emi":5,
    "information_gain": 6,
    "src": 7, 
    "surprise_add_resource": 8,
    "sac_minus_cost": 9,
    "sac_add_resource": 10,
    "ppo": 11
}

# LEGEND_ORDER = {
#     "surprise_vison_int01": 0,
#     "surprise_vison_int005":1,
#     "surprise_vison_int002": 2
# }

# LEGEND_ORDER = {
#     "RAEB": 0,
#     "only_resource_bonus":1,
#     "surprise": 2
# }

# envs_name = ["resource_ant_corridor_goal4", "resource_ant_corridor_goal5", "resource_ant_corridor_goal6"]
envs_name = ["goal4", "resource_cheetah_goal4", "resource_mountaincar_10"]
# envs_name = ["sensitivity"]
# envs_name = ["resource_ant_goal4_cost100"]

# envs_name = ["continuous_resource_5", "continuous_resource_10", "continuous_resource_25"]
# envs_name = ["ant_corridor_resource_env_goal_4_v0"]

legend = [r"$n=5$", r"$n=10$", r"$n=25$"]

def get_alg_name(algo_name):
    
    dict_name = {
        "surprise_vision": "RAEB",
        "sac": "SAC",
        "surprise": "Surprise",
        "information_gain": "JDRX",
        #"rnd": "RND",
        "src": "SFP",
        "surprise_add_resource": "SRB",
        "sac_minus_cost": "SACFP",
        "sac_add_resource": "SACRB",
        "ppo": "PPO"
        #"emi": "EMI"
    }

    # dict_name = {
    #     "surprise_vison_int01": r"$\eta=0.1$",
    #     "surprise_vison_int005":r"$\eta=0.05$",
    #     "surprise_vison_int002": r"$\eta=0.02$"
    # }

    # dict_name = {
    #     "RAEB": "RAEB",
    #     "only_resource_bonus": "without surprise",
    #     "surprise": "without resource"
    # }

    if algo_name not in dict_name.keys():
        # for key in dict_name.keys():
        #     if key in algo_name:
        #         return dict_name[key]
        return "NotThisAlg"

    return dict_name[algo_name]

COLORS = dict()
def get_color(label):
    if label not in COLORS.keys():
        COLORS[label] = colors.pop(0)
    return COLORS[label]

def env_name_to_figure_name(env_name):
    dic = {
        "goal4": "Cargo Delivery Ant",
        "resource_cheetah_goal4": "Cargo Delivery HalfCheetah",
        "resource_mountaincar_10": "Cargo Delivery Mountain Car",
        "sensitivity": "none",
        "component": "none",
        "resource_ant_goal4_cost100": "none",
        "continuous_resource_5": "continuous_resource_5",
        "continuous_resource_10": "continuous_resource_10",
        "continuous_resource_25": "continuous_resource_25",
        "ant_corridor_resource_env_goal_4_v0": "none"
    }

    # dic = {
    #      "continuous_resource_5": "InitialCargo 5",
    #      "continuous_resource_10": "InitialCargo 10",
    #      "continuous_resource_15": "InitialCargo 15",
    #      "continuous_resource_25": "InitialCargo 25"
    # }
    # dic = {
    #      "resource_ant_corridor_goal4": "Destination_[4,5]",
    #      "resource_ant_corridor_goal5": "Destination_[5,6]",
    #      "resource_ant_corridor_goal6": "Destination_[6,7]"
    # }
    
    if env_name not in dic.keys():
        return "NotThisEnv"
    
    return dic[env_name]

def steps_to_epoch(data_x):
    data_x = data_x / 1000
    return data_x

def epoch_to_steps(data_x, algo_name):
    if algo_name == "ppo":
        for i in range(len(data_x)):
            steps = 4000 * (data_x[i] + 1)
            data_x[i] = steps
    else:
        for i in range(len(data_x)):
            steps = 5000 + 1000 * (data_x[i] + 1)
            data_x[i] = steps
    return data_x

def get_plot_data_from_single_experiment(file_name, algo_name):
    try:
        if algo_name == "ppo":
            data = pd.read_csv(file_name, sep='\s+', error_bad_lines=False)
        else:
            data = pd.read_csv(file_name, error_bad_lines=False)
    except pd.errors.EmptyDataError:
        return None, None
    
    dic_y = {
        "surprise_vision": "evaluation/Average Returns",
        "rnd_vision": "evaluation/Average Returns",
        "sac": "evaluation/Average Returns",
        "surprise": "evaluation/Average Returns",
        "rnd": "evaluation/Average Returns",
        "src": "evaluation/Average Returns",
        "emi": "AverageRawReturn",
        "information_gain": "evaluation/Average Returns",
        "surprise_vison_int002": "evaluation/Average Returns",
        "surprise_vison_int005": "evaluation/Average Returns",
        "surprise_vison_int01": "evaluation/Average Returns",
        "RAEB": "evaluation/Average Returns",
        "only_resource_bonus": "evaluation/Average Returns",
        "add_resource_bonus": "evaluation/Average Returns",
        "sac_add_resource": "evaluation/Average Returns",
        "sac_minus_cost": "evaluation/Average Returns",
        "surprise_add_resource": "evaluation/Average Returns",
        "ppo": "AverageEpRet"
    }

    dic_x = {
        "surprise_vision": "exploration/num steps total",
        "rnd_vision": "Epoch",
        "sac": "Epoch",
        "surprise": "Epoch",
        "rnd": "Epoch",
        "src": "Epoch",
        "emi": "TotalTimesteps",
        "information_gain": "exploration/num steps total",
        "surprise_vison_int002": "exploration/num steps total",
        "surprise_vison_int005": "Epoch",
        "surprise_vison_int01": "exploration/num steps total",
        "RAEB": "exploration/num steps total",
        "only_resource_bonus": "exploration/num steps total",
        "add_resource_bonus": "exploration/num steps total",
        "sac_add_resource": "exploration/num steps total",
        "sac_minus_cost": "exploration/num steps total",
        "surprise_add_resource": "exploration/num steps total",
        "ppo": "Epoch"
    }

    # if algo_name not in dic_y.keys():
    #     for key in dic_y.keys():
    #         if key in algo_name:
    #             algo_name = key
    


    column_y = dic_y[algo_name]
    column_x = dic_x[algo_name]

    data_y = data[column_y]
    data_x = data[column_x]

    # if "steps" in dic_x[algo_name]:
    #     data_x = steps_to_epoch(data_x)

    data_x = np.array(data_x).astype(np.int)
    if "Epoch" in dic_x[algo_name]:
        data_x = epoch_to_steps(data_x, algo_name)

    return data_x, np.array(data_y).astype(np.float) # astype将array的数据类型投影到一个指定type

def get_data_from_algo_dir(algo_dir, algo_name, x_limit, alg_config):
    """
    log_dir: base_log/env/alg/xxx/seed_i/process.csv
    """
    plot_y = []
    plot_x = []
    for sub_name in os.listdir(algo_dir):
        if ".csv" in sub_name:
            # 适用于只有一个随机种子的
            file_name = os.path.join(algo_dir, sub_name)
        else:
            # 一级子目录
            sub_dir = os.path.join(algo_dir, sub_name)
            # 二级子目录
            for sub_sub_name in os.listdir(sub_dir):
                if ".csv" in sub_sub_name:
                    file_name = os.path.join(sub_dir, sub_sub_name)
                else:
                    sub_sub_dir = os.path.join(sub_dir, sub_sub_name)
                    file_name = os.path.join(sub_sub_dir, options.file_name)
                    
                if os.path.exists(file_name):
                    x, y = get_plot_data_from_single_experiment(file_name, algo_name)
                    if type(x) != type(None):
                        max_step = x[-1]
                        print("max step:%d"%max_step)
                        if max_step >= x_limit:
                            plot_x.append(x)
                            plot_y.append(y)


    if plot_x == []:
        return None, None, None, None
    plot_y = smooth(plot_y)
    x, y_mean, y_std, max_y = compute_mean_std_max(plot_x, plot_y)
    return x, y_mean, y_std, max_y

def smooth(y_data):
    # smooth y via moving average
    window_length = 30
    smooth_y=[]
    for y in y_data:
        y_ext = np.concatenate([np.zeros(window_length),y])
        tmp = [np.mean(y_ext[i:window_length+i]) for i in range(len(y))]
        smooth_y.append(tmp)
    return smooth_y

def compute_mean_std_max(plot_x, plot_y):
    max_len = len(plot_y[0])
    x = plot_x[0]
    for i in range(1, len(plot_y)):
        if len(plot_y[i]) > max_len:
            max_len = len(plot_y[i])
            x = plot_x[i]
    y_mean = []                            # should assign memory first
    y_std = []
    for itr in range(max_len):
        itr_values = []
        for curve_i in range(len(plot_y)):
            if itr < len(plot_y[curve_i]):
                itr_values.append(plot_y[curve_i][itr])
        y_mean.append(np.mean(itr_values))
        y_std.append(np.std(itr_values))

    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    return x, y_mean, y_std, np.max(y_mean)

def get_x_limit(env_name):
    dic = {
        "resource_ant_corridor_goal4": 990000,        
        "resource_ant_corridor_goal5": 990000,
        "resource_ant_corridor_goal6": 990000,
        "continuous_resource_10": 990000 
    }
    return dic[env_name] if env_name in dic else 990000 # default 1000

def data_truncate(x, y_mean, y_std, x_limit):
    # x[x<=limit] 返回所有满足条件的元素构成数组，只有数组才能这么操作
    # 这个功能是截断
    return x[x<=x_limit], y_mean[:len(x[x<=x_limit])], y_std[:len(x[x<=x_limit])]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")
    parser.add_argument("--log_dir", type=str, default=r"/home/zhwang/research/ICML2021_Finaldata/evaluation")
    parser.add_argument("--file_name", type=str, default="progress.csv")
    parser.add_argument("--column_x", type=str, default="Steps")# Time steps
    parser.add_argument("--column_y", type=str, default="Average return")
    parser.add_argument("--alg_config", type=str, default="xx")
    parser.add_argument("--fig_name", type=str, default="evaluation_add_ppo.pdf")
    parser.add_argument("--w", type=float, default=20)
    parser.add_argument("--h", type=float, default=6)
    parser.add_argument("--std_coeff", type=float, default=1)
    parser.add_argument("--mode", type=str, default="subplots")

    options = parser.parse_args()

    fig = plt.figure(figsize=(options.w,options.h))
    axs = []
    if options.mode == "subplots":
        if len(envs_name) == 3:
            gs = GridSpec(1, 3)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[0, 2])
            axs = [ax1, ax2, ax3]

        elif len(envs_name) == 4:
            gs = GridSpec(2, 2)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[1, 0])
            ax4 = plt.subplot(gs[1, 1])
            axs = [ax1, ax2, ax3, ax4]

        elif len(envs_name) == 1:
            gs = GridSpec(1,1)
            axs = [plt.subplot(gs[0, 0])]

        else:
            raise NotImplementedError
    
    else:
        gs = GridSpec(1,1)
        axs = [plt.subplot(gs[0, 0])]

    i = 0
    j = 0
    legend_line = ()
    legend_alg = ()
    log_dir = options.log_dir
    alg_config = options.alg_config
    #for env_name in os.listdir(log_dir):
    for env_name in envs_name:
        env_dir = os.path.join(log_dir, env_name)
        ax = axs[i]
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        sub_figure_name = env_name_to_figure_name(env_name)
        if sub_figure_name == "NotThisEnv":
            continue
        #print("plot: %s" % sub_figure_name)
        x_limit = get_x_limit(env_name)
        if sub_figure_name != "none":
            ax.set_title(sub_figure_name, fontsize=20)
        # ax.set_title("(a) initial cargo 5", y=-1)
        plot_data = []
        #spec_algo = ("surprise_vision","rnd_vision","sac","surprise","rnd","src")
        for algo_name in os.listdir(env_dir):
        # for algo_name in spec_algo:
            algo_dir = os.path.join(env_dir, algo_name)
            #algo_dir = env_dir
            #print(algo_dir)
            alg = get_alg_name(algo_name)
            print(algo_name)
            if alg == "NotThisAlg":
                continue
            x, y_mean, y_std, max_y = get_data_from_algo_dir(algo_dir, algo_name, x_limit, alg_config)


            plot_data.append((LEGEND_ORDER[algo_name], alg, x, y_mean, y_std, max_y))

        for _, algo_name, x, y_mean, y_std, max_y in sorted(plot_data, key=lambda x: x[0]):
            #print("\nName of Algorithm: %s"%algo_name)
            if algo_name in pass_alg_name:
                continue
            if type(x) != type(None):
                x, y_mean, y_std = data_truncate(x, y_mean, y_std, x_limit)
                line, = ax.plot(x, y_mean, label=algo_name, linewidth=LINEWIDTH, color=get_color(algo_name))
                # line, = ax.plot(x, y_mean, label=algo_name, linewidth=LINEWIDTH, color=colors[j])
                if i == 0:
                    legend_line = legend_line + (line,)
                    legend_alg = legend_alg + (algo_name,)
                    #ax.legend(legend_line, legend_alg, loc='upper left')
                ax.fill_between(x, y_mean + options.std_coeff * y_std, y_mean - options.std_coeff * y_std, alpha=0.2, color=get_color(algo_name))
                # ax.fill_between(x, y_mean + options.std_coeff * y_std, y_mean - options.std_coeff * y_std, alpha=0.2, color=colors[j])
                ax.set_xlim(0, x_limit)
                
                #print(ax.get_xlim())
                #print("DONE:%s"%algo_name) # better change the string
            else:
                print("FAILED:%s"%algo_name) # better change the string
        if i == 0:
            ax.legend(legend_line, legend_alg, loc='upper left')
        ax.set_xlabel(options.column_x)
        ax.set_ylabel(options.column_y)
        ax.xaxis.grid(True, which = 'major')
        ax.yaxis.grid(True, which = 'major')
        i += 1
        # j += 1
    
    
    plt.tight_layout(pad=4, w_pad=1.5, h_pad=3)
    #plt.tight_layout()
    if options.fig_name[-3:] == 'pdf':
        plt.savefig(os.path.join(os.getcwd(), options.fig_name),dpi=600,format='pdf')
    else:
        plt.savefig(os.path.join(os.getcwd(), options.fig_name))
    print("done")
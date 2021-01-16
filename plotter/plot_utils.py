import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D

"""
log_dir: base_log/env/alg/xxx/seed_i/process.csv
"""

"""
function need to be modified

LEGEND_ORDER
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

pass_alg_name = ['vision(rnd)', 'crl']

LEGEND_ORDER = {
    "surprise_vision_shape_weight": 0,
    "rnd_vision": 1,
    "sac":2,
    "surprise": 3,
    "rnd": 4,
    "src": 5,
}

def get_alg_name(algo_name):
    dict_name = {
        "surprise_vision_shape_weight": "vision(surprise)",
        "rnd_vision": "vision(rnd)",
        "sac": "sac",
        "surprise": "surprise",
        "rnd": "rnd",
        "src": "crl",
    }
    return dict_name[algo_name]

COLORS = dict()
def get_color(label):
    if label not in COLORS.keys():
        COLORS[label] = colors.pop(0)
    return COLORS[label]

def env_name_to_figure_name(env_name):
    dic = {
        "goal4": "AntCorridor-[4,5]",
        "goal5": "AntCorridor-[5,6]",
    }
    return dic[env_name] if env_name in dic else env_name

def get_plot_data_from_single_experiment(file_name, algo_name):
    try:
        data = pd.read_csv(file_name, error_bad_lines=False)
    except pd.errors.EmptyDataError:
        return None, None
    
    dic_y = {
        "surprise_vision_shape_weight": "evaluation/Average Returns",
        "rnd_vision": "evaluation/Average Returns",
        "sac": "evaluation/Average Returns",
        "surprise": "evaluation/Average Returns",
        "rnd": "evaluation/Average Returns",
        "src": "evaluation/Average Returns",
    }

    dic_x = {
        "surprise_vision_shape_weight": "Epoch",
        "rnd_vision": "Epoch",
        "sac": "Epoch",
        "surprise": "Epoch",
        "rnd": "Epoch",
        "src": "Epoch",
    }

    column_y = dic_y[algo_name]
    column_x = dic_x[algo_name]

    data_y = data[column_y]
    data_x = data[column_x]
    return np.array(data_x).astype(np.int), np.array(data_y).astype(np.float) # astype将array的数据类型投影到一个指定type

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
        "goal4": 500,        
        "goal5": 500,
        "goal6": 500,
        "continuous_resource_mountaincar": 500
    }
    return dic[env_name] if env_name in dic else 200000

def data_truncate(x, y_mean, y_std, x_limit):
    # x[x<=limit] 返回所有满足条件的元素构成数组，只有数组才能这么操作
    # 这个功能是截断
    return x[x<=x_limit], y_mean[:len(x[x<=x_limit])], y_std[:len(x[x<=x_limit])]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")
    parser.add_argument("--log_dir", type=str, default=r"/home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/final_result")
    parser.add_argument("--file_name", type=str, default="progress.csv")
    parser.add_argument("--column_x", type=str, default="Epochs")# Time steps
    parser.add_argument("--column_y", type=str, default="Average return")
    parser.add_argument("--alg_config", type=str, default="xx")
    parser.add_argument("--fig_name", type=str, default="comparison_plot.pdf")
    parser.add_argument("--w", type=float, default=18)
    parser.add_argument("--h", type=float, default=10)


    options = parser.parse_args()

    fig = plt.figure(figsize=(options.w,options.h))
    gs = GridSpec(1, 1)
    ax1 = plt.subplot(gs[0, 0:1])
    #ax2 = plt.subplot(gs[0, 2:4])
    #ax3 = plt.subplot(gs[1, 0:2])
    #ax4 = plt.subplot(gs[1, 2:4])

    axs = [ax1]

    i = 0
    legend_line = ()
    legend_alg = ()
    log_dir = options.log_dir
    alg_config = options.alg_config
    for env_name in os.listdir(log_dir):
        env_dir = os.path.join(log_dir, env_name)
        ax = axs[i]
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        sub_figure_name = env_name_to_figure_name(env_name)
        #print("plot: %s" % sub_figure_name)
        x_limit = get_x_limit(env_name)
        ax.set_title(sub_figure_name)
        plot_data = []
        #spec_algo = ("surprise_vision","rnd_vision","sac","surprise","rnd","src")
        for algo_name in os.listdir(env_dir):
        # for algo_name in spec_algo:
            algo_dir = os.path.join(env_dir, algo_name)
            #algo_dir = env_dir
            #print(algo_dir)
            x, y_mean, y_std, max_y = get_data_from_algo_dir(algo_dir, algo_name, x_limit, alg_config)

            alg = get_alg_name(algo_name)
            plot_data.append((LEGEND_ORDER[algo_name], alg, x, y_mean, y_std, max_y))

        for _, algo_name, x, y_mean, y_std, max_y in sorted(plot_data, key=lambda x: x[0]):
            #print("\nName of Algorithm: %s"%algo_name)
            if algo_name in pass_alg_name:
                continue
            if type(x) != type(None):
                x, y_mean, y_std = data_truncate(x, y_mean, y_std, x_limit)
                line, = ax.plot(x, y_mean, label=algo_name, linewidth=LINEWIDTH, color=get_color(algo_name))
                if i == 0:
                    legend_line = legend_line + (line,)
                    legend_alg = legend_alg + (algo_name,)
                    #ax.legend(legend_line, legend_alg, loc='upper left')
                ax.fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2, color=get_color(algo_name))
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
    
    
    plt.tight_layout(pad=4.0, w_pad=1.5, h_pad=3, rect=[0, 0, 1, 1])
    if options.fig_name[-3:] == 'pdf':
        plt.savefig(os.path.join(os.getcwd(), options.fig_name),dpi=600,format='pdf')
    else:
        plt.savefig(os.path.join(os.getcwd(), options.fig_name))
    print("done")
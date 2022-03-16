import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ipdb import set_trace
"""
log_dir: base_log/env/alg/seed_i/process.csv
定死log_dir 的格式
"""
# plt.style.use('seaborn-whitegrid') # plot style setting
import matplotlib
matplotlib.use('Agg')

LINEWIDTH = 4
WINDOWLENGTH = 10
STEPINTERVAL = 1e3
FONTSIZE = 36
SMALLFONTSIZE = 15
MEDIUMFONTSIZE = 28
XLimit = {
    "electric_ant": 1e6,
    "delivery_ant": 1e6
}

colors = ["lightseagreen", "cornflowerblue", "orchid", "gray", "darkseagreen", "goldenrod", "darkorange", "mediumorchid", "darkturquoise" ]
COLORS = {"surprise_vision_alpha2.5": "orangered"}

LEGENDORDER = ["surprise_vision_alpha2.5", "ant_corridor_fuel_done_140_beta1_alpha3fuel", "ant_corridor_fuel_done_140_beta1_alpha2fuel"]

def smooth(y_data):
    # smooth y via moving average
    window_length = WINDOWLENGTH
    smooth_y=[]
    if isinstance(y_data, list):
        for y in y_data:
            y_ext = np.concatenate([np.zeros(window_length),y])
            tmp = [np.mean(y_ext[i:window_length+i]) for i in range(len(y))]
            smooth_y.append(tmp)
    else:
        y = y_data
        y_ext = np.concatenate([np.zeros(window_length),y])
        smooth_y = [np.mean(y_ext[i:window_length+i]) for i in range(len(y))]
    return smooth_y

def get_color(label):
    if label not in COLORS.keys():
        COLORS[label] = colors.pop(0)
    return COLORS[label]

def env_name_to_figure_name(env_name):
    dic = {
        "electric_ant": "Electric Ant",
        "delivery_ant": "Delivery Ant"
    }
    return dic[env_name] if env_name in dic else env_name

def alg_name_to_legend(alg_name):
    dic = {
        "surprise_vision_alpha2.5": r"$\alpha =2.5 I_{max}$",
        "ant_corridor_fuel_done_140_beta1_alpha3fuel": r"$\alpha =3 I_{max}$",
        "ant_corridor_fuel_done_140_beta1_alpha2fuel": r"$\alpha = 2 I_{max}$"
    }
    return dic[alg_name] if alg_name in dic else alg_name

def get_legend_line_label(legend_dict):
    line = []
    label = []
    for alg_name in LEGENDORDER:
        assert alg_name in legend_dict.keys()
        la = alg_name_to_legend(alg_name)
        label.append(la)
        line.append(legend_dict[alg_name])

    return line, label

def compute_mean_std_max(plot_x, plot_y):
    max_len = len(plot_y[0])
    x = plot_x[0]
    for i in range(1, len(plot_y)):
        if len(plot_y[i]) > max_len:
            max_len = len(plot_y[i])
            x = plot_x[i]
    y_mean = []
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

def get_plot_data_from_single_experiment(file_name, algo_name):
    try:
        if algo_name == 'redq':
            data = pd.read_table(file_name)
        else:
            data = pd.read_csv(file_name)
    except pd.errors.EmptyDataError:
        return None, None
    dic = {
        "surprise_vision_alpha2.5": r"$\alpha =2.5 I_{max}$",
        "ant_corridor_fuel_done_140_beta1_alpha3fuel": r"$\alpha =3 I_{max}$",
        "ant_corridor_fuel_done_140_beta1_alpha2fuel": r"$\alpha = 2 I_{max}$"
    }
    dic_y = {
        "surprise_vision_alpha2.5": "evaluation/Average Returns",
        "ant_corridor_fuel_done_140_beta1_alpha3fuel": "evaluation/Average Returns",
        "ant_corridor_fuel_done_140_beta1_alpha2fuel": "evaluation/Average Returns"
    }
            
    dic_x_1 = {
        "surprise_vision_alpha2.5": "exploration/num steps total",
        "ant_corridor_fuel_done_140_beta1_alpha3fuel": "exploration/num steps total",
        "ant_corridor_fuel_done_140_beta1_alpha2fuel": "exploration/num steps total"
    }
    
    dic_x_2 = {
        "surprise_vision_alpha2.5": "Epoch",
        "ant_corridor_fuel_done_140_beta1_alpha3fuel": "Epoch",
        "ant_corridor_fuel_done_140_beta1_alpha2fuel": "Epoch"
    }
    
    column_y = dic_y[algo_name]
    data_y = data[column_y]
    if dic_x_1[algo_name] in data.columns:
        column_x = dic_x_1[algo_name]
    else:
        column_x = dic_x_2[algo_name]
    data_x = data[column_x]
    if dic_x_1[algo_name] not in data.columns:
        data_x = np.array(data_x).astype(np.int)
        data_x = data_x * 1000 + 6000


    # total_step = list(data_x)[-1]
    # interval = total_step / options.n_data
    # _data_x = []
    # _data_y = []
    # last_step = 0
    # for i in range(len(data_x)):
    #     if data_x[i] > last_step-1:
    #         _data_x.append(data_x[i])
    #         _data_y.append(data_y[i])
    #         last_step += interval 
    
    return np.array(data_x).astype(np.int), np.array(data_y).astype(np.float)

def get_data_from_algo_dir(algo_dir, algo_name, x_limit, env_name):
    plot_y = []
    plot_x = []
    for sub_name in os.listdir(algo_dir):
        if ".csv" in sub_name:
            file_name = os.path.join(algo_dir, sub_name)
        else:
            sub_dir = os.path.join(algo_dir, sub_name)
            if algo_name == "redq":
                file_name = os.path.join(sub_dir, "progress.txt")
            else:
                file_name = os.path.join(sub_dir, "progress.csv")
        if os.path.exists(file_name):
            print("Obtaining data from %s"%file_name)
            x, y = get_plot_data_from_single_experiment(file_name, algo_name)
            if type(x) != type(None):
                max_step = x[-1]
                print("max step:%d"%max_step)
                if max_step >= x_limit:
                    index = int(x_limit / STEPINTERVAL)
                    # 截取数据在xlimit 范围内
                    plot_x.append(x[:index])
                    plot_y.append(y[:index])
                    # plot_x.append(x)
                    # plot_y.append(y)

    if plot_x == []:
        return None, None, None, None
    if env_name != "invert_pendulum":
        plot_y = smooth(plot_y)
    x, y_mean, y_std, max_y = compute_mean_std_max(plot_x, plot_y)
    # if algo_name == "sac" and env_name != "invert_pendulum":
    #     print(f"env_name: {env_name}, 100 {y_mean[999]}, 200 {y_mean[1999]}, 300 {y_mean[2999]}, max {max_y}")
    return x, y_mean, y_std, max_y

def run_plot(args):
    # preporcess data dir
    fig = plt.figure(figsize=(args.w,args.h)) # 创建一张图 figure 指定大小
    # gs = GridSpec(6, 9)
    # ax1 = plt.subplot(gs[:3, 6:9]) # 在figure中创建子图 # walker to 5 walker to 3
    # ax2 = plt.subplot(gs[3:, 3:6]) # cheetah to 4 cheetah to 5 
    # ax3 = plt.subplot(gs[:3, 3:6]) # humanoid to 2 
    # ax4 = plt.subplot(gs[3:, :3]) # hopper to 3 hopper to 4
    # ax5 = plt.subplot(gs[3:, 6:9]) # invert to 6
    # ax6 = plt.subplot(gs[:3, :3]) # ant to 1
    gs = GridSpec(3, 3)
    ax1 = plt.subplot(gs[:3, :3])
    axs = [ax1]
    print(os.listdir(args.log_dir))

    for i, env_name in enumerate(os.listdir(args.log_dir)):   
        env_dir = os.path.join(args.log_dir, env_name)
        if env_name == "delivery_ant":
            continue
        if os.path.exists(env_dir):
            x_limit = XLimit[env_name]
            ax = axs[0]
            legend_dict = {}
            for alg_name in os.listdir(env_dir):
                if alg_name == 'mbpo':
                    ## load results 
                    fname = '{}_{}.pkl'.format(env_name, alg_name)
                    print("***********************")
                    print(fname)
                    print("***********************")
                    fname = os.path.join("mbpo_data", fname)
                    data = pickle.load(open(fname, 'rb'))
                    data['x'] = data['x'] * 1000
                    if 'invert' not in env_name:
                        index = int(x_limit / STEPINTERVAL)
                        # 截取数据
                        data['x'] = data['x'][:index]
                        data['y'] = data['y'][:index]
                        data['std'] = data['std'][:index]
                    if env_name != "invert_pendulum":
                        data['y'] = smooth(data['y'])
                    p = ax.plot(data['x'], data['y'], linewidth=LINEWIDTH, color=get_color(alg_name))
                    ax.fill_between(data['x'], data['y'] + data['std'], data['y'] - data['std'], alpha=0.2, color=get_color(alg_name))
                    legend_dict[alg_name] = p[0]
                else:
                    if alg_name == "sac_fuel_12" or alg_name == "sac_no_fuel" or alg_name == "raeb_bad" or alg_name == "surprise_bad":
                        print(alg_name)
                        continue
                    algo_dir = os.path.join(env_dir, alg_name)
                    # load data 
                    print(alg_name)
                    x, y_mean, y_std, max_y = get_data_from_algo_dir(algo_dir, alg_name, x_limit, env_name)
                    # plot data
                    #if alg_name == "sac":
                    #    set_trace()
                    sub_figure_name = env_name_to_figure_name(env_name)
                    ax.set_title(sub_figure_name)
                    ax.tick_params(labelsize=SMALLFONTSIZE)
                    p = ax.plot(x, y_mean, linewidth=LINEWIDTH, color=get_color(alg_name))
                    ax.fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2, color=get_color(alg_name))
                    legend_dict[alg_name] = p[0]
                # else:
                #     continue
                #     # NotImplemented 

            ax.set_title(env_name_to_figure_name(env_name), fontsize=FONTSIZE)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.set_xlabel(args.column_x, fontsize=FONTSIZE)
            ax.set_ylabel(args.column_y, fontsize=FONTSIZE)
            ax.grid()
            if i == 1:
                line, label = get_legend_line_label(legend_dict)
                ax.legend(line, label, loc='upper left', prop={'size': MEDIUMFONTSIZE})
            
    plt.tight_layout(pad=4.0, w_pad=10, h_pad=2, rect=[0, 0, 1, 1])
    plt.savefig(os.path.join(args.save_dir, args.fig_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")

    parser.add_argument("--log_dir", type=str, default="/home/zhwang/research/ICML2021_Finaldata/final_data_211226/sensitivity_analysis/alpha")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--file_name", type=str, default="progress.csv")   
    parser.add_argument("--column_x", type=str, default="Time steps")
    parser.add_argument("--column_y", type=str, default="Average return")
    parser.add_argument("--fig_name", type=str, default="alpha_electric_ant.pdf")
    parser.add_argument("--n_data", type=int, default=100)
    parser.add_argument("--w", type=float, default=12)
    parser.add_argument("--h", type=float, default=9)
    
    options = parser.parse_args()
    run_plot(options)

    ## env_name_list = os.listdir(log_dir)
    ## alg_name_list = os.listdir(下一层log_dir)

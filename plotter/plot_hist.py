import matplotlib.pyplot as plt
import random
import matplotlib
import os
matplotlib.rc('font', family='SimHei', weight='bold')

# city_name = ['python脚本', 'Hive数据处理框架']
city_name = ['MIRA_MCTS', 'LeelaZero']
# city_name.reverse()

# data = [22.47, 4.28]
data = [3.5, 0.5]
data.reverse()

 
# colors = ['red', 'yellow', 'blue', 'green', 'gray']
colors = ['#0070C0', '#FF7B06']
# colors.reverse()

plt.figure(figsize=(10,2))
b = plt.barh(range(len(data)), data, tick_label=city_name, color=colors)
for rect in b:
    w = rect.get_width()
    plt.text(w, rect.get_y()+rect.get_height()/2, '%0.1f' %w, ha='left', va='center')
plt.xticks(())
# plt.title('每分钟平均处理局数')
#plt.title('每天仿真对战局数(万)')
plt.savefig(os.path.join(os.getcwd(), "search.png"))
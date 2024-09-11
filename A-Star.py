import numpy as np
# queue.PriorityQueue 是 Python 标准库 queue 模块中的一个类，提供了线程安全的优先队列实现
#在 PriorityQueue 中，元素按照其优先级排序，具有最小优先级的元素先被取出
from queue import PriorityQueue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

#创建画布
global ax
fig, ax = plt.subplots()

class Map:
    #创建迷宫地图对象初始化
    def __init__(self, length: int, width: int) -> None:
        #迷宫的尺寸
        self.length = length
        self.width = width
        #创建一个包含 length * width 个空列表的列表，用于表示每个点的邻接表
        self.neighbor = []
        for i in range(length * width):
            self.neighbor.append([])
    
    #向迷宫的邻接表中添加边，表示两个点之间的连接关系
    def add_edge(self, from_: int, to_: int):
        if (from_ not in range(self.length * self.width)) or (to_ not in range(self.length * self.width)):
            return 0
        #由于是无向图，因此将 to_ 添加到 from_ 的邻接表中,并将 from_ 添加到 to_ 的邻接表中
        self.neighbor[from_].append(to_)
        self.neighbor[to_].append(from_)
        return 1
    
    #计算迷宫地图中第 num 个节点从0开始的笛卡尔直角坐标
    def get_coordinate(self, num: int):
        if num not in range(self.length * self.width):
            return -1, -1
        x = num % self.length+1
        y = num // self.length+1
        return x, y
    
    #绘制节点
    def draw_node(self, num, color):
        x, y = self.get_coordinate(num)
        node = patches.Circle(np.array([x, y]), 0.1, color = color)
        # ax.add_patch 是 Matplotlib 中 Axes 对象的一个方法，用于将图形对象添加到坐标轴上
        ax.add_patch(node)
    
    #绘制迷宫地图的边
    def draw_edge(self, from_: int, to_: int, color):
        x1, y1 = self.get_coordinate(from_)
        x2, y2 = self.get_coordinate(to_)
        #由于边（本质是矩形）需要一定的宽度，设定一个 offset 帮助清楚描绘边的形状
        offset = 0.05
        # ← 方向的边
        if from_-to_ == 1:
            rectangle_left = patches.Rectangle(np.array([x2-offset, y2-offset]), 1+2*offset, 2*offset, color=color)
            ax.add_patch(rectangle_left)
        # → 方向的边
        elif from_-to_ == -1:
            rectangle_right = patches.Rectangle(np.array([x1-offset, y1-offset]), 1+2*offset, 2*offset, color=color)
            ax.add_patch(rectangle_right)
        # ↑ 方向的边
        elif from_-to_ == self.width:
            rectangle_up = patches.Rectangle(np.array([x2-offset, y2-offset]), 2*offset, 1+2*offset, color=color)
            ax.add_patch(rectangle_up)
        # ↓ 方向的边   
        else:
            rectangle_down = patches.Rectangle(np.array([x1-offset, y1-offset]), 2*offset, 1+2*offset, color=color)
            ax.add_patch(rectangle_down)

    #绘制迷宫
    def initMap(self):
        #绘制边
        for i in range(self.length*self.width):
            for next in self.neighbor[i]:
                self.draw_edge(i, next, '#FAF9DE')
        #绘制节点
        for i in range(self.length*self.width):
            self.draw_node(i, '#FFEC8B')

    #寻找
    def search(self, current: int):
        #四个方向的顺序
        sequence = [i for i in range(4)]
        # 打乱顺序
        random.shuffle(sequence)
        #依次选择四个方向
        for i in sequence:
            #要探索的位置
            x = self.direction[i]+current
            #跨了一行
            if (current % self.length == self.length-1 and self.direction[i] == 1) or (current % self.length == 0 and self.direction[i] == -1):
                continue
            #要探索的位置没有超出范围且该位置没有被探索过
            if 0 <= x < self.length*self.width and self.visited[x] == 0:
                self.add_edge(current, x)
                self.visited[x] = 1
                self.search(x)

    #随机生成迷宫
    def randomCreateMap(self, start, k):
        #标识每个节点是否被探索过
        self.visited = np.zeros(self.length*self.width)
        self.visited[start] = 1
        #四个方向,分别代表上、下、左、右
        self.direction = {0: -self.length,
                          1: self.length,
                          2: -1,
                          3: 1}
        # 从起点开始
        self.search(start)
    
    #随机添加k条边
    def randomAddEdges(self, k):
        # 循环k次(可能不止k次)
        for i in range(k):
            node = random.randint(0, self.length*self.width)
            # 随机添加一个方向
            sequence = [i for i in range(4)]
            random.shuffle(sequence)
            isPick = 0
            for d in sequence:
                # 跨了一行,不存在该方向的边
                if (node % self.length == self.length-1 and self.direction[d] == 1) or (node % self.length == 0 and self.direction[d] == -1):
                    continue
                x = self.direction[d]+node
                # 该边存在
                if x in self.neighbor[node]:
                    continue
                # 该边不存在
                self.add_edge(node, x)
                isPick = 1
            # 重新添加一条边,即重新循环一次
            if isPick == 0:
                if i == 0:  # 第一次
                    i = 0
                else:
                    i -= 1
 

class AStar:

    #初始化 A* 算法
    def __init__(self, map: Map, start: int, end: int) -> None:
        self.map = map
        self.start = start
        self.end = end
        #每个节点的优先级，即放入 open 表的顺序,数据类型为元组，其形式为 (priority, node)
        self.open_set = PriorityQueue()
        #每个节点的代价，即从起点到当前路径的代价(key:node, value:cost)
        self.closed_set = dict()
        #每个节点的前驱节点(key:node, value:previous node)
        self.predecessor = dict()
        #将起点放入,优先级设为0，无所谓设置多少，因为总是第一个被取出
        self.open_set.put((0, start))
        self.predecessor[start] = -1
        self.closed_set[start] = 0
        #标记起点和终点
        self.map.draw_node(start, '#00BFFF')
        self.map.draw_node(end, '#FF8C00')

    # h 函数计算,即启发式信息
    def heuristic_function(self, a, b):
        x1, y1 = self.map.get_coordinate(a)
        x2, y2 = self.map.get_coordinate(b)
        #返回两点 Manhattan Distance
        return abs(x1-x2) + abs(y1-y2)
    
    #运行 A* 寻路算法，如果没找到路径返回0，找到返回1
    def find_way(self):
        while not self.open_set.empty():
            # current 是一个元组，其形式为 (priority, node)
            current = self.open_set.get()
            # current[0] 表示节点的优先级（代价加上启发式估计），而 current[1] 则表示实际的图节点
            if current[1] != self.start:
                #当前节点的前序
                pre = self.predecessor[current[1]]
                #可视化
                self.map.draw_edge(pre, current[1], '#87CEFA')
                self.map.draw_node(pre, '#00BFFF')
                if current[1] != self.end:
                    self.map.draw_node(current[1], '#00BFFF')
                else:
                    self.map.draw_node(current[1], '#FF8C00')
                plt.show()
                plt.pause(0.01)
            if current[1] == self.end:
                break
            for next in self.map.neighbor[current[1]]:
                new_cost = self.closed_set[current[1]] + 1
                if (next not in self.closed_set) or (new_cost < self.closed_set[next]):
                    self.closed_set[next] = new_cost
                    priority = new_cost+self.heuristic_function(next, self.end)
                    self.open_set.put((priority, next))
                    self.predecessor[next] = current[1]
        if self.end not in self.closed_set:
            return 0
        return 1
    
    #寻找并打印路径
    def show_way(self):
        result = []
        current = self.end
        #到起点的前一个节点结束循环
        while self.predecessor[current] != -1:
            result.append(current)
            current = self.predecessor[current]
        result.append(self.start)
        # result 列表为[终点......起点]，需要颠倒列表
        result.reverse()
        for point in result:
            if point != self.start:
                #当前节点的前序
                pre = self.predecessor[point]
                #可视化
                self.map.draw_edge(pre, point, '#FFA54F')
                if pre == self.start:
                    self.map.draw_node(pre, '#FF8C00')
                elif point == self.end:
                    self.map.draw_node(point, '#FF8C00')
                #显示当前状态
                plt.show()
                plt.pause(0.01)
    
    #返回最短路径的长度
    def print_cost(self):
        if self.end not in self.closed_set:
            return -1
        else:
            return self.closed_set[self.end]
print("请自行定义迷宫的长和宽：")        
length, width = map(int, input().split())
theMap = Map(length, width)
#设置迷宫显示的一些参数
plt.xlim(0, theMap.length+1)
plt.ylim(0, theMap.width+1)
#等距
plt.axis('equal')
#不显示背景的网格线
plt.grid(False)
#允许动态
plt.ion()
# 随机添加边，生成迷宫，第一个参数为起点；第二个参数为额外随机生成的边，可以表示为图的复杂程度
theMap.randomCreateMap(0, 50)
#初始化迷宫
theMap.initMap()
# A* 算法寻路
theAStar = AStar(theMap, 0, length*width-1)
theAStar.find_way()
theAStar.show_way()
theCost = theAStar.print_cost()
#打印最短路径的长度
if theCost == -1:
    print("无法到达终点")
else:
    print("从起点到终点的最短路径长度为: ", theCost)
#关闭交互，展示结果
plt.ioff()
plt.show()
from PyQt5.Qt import *
from RL_brain import QLearningTable
import math
import random
import time

UNIT = 70   # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width

class PaintArea(QWidget):#画图类
    def __init__(self,parent = None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setPalette(QPalette(QColor(240,240,240)))#设置背景颜色
        self.setAutoFillBackground(True)#设置窗口自动填充背景
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.loc_x = 0
        self.loc_y = 0
        self.pix = QPixmap()  # 实例化一个 QPixmap 对象

    # 重绘的复写函数 主要在这里绘制
    def paintEvent(self, event):
        qp = QPainter(self.pix)
        qp = QPainter()
        qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        # 设置颜色和画笔
        qp.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        qp.setBrush(QBrush(QColor(200, 200, 200)))
        # 绘制4x4的方格
        for i in range(0, MAZE_W):
            for j in range(0, MAZE_H):
                qp.drawRect(i * UNIT, j * UNIT, UNIT - 1, UNIT - 1)
        # 设置颜色和画笔
        qp.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        qp.setBrush(QBrush(QColor(0, 0, 0)))
        # 两个障碍物,坐标分别为（2,1）和（1,2）
        qp.drawRect(2 * UNIT, UNIT, UNIT - 1, UNIT - 1)
        qp.drawRect(UNIT, 2 * UNIT, UNIT - 1, UNIT - 1)
        # 设置颜色和画笔
        qp.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        qp.setBrush(QBrush(QColor(255, 0, 0)))
        # 迷宫出口，坐标为（2,2）
        qp.drawRect(2 * UNIT, 2 * UNIT, UNIT - 1, UNIT - 1)
        # 设置颜色和画笔
        qp.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        qp.setBrush(QBrush(QColor(255, 255, 0)))
        # 绘制当前位置
        qp.drawRect(self.loc_x, self.loc_y, UNIT - 1, UNIT - 1)

        qp.end()

    def reset(self):
        self.loc_x = 0
        self.loc_y = 0
        self.update()
        # return observation
        return [self.loc_x,self.loc_y]

    def step(self, action):
        # up
        if (action == 0):
            if (self.loc_y >= UNIT):
                self.loc_y = self.loc_y - UNIT
        # down
        if (action == 1):
            if (self.loc_y < (MAZE_H - 1)*UNIT):
                self.loc_y = self.loc_y + UNIT
        # left
        if (action == 2):
            if (self.loc_x >= UNIT):
                self.loc_x = self.loc_x - UNIT
        # right
        if (action == 3):
            if (self.loc_x < (MAZE_W - 1)*UNIT):
                self.loc_x = self.loc_x + UNIT

        s_ = [self.loc_x,self.loc_y]
        # reward function
        if ((self.loc_x == 2*UNIT) and (self.loc_y == 2*UNIT)):
            reward = 1
            done = True
            s_ = 'terminal'
        elif (((self.loc_x == UNIT) and (self.loc_y == 2*UNIT)) or ((self.loc_x == 2*UNIT) and (self.loc_y == UNIT))):
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False
        return s_, reward, done

    def draw(self, action):
        self.loc_x += UNIT

#窗口类
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyqt5画图类")
        self.resize(350,350)
        self.setup_ui()
        self.temp_shuzu = [0]*10000
        self.index = 0

        #初始化先得到一组解
        self.update()

        #定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.get_wind)
        self.timer.start(100)

    def setup_ui(self):
        #子控件加这里
        self.area = PaintArea(self)
        self.area.resize(350,350)
        self.area.move(0,0)
        self.RL = QLearningTable(actions=list(range(self.area.n_actions)))

    def get_wind(self):
        num = self.temp_shuzu[self.index]
        t1_,t2_,t3_ = self.area.step(num)
        self.index += 1
        self.area.update()

        if (num == 4):
            self.update()
            self.index = 0

    def update(self):
        observation = self.area.reset()
        list_index = 0
        while True:
            # RL choose action based on observation
            action = self.RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = self.area.step(action)

            self.temp_shuzu[list_index] = action
            list_index += 1

            # RL learn from this transition
            self.RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                self.temp_shuzu[list_index] = 4
                self.area.loc_y = 0
                self.area.loc_x = 0
                break


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


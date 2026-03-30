import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

# --- 1. 基础配置与参数 ---
res = 700
aspect_ratio = 1.0
zNear = 0.1
zFar = 50.0
eye_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)

# --- 2. 定义数据容器 (Fields) ---
# 原始 3D 顶点
vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
vertices.from_numpy(np.array([[2.0, 0.0, -2.0], [0.0, 2.0, -2.0], [-2.0, 0.0, -2.0]], dtype=np.float32))

# 变换后的 2D 屏幕坐标
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=3)

# 为三条不同颜色的边准备三个小 field (每条边 2 个点)
line_red = ti.Vector.field(2, dtype=ti.f32, shape=2)
line_blue = ti.Vector.field(2, dtype=ti.f32, shape=2)
line_green = ti.Vector.field(2, dtype=ti.f32, shape=2)

# --- 3. 矩阵生成函数 (Python 逻辑) ---

def get_model_matrix(angle):
    theta = angle * math.pi / 180.0
    c, s = math.cos(theta), math.sin(theta)
    return ti.Matrix([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def get_view_matrix(eye_pos):
    return ti.Matrix([[1, 0, 0, -eye_pos[0]], [0, 1, 0, -eye_pos[1]], [0, 0, 1, -eye_pos[2]], [0, 0, 0, 1]])

def get_projection_matrix(fov, aspect_ratio, zNear, zFar):
    # 实验要求的透视投影推导
    t = math.tan(fov * math.pi / 360.0) * zNear
    r = aspect_ratio * t
    n, f = -zNear, -zFar
    return ti.Matrix([
        [n/r, 0, 0, 0],
        [0, n/t, 0, 0],
        [0, 0, (n+f)/(n-f), -2*n*f/(n-f)],
        [0, 0, 1, 0]
    ])

# --- 4. Taichi 计算内核 ---

@ti.kernel
def compute_transform(M: ti.template(), V: ti.template(), P: ti.template()):
    mvp = P @ V @ M   #顺序很重要，先放进世界坐标系，再放进看的坐标系，再放进投影坐标系
    for i in range(3):
        v4 = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        v_clip = mvp @ v4
        # 透视除法：NDC 坐标 [-1, 1]
        v_ndc = v_clip.xyz / v_clip.w
        # 视口变换：映射到屏幕 [0, 1]
        screen_coords[i] = (v_ndc.xy + 1.0) * 0.5

@ti.kernel
def prepare_render_data():
    # 红色边: v0 -> v1
    line_red[0], line_red[1] = screen_coords[0], screen_coords[1]
    # 蓝色边: v1 -> v2
    line_blue[0], line_blue[1] = screen_coords[1], screen_coords[2]
    # 绿色边: v2 -> v0
    line_green[0], line_green[1] = screen_coords[2], screen_coords[0]

# --- 5. 渲染与交互主循环 ---

window = ti.ui.Window("3D Transformation Lab", (res, res))
canvas = window.get_canvas()
angle = 0.0

while window.running:
    # 键盘监听
    if window.is_pressed('a'): angle += 2.0
    if window.is_pressed('d'): angle -= 2.0
    if window.is_pressed(ti.ui.ESCAPE): break

    # 1. 计算当前帧的矩阵
    M = get_model_matrix(angle)
    V = get_view_matrix(eye_pos)
    P = get_projection_matrix(45.0, aspect_ratio, zNear, zFar)

    # 2. 运行内核更新坐标
    compute_transform(M, V, P)
    prepare_render_data()

    # 3. 绘制
    canvas.set_background_color((0, 0, 0))
    # 分三次调用 lines 实现三色效果
    canvas.lines(line_red, width=0.005, color=(1, 0, 0))   # 红
    canvas.lines(line_blue, width=0.005, color=(0, 0, 1))  # 蓝
    canvas.lines(line_green, width=0.005, color=(0, 1, 0)) # 绿
    
    window.show()
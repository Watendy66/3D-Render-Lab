import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

# --- 1. 配置与数据 ---
res = 700
aspect_ratio = 1.0
eye_pos = np.array([0.0, 0.0, 6.0], dtype=np.float32)

# 立方体 8 个顶点
cube_v_np = np.array([
    [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1], # 前 0,1,2,3
    [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1]  # 后 4,5,6,7
], dtype=np.float32)

# 12 个三角形（6 个面，每个面 2 个三角形）a
# 每个三角形 3 个顶点索引
cube_t_np = np.array([
    [0,1,2], [0,2,3], # 前
    [5,4,7], [5,7,6], # 后
    [4,0,3], [4,3,7], # 左
    [1,5,6], [1,6,2], # 右
    [3,2,6], [3,6,7], # 上
    [4,5,1], [4,1,0]  # 下
], dtype=np.int32).flatten()

vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
vertices.from_numpy(cube_v_np)

face_indices = ti.field(dtype=ti.i32, shape=36)
face_indices.from_numpy(cube_t_np)

# 变换后的坐标和最终渲染用的三角形顶点
screen_points = ti.Vector.field(2, dtype=ti.f32, shape=8)
render_triangles = ti.Vector.field(2, dtype=ti.f32, shape=36)

# 为 12 个三角形准备颜色 (每 2 个三角形颜色相近，代表一个面)
# 这里使用 ti.Vector.field(3, ...) 存储 RGB
triangle_colors = ti.Vector.field(3, dtype=ti.f32, shape=36)

# --- 2. 矩阵与工具函数 ---
def get_mvp(angle_x, angle_y):
    rx, ry = angle_x * math.pi / 180.0, angle_y * math.pi / 180.0
    # 绕 X 轴旋转
    Mx = ti.Matrix([[1, 0, 0, 0], [0, math.cos(rx), -math.sin(rx), 0], [0, math.sin(rx), math.cos(rx), 0], [0, 0, 0, 1]])
    # 绕 Y 轴旋转
    My = ti.Matrix([[math.cos(ry), 0, math.sin(ry), 0], [0, 1, 0, 0], [-math.sin(ry), 0, math.cos(ry), 0], [0, 0, 0, 1]])
    M = My @ Mx
    V = ti.Matrix([[1, 0, 0, -eye_pos[0]], [0, 1, 0, -eye_pos[1]], [0, 0, 1, -eye_pos[2]], [0, 0, 0, 1]])
    
    fov, n, f = 45.0, -0.1, -50.0
    t = math.tan(fov * math.pi / 360.0) * abs(n)
    r = aspect_ratio * t
    P = ti.Matrix([[n/r, 0, 0, 0], [0, n/t, 0, 0], [0, 0, (n+f)/(n-f), -2*n*f/(n-f)], [0, 0, 1, 0]])
    return P @ V @ M

# --- 3. 计算内核 ---
@ti.kernel
def compute_transform(mvp: ti.types.matrix(4, 4, ti.f32)):
    for i in range(8):
        v4 = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip.xyz / v_clip.w
        screen_points[i] = (v_ndc.xy + 1.0) * 0.5

@ti.kernel
def update_render_data():
    for i in range(36):
        v_idx = face_indices[i]
        render_triangles[i] = screen_points[v_idx]
        # 设置颜色：每 6 个顶点（一个面）给一个略有不同的蓝色
        face_id = i // 6
        brightness = 0.4 + face_id * 0.1
        triangle_colors[i] = ti.Vector([0.2 * brightness, 0.4 * brightness, 0.8 * brightness])

# --- 4. 主循环 ---
window = ti.ui.Window("3D Cube Face Render", (res, res))
canvas = window.get_canvas()
ang_x, ang_y = 30.0, 45.0

while window.running:
    # 键盘交互
    if window.is_pressed('w'): ang_x += 2.0
    if window.is_pressed('s'): ang_x -= 2.0
    if window.is_pressed('a'): ang_y -= 2.0
    if window.is_pressed('d'): ang_y += 2.0

    mvp = get_mvp(ang_x, ang_y)
    compute_transform(mvp)
    update_render_data()

    canvas.set_background_color((0.05, 0.05, 0.05))
    canvas.triangles(render_triangles, color=(0.4, 0.6, 0.9), per_vertex_color=triangle_colors)
    window.show()
import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from skimage import color
import json
import os
from scipy.optimize import minimize
import itertools
import base64
from io import BytesIO
import hashlib

# -----------------------------
# 初始化 session_state
# -----------------------------
if "user_colors" not in st.session_state or not isinstance(st.session_state.get("user_colors"), dict):
    st.session_state.user_colors = {}
if "active_colors" not in st.session_state or not isinstance(st.session_state.get("active_colors"), dict):
    st.session_state.active_colors = {}
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "canvas_image" not in st.session_state:
    st.session_state.canvas_image = None
if "image_hash" not in st.session_state:
    st.session_state.image_hash = None

# -----------------------------
# 载入官方油画颜料色库
# -----------------------------
data_path = os.path.join(os.path.dirname(__file__), "paint_colors.json")
with open(data_path, "r", encoding="utf-8") as f:
    paint_colors = json.load(f)

# -----------------------------
# 辅助函数
# -----------------------------
def get_image_hash(file_content):
    """生成文件内容的哈希值"""
    return hashlib.md5(file_content).hexdigest()

def create_canvas_image(img, canvas_width):
    """创建用于画布的图片"""
    canvas_height = int(img.height * canvas_width / img.width)
    max_height = 1500
    if canvas_height > max_height:
        canvas_height = max_height
        canvas_width = int(img.width * canvas_height / img.height)
    return img.resize((canvas_width, canvas_height)), canvas_width, canvas_height

@st.cache_data
def process_uploaded_image(file_content, canvas_width, file_hash):
    """处理上传的图片，返回原图和调整后的画布图片"""
    img = Image.open(BytesIO(file_content)).convert("RGB")
    
    # 计算画布尺寸
    canvas_height = int(img.height * canvas_width / img.width)
    max_height = 1500
    if canvas_height > max_height:
        canvas_height = max_height
        canvas_width = int(img.width * canvas_height / img.height)
    
    # 创建画布用的调整后图片
    canvas_img = img.resize((canvas_width, canvas_height))
    
    return img, canvas_img, canvas_width, canvas_height

# -----------------------------
# 页面布局
# -----------------------------
# title部分
st.set_page_config(page_title="油画调色工坊", layout="wide")
# 高级渐变 CSS
st.markdown(
    """
    <style>
    .header-container {
        background: linear-gradient(135deg, #667eea, #764ba2, #ff9a9e);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
    }

    /* 在背景上加一个半透明光效 */
    .header-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at center, rgba(255,255,255,0.15), transparent 70%);
        transform: rotate(25deg);
    }

    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
        position: relative;
        z-index: 1;
    }

    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        margin-top: 0.8rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }

    .header-divider {
        width: 70px;
        height: 3px;
        background-color: rgba(255,255,255,0.85);
        margin: 1rem auto;
        border-radius: 3px;
        position: relative;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="header-container">
        <div class="header-title">油画调色工坊</div>
        <div class="header-divider"></div>
        <div class="header-subtitle">上传图片 · 点击取色 · 获取颜料配比</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ----------------------------------------------------------------------------------
# 我的调色盘区
# st.header("🖌️ 我的调色盘")

# st.markdown(
#     f"<span style='font-size:15px;color:#888;'>当前调色盘颜色数量：<b>{len(st.session_state.active_colors)}</b></span>",
#     unsafe_allow_html=True
# )
st.markdown(
    f"""
    <div style="
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 16px solid #764ba2;
    ">
        <h3 style="margin:0; color:#333;">🖌️ 我的调色盘</h3>
        <p style="margin:0; font-size:14px; color:#666;">
            当前调色盘颜色数量：<b>{len(st.session_state.active_colors)}</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# 保存/加载/清空
btn_cols = st.columns([1, 1, 1, 10])
with btn_cols[0]:
    if st.button("💾 保存", help="保存当前调色盘"):
        with open("my_palette.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.active_colors, f, ensure_ascii=False, indent=2)
        st.success("已保存调色盘到 my_palette.json")
with btn_cols[1]:
    if st.button("📂 加载", help="从文件加载调色盘"):
        if os.path.exists("my_palette.json"):
            with open("my_palette.json", "r", encoding="utf-8") as f:
                loaded = json.load(f)
            st.session_state.active_colors = loaded
            st.session_state.user_colors.update(loaded)
            st.experimental_rerun()
        else:
            st.warning("my_palette.json 文件不存在")
with btn_cols[2]:
    if st.button("🧹 清空", help="清空当前调色盘"):
        st.session_state.active_colors = {}
        st.experimental_rerun()

# 显示缩略调色盘，可点击删除
if st.session_state.active_colors:
    color_cols = st.columns(16)
    keys = list(st.session_state.active_colors.keys())
    for i, name in enumerate(keys):
        rgb = st.session_state.active_colors[name]
        with color_cols[i % 16]:
            st.markdown(
                f"""
                <div style='position:relative; display:inline-block; margin:0; width:54px;'>
                    <div style='width:50px;height:50px;border-radius:4px;background:rgb{tuple(rgb)};border:1px solid #aaa;'></div>
                    <div style='width:74px;text-align:left;font-size:14px;margin-top:2px;line-height:1.2;white-space:normal;overflow:visible;'>{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            btn_clicked = st.button("×", key=f"del_{name}", help="删除该色块")
            if btn_clicked:
                del st.session_state.active_colors[name]
                st.experimental_rerun()
else:
    st.write("当前色库为空")

st.markdown("---")

# ----------------------------------------------------------------------------------
# st.header("📤 上传图片")
st.markdown(
    f"""
    <div style="
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 16px solid #764ba2;
    ">
        <h3 style="margin:0; color:#333;">📤 上传图片</h3>
        <p style="margin:0; font-size:14px; color:#666;">
            支持 PNG/JPG/JPEG 格式，点击下方选择图片
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 读取文件内容
    file_content = uploaded_file.read()
    current_hash = get_image_hash(file_content)
    
    # 显示原始图片信息
    uploaded_file.seek(0)
    temp_img = Image.open(uploaded_file).convert("RGBA")  # 转 RGBA 保证兼容性
    st.write(f"图片尺寸：{temp_img.width} × {temp_img.height} 像素")

    # 使用稳定的key避免slider变化导致的问题
    canvas_width = st.slider(
        "取色画布宽度", 
        min_value=200, 
        max_value=1500, 
        value=min(600, temp_img.width),
        key="canvas_width_slider"
    )
    
    # 使用缓存函数处理图片，传入canvas_width作为缓存参数
    img, canvas_img, actual_width, canvas_height = process_uploaded_image(file_content, canvas_width, current_hash)
    
    # 强制保证 PIL.Image 格式，并限制大小
    if not isinstance(canvas_img, Image.Image):
        canvas_img = Image.fromarray(canvas_img)
    canvas_img = canvas_img.convert("RGBA")
    canvas_img.thumbnail((1500, 1500))  # 避免过大导致前端崩溃
    
    st.subheader("🎯 取色画布")
    
    # 使用包含尺寸信息的key，确保slider变化时canvas正确更新
    canvas_key = f"canvas_{current_hash[:8]}_{actual_width}_{canvas_height}"
    
     # 添加重置画布按钮
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("🔄", help="重置画布显示", key="reset_canvas"):
            # 清理相关缓存
            process_uploaded_image.clear()
            st.experimental_rerun()
    with col2:
        st.markdown("💡如果画布显示异常，可点击左侧的重置按钮")


    # Canvas 设置背景图片（用 PIL.Image）
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=canvas_img,  # 这里保持 PIL 对象，不用 np.array
        update_streamlit=True,
        height=canvas_img.height,
        width=canvas_img.width,
        drawing_mode="point",
        key=canvas_key,
    )
    
# -----------------------------------------------------------------------------------
    # ============ 新的函数：Lab空间取色 ============
    def get_avg_rgb_lab(img_array, x, y, radius=10):
        """
        在 Lab 空间取邻域平均，再转回 RGB
        img_array: numpy数组 (H, W, 3)，RGB格式 [0,255]
        x, y: 中心点坐标
        radius: 邻域半径，1表示3x3区域
        """
        h, w, _ = img_array.shape
        x_min, x_max = max(0, x-radius), min(w, x+radius+1)
        y_min, y_max = max(0, y-radius), min(h, y+radius+1)
        patch = img_array[y_min:y_max, x_min:x_max, :]

        # 转到 [0,1] 再转 Lab
        patch_lab = color.rgb2lab(patch / 255.0)
        mean_lab = patch_lab.mean(axis=(0, 1))

        # 再转回 RGB
        mean_rgb = (color.lab2rgb(mean_lab[np.newaxis, np.newaxis, :])[0, 0] * 255).astype(int)
        return np.clip(mean_rgb, 0, 255)

    def rgb_to_hex(rgb):
        """RGB 转 HEX"""
        return "#{:02x}{:02x}{:02x}".format(*rgb)

# -------------------------计算混色建议------------------------------------------
    def suggest_mix(target_rgb, palette_source, paint_colors=None, max_candidates=6):
        """
        给定目标 RGB 值和一个颜料调色盘（字典或 my_palette.json 路径），返回候选颜料名称与对应的混合权重。

        输入:
          - target_rgb: 可迭代对象，目标颜色的 RGB 值，例如 [255, 128, 0]
          - palette_source: 字典 {name: [r,g,b], ...} 或者指向 my_palette.json 的文件路径字符串。
          - paint_colors: 可选，完整的颜料色库（当 palette_source 为空时作为后备）。
          - max_candidates: 从调色盘中选取最接近的候选颜色数（默认 6）。

        输出:
          - top_colors: 列表，形如 [(name, [r,g,b]), ...]（按顺序为权重对应顺序）
          - weights: numpy 数组，对应于 top_colors 的比例（归一化和过滤掉很小的权重）

        注: 该函数是独立且自包含的，内部使用Lab空间作为色差度量，在CMY空间进行线性混色模拟，并尝试 1~4 色的线性混合优化。
        """
        # 规范化并加载 palette
        if isinstance(palette_source, str):
            try:
                if os.path.exists(palette_source):
                    with open(palette_source, 'r', encoding='utf-8') as f:
                        palette = json.load(f)
                else:
                    palette = {}
            except Exception:
                palette = {}
        elif isinstance(palette_source, dict):
            palette = palette_source
        else:
            palette = {}

        if not palette:
            palette = paint_colors or {}

        # 保证 palette 是 dict
        if not isinstance(palette, dict):
            palette = {}

        # 计算 Lab 色差的辅助函数（在函数内部自包含，方便迁移）
        def delta_e_local(rgb1, rgb2):
            lab1 = color.rgb2lab(np.array([[rgb1]])/255.0)[0, 0]
            lab2 = color.rgb2lab(np.array([[rgb2]])/255.0)[0, 0]
            return np.linalg.norm(lab1 - lab2)

        # 选择最接近的候选颜色
        try:
            sorted_items = sorted(palette.items(), key=lambda item: delta_e_local(target_rgb, item[1]))
        except Exception:
            sorted_items = list(palette.items())

        candidate_colors = sorted_items[:max_candidates]

        def rgb_to_cmy_local(rgb):
            return 1 - np.array(rgb) / 255.0

        def cmy_to_rgb_local(cmy):
            return np.clip((1 - cmy) * 255, 0, 255).astype(int)

        # 单色优先检查
        best_loss = 1e9
        best_colors = None
        best_weights = None

        for name, rgb_paint in candidate_colors:
            loss = delta_e_local(target_rgb, rgb_paint)
            if loss < best_loss:
                best_loss = loss
                best_colors = [(name, rgb_paint)]
                best_weights = np.array([1.0])

        # 如果单色已经足够接近则直接返回（阈值可调）
        if best_loss < 3:
            return best_colors, best_weights

        rng = np.random.default_rng(42)

        # 尝试 2~4 色组合的线性混合（CMY 空间混合）
        for n in range(2, 5):
            for comb in itertools.combinations(candidate_colors, n):
                palette_cmy = np.array([rgb_to_cmy_local(c[1]) for c in comb])

                def loss(w):
                    mixed_cmy = np.dot(w, palette_cmy) # 线性混合模拟颜料混色 CMYmix​=w1​⋅CMY1​+w2​⋅CMY2​+...+wn​⋅CMYn​
                    mixed_rgb = cmy_to_rgb_local(mixed_cmy)
                    lab1 = color.rgb2lab(np.array([[target_rgb]])/255.0)[0, 0]
                    lab2 = color.rgb2lab(np.array([[mixed_rgb]])/255.0)[0, 0]
                    return np.linalg.norm(lab1 - lab2)

                N = len(comb)
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
                bounds = [(0, 1)] * N

                for _ in range(6): # 多次（6次）随机初始化以避免局部最优
                    w0 = rng.random(N)
                    w0 /= w0.sum()
                    try:
                        res = minimize(loss, w0, bounds=bounds, constraints=cons, method='SLSQP')
                    except Exception:
                        continue
                    if res.success and res.fun < best_loss:
                        best_loss = res.fun
                        best_weights = res.x
                        best_colors = comb

            if best_loss < 2:
                break

        # 如果没有找到（极少情况），回退到最接近的单色
        if best_colors is None or best_weights is None:
            return [(sorted_items[0][0], sorted_items[0][1])], np.array([1.0])

        # 过滤掉极小权重并返回
        filtered = [(c, w) for c, w in zip(best_colors, best_weights) if w > 0.01]
        if filtered:
            top_colors, weights = zip(*filtered)
            return list(top_colors), np.array(weights)
        else:
            return list(best_colors), np.array(best_weights)
#------------------------------------------------------------------
    # 顶层 RGB<->CMY 辅助函数（供展示和理论混合使用）
    def rgb_to_cmy(rgb):
        """将 RGB(0-255) 转到 CMY（0-1）"""
        return 1 - np.array(rgb) / 255.0

    def cmy_to_rgb(cmy):
        """将 CMY（0-1）转换回 RGB(0-255) 整数数组"""
        return np.clip((1 - cmy) * 255, 0, 255).astype(int)

    st.markdown("<div style='color:#fa8c16;font-size:16px;margin:8px 0 0 0;'><b>提示：</b>点击画布任意位置即可取色</div>", unsafe_allow_html=True)

    st.header("🎯 取色结果")
    if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        if objects:
            # 获取最后一个点击点的坐标
            last_point = objects[-1]
            x, y = round(last_point["left"]), round(last_point["top"])
            
            # 将画布坐标转换为原图坐标
            x_img = round(x * img.width / actual_width)
            y_img = round(y * img.height / canvas_height)
            
            # 确保坐标在图片范围内
            x_img = max(0, min(img.width - 1, x_img))
            y_img = max(0, min(img.height - 1, y_img))
            
            img_array = np.array(img)
            # 使用 Lab 空间取色
            rgb = get_avg_rgb_lab(img_array, x_img, y_img, radius=1)
            hex_color = rgb_to_hex(rgb)     
            
            # 显示取色结果
            color_col1, color_col2 = st.columns([1, 20])
            with color_col1:
                st.markdown(f"<div style='width:80px;height:80px;background:{hex_color};border:2px solid #333;border-radius:8px;'></div>", unsafe_allow_html=True)
            with color_col2:
                st.markdown(f"**📍 坐标：** ({x_img}, {y_img})")
                st.markdown(f"**🎨 RGB值：** {rgb}")
                st.markdown(f"**🔖 HEX值：** {hex_color}")

            # 推荐颜料：使用可复用的 suggest_mix 函数来计算 top_colors 和 weights
            palette_colors = st.session_state.active_colors if st.session_state.active_colors else paint_colors

            # 调用封装好的混合建议函数（可直接迁移到其他项目使用）
            top_colors, weights = suggest_mix(rgb, palette_colors, paint_colors=paint_colors, max_candidates=6)

            # 显示推荐结果（再次过滤非常小的权重以便展示）
            if top_colors and weights is not None:
                weights = np.array(weights)
                filtered = [(c, w) for c, w in zip(top_colors, weights) if w > 0.01]
                if filtered:
                    top_colors, weights = zip(*filtered)
                    top_colors = list(top_colors)
                    weights = np.array(weights)

                    st.header("🖌️ 推荐油画颜料及混合比例")
                    st.markdown('<div style="display:flex;flex-direction:column;gap:10px;margin:12px 0 18px 0;">', unsafe_allow_html=True)
                    for (name, rgb_paint), percent in zip(top_colors, (weights*100).round().astype(int)):
                        st.markdown(
                            f'''<div style="display:flex;align-items:center;gap:18px;min-height:44px;">
                                <div style="width:38px;height:38px;border-radius:8px;background:rgb{tuple(rgb_paint)};border:2px solid #aaa;"></div>
                                <div style="font-size:18px;font-weight:bold;color:#fa8c16;min-width:48px;text-align:center;">{percent}%</div>
                                <div style="font-size:16px;color:#333;word-break:break-all;">{name}</div>
                            </div>''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 混合后理论色块（使用全局的 rgb_to_cmy / cmy_to_rgb）
                    palette_cmy_used = np.array([rgb_to_cmy(c[1]) for c in top_colors])
                    mixed_cmy = np.dot(weights, palette_cmy_used)
                    mixed_rgb = cmy_to_rgb(mixed_cmy)
                    mixed_hex = "#{:02x}{:02x}{:02x}".format(*mixed_rgb)
                    
                    st.subheader("🎨 混合效果对比")
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:center;gap:20px;margin:12px 0;">
                            <div style="text-align:center;">
                                <div style="margin-bottom:8px;font-weight:bold;color:#333;">原始颜色</div>
                                <div style="width:80px;height:80px;background:{hex_color};border:2px solid #333;border-radius:8px;"></div>
                                <div style="margin-top:4px;font-size:12px;color:#666;">{hex_color}</div>
                            </div>
                            <div style="font-size:24px;color:#fa8c16;">→</div>
                            <div style="text-align:center;">
                                <div style="margin-bottom:8px;font-weight:bold;color:#333;">混合后理论色</div>
                                <div style="width:80px;height:80px;background:{mixed_hex};border:2px solid #333;border-radius:8px;"></div>
                                <div style="margin-top:4px;font-size:12px;color:#666;">{mixed_hex}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.info("👆 请在画布上点击任意位置进行取色")

# -----------------------------
# 侧边栏颜料选择
# -----------------------------
st.sidebar.subheader("🎨 油画色卡")
search_term = st.sidebar.text_input("🔍 搜索颜料名称", placeholder="输入颜料名称...")

# 过滤颜料
filtered_colors = paint_colors
if search_term:
    filtered_colors = {name: rgb for name, rgb in paint_colors.items() 
                      if search_term.lower() in name.lower()}

st.sidebar.write(f"显示 {len(filtered_colors)} / {len(paint_colors)} 种颜料")

for name, rgb in filtered_colors.items():
    cols_side = st.sidebar.columns([1, 3])
    with cols_side[0]:
        st.markdown(f"<div style='width:20px;height:20px;background:rgb{tuple(rgb)};border:1px solid #ccc;border-radius:2px;'></div>", unsafe_allow_html=True)
    with cols_side[1]:
        if st.button(name, key=f"btn_{name}", help=f"添加 {name} 到调色盘"):
            if name not in st.session_state.user_colors:
                st.session_state.user_colors[name] = rgb
            st.session_state.active_colors[name] = rgb
            st.experimental_rerun()

# -----------------------------
# 联系方式与反馈
# -----------------------------
st.markdown('''
<style>
.contact-float {
    position: fixed;
    right: 24px;
    bottom: 24px;
    background: #fffbe6;
    border: 1px solid #ffd666;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 16px 20px 12px 20px;
    font-size: 16px;
    color: #666;
    z-index: 9999;
    min-width: 220px;
}
.contact-float a { color: #fa8c16; text-decoration: none; }
</style>
<div class="contact-float">
<div style="font-size:16px; font-weight:bold; margin-bottom:6px;">如有建议或问题欢迎反馈：</div>
<span style="font-size:14px;vertical-align:middle;">🟩</span> <span style="font-size:14px;">微信号：Veep625</span><br>
<span style="font-size:14px;vertical-align:middle;">✉️</span> <span style="font-size:14px;">邮箱：<a href="mailto:wanyixin625@gmail.com">wanyixin625@gmail.com</a></span>
</div>
''', unsafe_allow_html=True)
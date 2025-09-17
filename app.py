import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from skimage import color
import json
import os

# -----------------------------
# 初始化 session_state
# -----------------------------
if "user_colors" not in st.session_state:
    st.session_state.user_colors = {}  # 已选择的颜料
if "active_colors" not in st.session_state:
    st.session_state.active_colors = {}  # 当前调色盘显示

# -----------------------------
# 载入官方油画颜料色库
# -----------------------------
with open("paint_colors.json", "r", encoding="utf-8") as f:
    paint_colors = json.load(f)

# -----------------------------
# 页面布局
# -----------------------------
st.set_page_config(layout="wide")

st.title("🎨 点色取色 + 油画颜料配比")

# 我的色库区
st.header("🖌️ 我的色库")


# 显示色库颜色数量
st.markdown(f"<span style='font-size:15px;color:#888;'>当前色库颜色数量：<b>{len(st.session_state.active_colors)}</b></span>", unsafe_allow_html=True)

# 保存、加载、清空按钮在同一行左侧
btn_cols = st.columns([1,1,1,10])
with btn_cols[0]:
    if st.button("💾 保存"):
        with open("my_palette.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.active_colors, f, ensure_ascii=False, indent=2)
        st.success("已保存调色盘到 my_palette.json")
with btn_cols[1]:
    if st.button("📂 加载"):
        if os.path.exists("my_palette.json"):
            with open("my_palette.json", "r", encoding="utf-8") as f:
                loaded = json.load(f)
            st.session_state.active_colors = loaded
            st.session_state.user_colors.update(loaded)
            st.experimental_rerun()
        else:
            st.warning("my_palette.json 文件不存在")
with btn_cols[2]:
    if st.button("🧹 清空"):
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

# -----------------------------
st.header("📤 上传图片")
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # 新增slider控制取色画布宽度
    canvas_width = st.slider("取色画布宽度", min_value=200, max_value=2400, value=600)
    canvas_height = int(img.height * canvas_width / img.width)

    # resize一份新的PIL图片作为background_image
    img_resized = img.resize((canvas_width, canvas_height))

    # 画布
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=0,
        background_image=img_resized,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="point",
        key=f"canvas_{canvas_width}",
    )

    st.markdown("<div style='color:#fa8c16;font-size:16px;margin:8px 0 0 0;'><b>提示：</b>点击画布任意位置即可取色</div>", unsafe_allow_html=True)

    st.header("🎯 取色")
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            x, y = int(objects[-1]["left"]), int(objects[-1]["top"])
            # 坐标映射到原图（用canvas宽高）
            x_img = int(x * img.width / canvas_width)
            y_img = int(y * img.height / canvas_height)
            img_array = np.array(img)

            # 取周围像素平均
            def get_avg_rgb(img_array, x, y, radius=2):
                h, w, _ = img_array.shape
                x_min, x_max = max(0, x-radius), min(w, x+radius+1)
                y_min, y_max = max(0, y-radius), min(h, y+radius+1)
                patch = img_array[y_min:y_max, x_min:x_max, :]
                return patch.mean(axis=(0,1)).astype(int)

            rgb = get_avg_rgb(img_array, x_img, y_img, radius=0)
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)

            st.markdown(f"**🎯 取色结果：** RGB={tuple(rgb)}, HEX={hex_color}")
            st.markdown(f"<div style='width:100px;height:50px;background:{hex_color}'></div>", unsafe_allow_html=True)

            # -----------------------------
            # 推荐颜料及混合比例
            # -----------------------------
            if st.session_state.active_colors:
                palette_colors = st.session_state.active_colors
            else:
                palette_colors = paint_colors

            def delta_e(rgb1, rgb2):
                lab1 = color.rgb2lab(np.array([[rgb1]])/255.0)[0,0]
                lab2 = color.rgb2lab(np.array([[rgb2]])/255.0)[0,0]
                return np.linalg.norm(lab1 - lab2)

            closest = sorted(palette_colors.items(), key=lambda item: delta_e(rgb, item[1]))

            # ---自动遍历2~4种颜料组合，选最优混合---
            from scipy.optimize import minimize
            import itertools

            def rgb_to_cmy(rgb):
                return 1 - np.array(rgb) / 255.0

            def cmy_to_rgb(cmy):
                return np.clip((1 - cmy) * 255, 0, 255).astype(int)


            best_loss = 1e9
            best_weights = None
            best_colors = None
            best_palette_cmy = None
            candidate_colors = closest[:6]
            rng = np.random.default_rng(42)
            for n in range(2, 5):
                for comb in itertools.combinations(candidate_colors, n):
                    palette_cmy = np.array([rgb_to_cmy(c) for _, c in comb])
                    def loss(w):
                        w = np.clip(w, 0, 1)
                        if w.sum() == 0:
                            return 1e6
                        w = w / w.sum()
                        mixed_cmy = np.dot(w, palette_cmy)
                        mixed_rgb = cmy_to_rgb(mixed_cmy)
                        lab1 = color.rgb2lab(np.array([[rgb]])/255.0)[0,0]
                        lab2 = color.rgb2lab(np.array([[mixed_rgb]])/255.0)[0,0]
                        return np.linalg.norm(lab1 - lab2)
                    N = len(comb)
                    cons = ({'type': 'eq', 'fun': lambda w: np.sum(np.clip(w,0,1)) - 1})
                    bounds = [(0,1)] * N
                    best_local_loss = 1e9
                    best_local_weights = None
                    # 多组初始权重尝试，避免均分陷阱
                    for _ in range(8):
                        w0 = rng.random(N)
                        w0 = w0 / w0.sum()
                        res = minimize(loss, w0, bounds=bounds, constraints=cons)
                        weights = np.clip(res.x, 0, 1)
                        if weights.sum() > 0:
                            weights /= weights.sum()
                        l = loss(weights)
                        if l < best_local_loss:
                            best_local_loss = l
                            best_local_weights = weights
                    if best_local_loss < best_loss:
                        best_loss = best_local_loss
                        best_weights = best_local_weights
                        best_colors = comb
                        best_palette_cmy = palette_cmy

            # 用最优组合
            top_colors = list(best_colors)
            palette_cmy = best_palette_cmy
            weights = best_weights
            # 只保留主要配比（大于5%）的颜色
            filtered = [(c, w) for c, w in zip(top_colors, weights) if w > 0.05]
            if filtered:
                top_colors, weights = zip(*filtered)
                top_colors = list(top_colors)
                weights = np.array(weights)


            st.header("🖌️ 推荐油画颜料及混合比例")
            # 横向一行排列：色块+比例+名称，每组纵向排列
            show_colors = [(name, c, int(round(w*100))) for (name, c), w in zip(top_colors, weights) if w > 0][:4]
            if show_colors:
                st.markdown('<div style="display:flex;flex-direction:column;gap:10px;margin:12px 0 18px 0;">', unsafe_allow_html=True)
                for name, rgb, percent in show_colors:
                    st.markdown(
                        f'''<div style="display:flex;align-items:center;gap:18px;min-height:44px;">
                            <div style="width:38px;height:38px;border-radius:8px;background:rgb{tuple(rgb)};border:2px solid #aaa;"></div>
                            <div style="font-size:18px;font-weight:bold;color:#fa8c16;min-width:48px;text-align:center;">{percent}%</div>
                            <div style="font-size:16px;color:#333;word-break:break-all;">{name}</div>
                        </div>''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 显示混合后理论色块（放在比例之后）
            # 只剩1种颜色时直接用该颜色，否则保证palette_cmy和weights长度一致
            if len(weights) == 1:
                mixed_rgb = np.array(top_colors[0][1])
            else:
                # palette_cmy 可能比 weights 长，需截断
                palette_cmy_used = palette_cmy[-len(weights):] if palette_cmy.shape[0] != len(weights) else palette_cmy
                mixed_cmy = np.dot(weights, palette_cmy_used)
                mixed_rgb = cmy_to_rgb(mixed_cmy)
            mixed_hex = "#{:02x}{:02x}{:02x}".format(*mixed_rgb)
            st.markdown(
                f"<div style='display:inline-block;margin:4px 0 8px 0;'>"
                f"<span style='font-size:15px;color:#888;'>混合后理论色块：</span>"
                f"<span style='display:inline-block;width:40px;height:24px;background:{mixed_hex};border-radius:4px;border:1px solid #ccc;vertical-align:middle;'></span>"
                f" <span style='font-size:13px;color:#888;'>{mixed_hex.upper()}</span>"
                f"</div>", unsafe_allow_html=True
            )


# -----------------------------
# 侧边栏颜料选择
# -----------------------------
st.sidebar.subheader("🎨 官方油画色卡")
for name, rgb in paint_colors.items():
    cols_side = st.sidebar.columns([1,3])
    with cols_side[0]:
        st.markdown(f"<div style='width:20px;height:20px;background:rgb{tuple(rgb)}'></div>", unsafe_allow_html=True)
    with cols_side[1]:
        if st.button(name, key=f"btn_{name}"):
            if name not in st.session_state.user_colors:
                st.session_state.user_colors[name] = rgb
            st.session_state.active_colors[name] = rgb
            st.experimental_rerun()  # 点击立即刷新页面

# -----------------------------
# 联系方式与反馈
# -----------------------------

# 右下角悬浮联系方式与反馈
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
    font-size: 18px;
    color: #666;
    z-index: 9999;
    min-width: 220px;
}
.contact-float a { color: #fa8c16; text-decoration: none; }
</style>
<div class="contact-float">
<div style="font-size:18px; font-weight:bold; margin-bottom:6px;">如有建议或问题欢迎反馈：</div>
<span style="font-size:16px;vertical-align:middle;">🟩</span> <span style="font-size:16px;">微信号：Veep625</span><br>
<span style="font-size:16px;vertical-align:middle;">✉️</span> <span style="font-size:16px;">邮箱：<a href="mailto:wanyixin625@gmail.com">wanyixin625@gmail.com</a></span>
</div>
''', unsafe_allow_html=True)
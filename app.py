import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from skimage import color
import json
import os
from scipy.optimize import minimize
import itertools

# -----------------------------
# 初始化 session_state
# -----------------------------
if "user_colors" not in st.session_state or not isinstance(st.session_state.get("user_colors"), dict):
    st.session_state.user_colors = {}
if "active_colors" not in st.session_state or not isinstance(st.session_state.get("active_colors"), dict):
    st.session_state.active_colors = {}

# -----------------------------
# 载入官方油画颜料色库
# -----------------------------
data_path = os.path.join(os.path.dirname(__file__), "paint_colors.json")
with open(data_path, "r", encoding="utf-8") as f:
    paint_colors = json.load(f)

# -----------------------------
# 页面布局
# -----------------------------
st.set_page_config(layout="wide")
st.title("🎨 点色取色 + 油画颜料配比")

# 我的色库区
st.header("🖌️ 我的色库")

st.markdown(
    f"<span style='font-size:15px;color:#888;'>当前色库颜色数量：<b>{len(st.session_state.active_colors)}</b></span>",
    unsafe_allow_html=True
)

# 保存/加载/清空
btn_cols = st.columns([1, 1, 1, 10])
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
            st.rerun()
        else:
            st.warning("my_palette.json 文件不存在")
with btn_cols[2]:
    if st.button("🧹 清空"):
        st.session_state.active_colors = {}
        st.rerun()

# 显示缩略调色盘（hover 显示 ×）
if st.session_state.active_colors:
    st.markdown(
        """
        <style>
        .color-box { position:relative; display:inline-block; margin:4px; }
        .color-square { width:50px; height:50px; border-radius:4px; border:1px solid #aaa; }
        .color-name { width:74px; text-align:left; font-size:14px; margin-top:2px; line-height:1.2; word-break:break-word; }
        .del-btn {
            position:absolute; top:-6px; right:-6px;
            width:20px; height:20px;
            background:#ff4d4f; color:white; border:none;
            border-radius:50%; cursor:pointer; font-size:14px;
            display:none;
        }
        .color-box:hover .del-btn { display:block; }
        </style>
        """,
        unsafe_allow_html=True
    )

    keys = list(st.session_state.active_colors.keys())
    for name in keys:
        rgb = st.session_state.active_colors[name]
        delete_key = f"del_{name}"
        st.markdown(
            f"""
            <div class="color-box">
                <div class="color-square" style="background:rgb{tuple(rgb)}"></div>
                <div class="color-name">{name}</div>
                <form action="" method="get">
                    <button class="del-btn" name="del" value="{name}">×</button>
                </form>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 处理删除动作
    del_name = st.query_params.get("del")
    if del_name and del_name in st.session_state.active_colors:
        del st.session_state.active_colors[del_name]
        st.query_params.clear()
        st.rerun()
else:
    st.write("当前色库为空")

st.markdown("---")

# -----------------------------
st.header("📤 上传图片")
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    canvas_width = st.slider("取色画布宽度", min_value=200, max_value=2400, value=600)
    canvas_height = int(img.height * canvas_width / img.width)
    img_resized = img.resize((canvas_width, canvas_height))

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
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        if objects:
            x, y = round(objects[-1]["left"]), round(objects[-1]["top"])
            x_img = round(x * img.width / canvas_width)
            y_img = round(y * img.height / canvas_height)
            img_array = np.array(img)

            def get_avg_rgb(img_array, x, y, radius=2):
                h, w, _ = img_array.shape
                x_min, x_max = max(0, x-radius), min(w, x+radius+1)
                y_min, y_max = max(0, y-radius), min(h, y+radius+1)
                patch = img_array[y_min:y_max, x_min:x_max, :]
                return patch.mean(axis=(0, 1)).astype(int)

            rgb = get_avg_rgb(img_array, x_img, y_img, radius=0)
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)

            st.markdown(f"**🎯 取色结果：** RGB={tuple(rgb)}, HEX={hex_color}")
            st.markdown(f"<div style='width:100px;height:50px;background:{hex_color}'></div>", unsafe_allow_html=True)

            # 推荐颜料
            palette_colors = st.session_state.active_colors if st.session_state.active_colors else paint_colors

            def delta_e(rgb1, rgb2):
                lab1 = color.rgb2lab(np.array([[rgb1]])/255.0)[0, 0]
                lab2 = color.rgb2lab(np.array([[rgb2]])/255.0)[0, 0]
                return np.linalg.norm(lab1 - lab2)

            closest = sorted(palette_colors.items(), key=lambda item: delta_e(rgb, item[1]))
            candidate_colors = closest[:6]

            def rgb_to_cmy(rgb):
                return 1 - np.array(rgb) / 255.0

            def cmy_to_rgb(cmy):
                return np.clip((1 - cmy) * 255, 0, 255).astype(int)

            best_loss = 1e9
            best_weights, best_colors = None, None
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
                        lab1 = color.rgb2lab(np.array([[rgb]])/255.0)[0, 0]
                        lab2 = color.rgb2lab(np.array([[mixed_rgb]])/255.0)[0, 0]
                        return np.linalg.norm(lab1 - lab2)

                    N = len(comb)
                    cons = ({'type': 'eq', 'fun': lambda w: np.sum(np.clip(w, 0, 1)) - 1})
                    bounds = [(0, 1)] * N
                    best_local_loss = 1e9
                    best_local_weights = None

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

            if best_colors and best_weights is not None:
                filtered = [(c, w) for c, w in zip(best_colors, best_weights) if w > 0.05]
                if filtered:
                    top_colors, weights = zip(*filtered)
                    weights = np.array(weights)

                    st.header("🖌️ 推荐油画颜料及混合比例")
                    st.markdown('<div style="display:flex;flex-direction:column;gap:10px;margin:12px 0 18px 0;">', unsafe_allow_html=True)
                    for (name, rgb), percent in zip(top_colors, (weights*100).round().astype(int)):
                        st.markdown(
                            f'''<div style="display:flex;align-items:center;gap:18px;min-height:44px;">
                                <div style="width:38px;height:38px;border-radius:8px;background:rgb{tuple(rgb)};border:2px solid #aaa;"></div>
                                <div style="font-size:18px;font-weight:bold;color:#fa8c16;min-width:48px;text-align:center;">{percent}%</div>
                                <div style="font-size:16px;color:#333;word-break:break-all;">{name}</div>
                            </div>''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 混合后理论色块
                    palette_cmy_used = np.array([rgb_to_cmy(c[1]) for c in top_colors])
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
    cols_side = st.sidebar.columns([1, 3])
    with cols_side[0]:
        st.markdown(f"<div style='width:20px;height:20px;background:rgb{tuple(rgb)}'></div>", unsafe_allow_html=True)
    with cols_side[1]:
        if st.button(name, key=f"btn_{name}"):
            if name not in st.session_state.user_colors:
                st.session_state.user_colors[name] = rgb
            st.session_state.active_colors[name] = rgb
            st.rerun()

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

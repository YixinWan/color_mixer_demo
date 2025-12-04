# 油画调色demo — 简明说明
基于streamlit的简洁的本地 Web 应用，用于从图片取色并给出油画颜料的混合建议。

快速上手
1. 安装依赖：

```powershell
pip install -r requirements.txt
```

2. 启动应用：

```powershell
streamlit run app.py
```

主要功能（简述）
- 上传图片并在画布上点击取色
- 基于Lab空间色差给出颜料混合建议与比例
- 支持保存/加载自定义调色盘（`my_palette.json`）

关键算法说明
- `suggest_mix`：基础算法，使用 CMY 线性混合 + Lab 最小化。【已测试过，可用】
- `suggest_mix_advanced`：进阶算法，按色相优先筛选主色，并加入白/黑与互补色规则以更贴近画家习惯。 【未测试，暂不用】

文件说明
- `app.py`：主程序（Streamlit）
- `paint_colors.json`：内置颜料库，可自定义
- `my_palette.json`：用户调色盘（可保存/加载）
- `requirements.txt`：依赖
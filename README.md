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

#### 混色建议函数

1. **`suggest_mix(target_rgb, palette_source, paint_colors=None, max_candidates=6)`** 
   - **功能**：基础混色建议算法，给定目标颜色，返回最佳混合的颜料名称与权重
   - **原理**：
     - Lab 空间色差度量（筛选候选颜料）
     - CMY 颜色空间线性混合模拟颜料混色
     - SLSQP 约束优化寻找 1~4 种颜料的最优混合比例
   - **输入**：`target_rgb` (目标RGB) + `palette_source` (调色盘字典或文件路径)
   - **输出**：`top_colors` (颜料列表) + `weights` (权重数组)
   - **状态**：✅ 已测试，主流程使用

2. **`suggest_mix_advanced(target_rgb, palette_source, paint_colors=None, max_candidates=6, hue_filter_n=10)`**
   - **功能**：进阶混色建议，更符合传统油画调色习惯
   - **特点**：
     - 色相优先筛选（先找"本色"）
     - 亮度智能规则（高亮允许加白色，低亮允许加黑色）
     - 互补色策略（以小比例加入互补色以降低饱和度）
   - **输入/输出**：同 `suggest_mix`
   - **状态**：⚠️ 实验中，暂未在主流程使用

#### 分步调色指导函数

3. **`generate_steps_from_mix(top_colors, weights, max_total=15)`** ⭐ **【新增】**
   - **功能**：纯逻辑函数，将混色建议转换为分步调色指导数据（无 UI 依赖）
   - **特点**：
     - 内置小整数近似算法（穷举最优整数份数组合）
     - 逐步加入颜料（主色份数固定，次要颜料逐份增加）
     - 返回每步的部件、颜色名称、RGB值、混合效果
   - **输入**：
     - `top_colors`：`suggest_mix` 的输出（颜料列表）
     - `weights`：混合权重数组
     - `max_total`：份数上限（默认15）
   - **输出**：步骤列表，每项为字典：
     ```python
     {
       'step_num': 步骤号,
       'parts': 各颜色份数,
       'names': 各颜色名称,
       'rgbs': 各颜色RGB值,      # 用于 UI 渲染色块 + 计算混合色
       'mixed_hex': 混合后HEX值  # 当前步的理论混合色
     }
     ```
   - **状态**：✅ 已测试，已集成到主流程

#### 辅助函数

4. **`rgb_to_cmy(rgb)` / `cmy_to_rgb(cmy)`**
   - **功能**：RGB ↔ CMY 色彩空间转换
   - **用途**：模拟油画颜料的减色混合（CMY 模型比 RGB 更符合颜料特性）

---

## 快速使用示例（复用纯逻辑函数）

如果你想在**其他项目**中复用分步调色指导的纯逻辑部分（无需 Streamlit），可以这样做：

```python
import numpy as np
from skimage import color

# 从 app.py 中复制以下函数：
# - suggest_mix(target_rgb, palette_source, ...)
# - generate_steps_from_mix(top_colors, weights, max_total=15)
# - rgb_to_cmy(rgb) / cmy_to_rgb(cmy)

# 示例：给定目标颜色和调色盘，生成分步混色方案
target_color = [230, 100, 50]  # 某个橙红色
my_palette = {
    "Cadmium Red": [255, 0, 0],
    "Titanium White": [255, 255, 255],
    "Alizarin": [200, 50, 50]
}

# 步骤1：获取混色建议
top_colors, weights = suggest_mix(target_color, my_palette)

# 步骤2：生成分步调色数据（纯逻辑，无 UI）
steps = generate_steps_from_mix(top_colors, weights, max_total=10)

# 步骤3：使用步骤数据
for step in steps:
    print(f"第 {step['step_num']} 步：")
    for name, parts in zip(step['names'], step['parts']):
        print(f"  - {name}: {parts} 份")
    print(f"  预期混合色: {step['mixed_hex']}")
```

---

文件说明
- `app.py`：主程序（Streamlit）
- `paint_colors.json`：内置颜料库，可自定义
- `my_palette.json`：用户调色盘（可保存/加载）
- `requirements.txt`：依赖
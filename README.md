# SettlementPro

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![Status](https://img.shields.io/badge/status-Developing-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

SettlementPro 是一个基于 Python 的桌面端结算工具，提供图形化界面，用于项目/工程相关数据的整理与结算计算，适合个人或小型项目在本地使用。

> 本项目为个人开源项目，当前处于持续开发与优化阶段。

---

## ✨ 功能特性

- 图形化界面操作（GUI）
- 项目数据录入与管理
- 自动结算与结果汇总
- 本地数据保存
- 无需数据库，开箱即用

---

## 📸 界面截图

```
docs/
 ├── main_window.png
 └── settlement_result.png
```

在 README 中引用：

```markdown
![主界面](docs/main_window.png)
![结算结果](docs/settlement_result.png)
```

---

## 🧩 项目结构

```
SettlementPro/
├── SettlementPro.py
└── README.md
```

---

## ⚙️ 运行环境

- Python >= 3.8
- Windows（推荐）
- Linux / macOS（未完整测试）

---

## 📦 依赖库

示例：

```
PyQt5
pandas
openpyxl
```

安装：

```bash
pip install -r requirements.txt
```

---

## 🚀 快速开始

```bash
git clone https://github.com/ForestSun2023/SettlementPro.git
cd SettlementPro
python SettlementPro.py
```

---

## 🛠 开发说明

- 当前代码为单文件结构（3000+ 行）
- 建议拆分模块：
  - ui
  - core
  - models
  - utils

---

## 🐞 已知问题

- 结构耦合较高
- 缺少测试
- 跨平台未充分验证
- 文档不足

---

## 🗺 计划功能

- [ ] 模块化重构
- [ ] Excel / CSV 导入导出
- [ ] 自动生成报表
- [ ] 日志系统
- [ ] Windows 可执行文件
- [ ] 多项目管理

---

## 🤝 贡献方式

1. Fork 仓库
2. 新建分支
3. 提交代码
4. 发起 PR

---

## 📄 许可证

MIT License

---

## 👤 作者

Sun Fusen  
GitHub: https://github.com/ForestSun2023

---

## ⭐ 支持项目

欢迎 Star ⭐

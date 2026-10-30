# 开发验证

在插件目录使用 ComfyUI 对应 Python 环境执行测试，依赖使用 requirements.txt，不回退全局 pip。

```powershell
<ComfyUI-Python> -m unittest discover -s tests -p "test_*.py" -v
node --test tests/test_*.mjs
```

Windows 下若临时目录缩写导致路径比较失败，请在当前终端将 TEMP/TMP 指向可写的完整绝对目录。不要修改系统级临时目录配置。

本地模型 CPU/tokenizer 检查需要完整的 MOSS-Music 文件；可通过 H3LV_TEST_MODEL_DIR 指定模型目录。缺少模型时相关可选检查应明确跳过；不能把跳过写成已验证真实模型。

浏览器交互、真实模型推理、本地 H3 成片、RunningHub 调用应分别使用有权使用的素材和隔离项目验收。测试不得依赖个人工作流、真实项目标识或私密配置。公开结果只保留必要摘要。

[README.md](https://github.com/user-attachments/files/32162350/README.md)
# CAIL_test

基于 **Qwen3-4B + vLLM** 的中文法律问答推理测试项目。

本仓库用于在 CAIL 风格法律案例问答数据上测试本地大语言模型的推理与回答能力。当前实验环境使用 Qwen3-4B，并通过 vLLM 提供 OpenAI-compatible API，便于后续进行批量推理、结果保存和效果评估。

## 项目内容

当前仓库主要包含：

```text
CAIL_test/
├── models/
│   └── Qwen3-4B/          # Qwen3-4B 模型目录 / 模型配置
├── outputs/               # 模型推理输出
├── official_test.jsonl    # 测试数据
├── test_data.json         # 测试数据
└── README.md
```

> 大模型权重通常体积较大。若仓库中未包含完整权重文件，请自行准备 Qwen3-4B，并确保本地模型目录为 `./models/Qwen3-4B`。

## 数据格式

测试数据以法律案例问答为主，每条样本包含以下字段：

```json
{
  "id": "0_1_0",
  "big_ques": "案件事实与完整题干",
  "small_ques": "需要回答的具体法律问题",
  "score": 9.0
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `id` | 样本唯一编号 |
| `big_ques` | 完整案件事实或主问题 |
| `small_ques` | 针对案件提出的具体子问题 |
| `score` | 该题对应的分值 |

部分样本可能没有单独的 `small_ques`，此时可直接使用 `big_ques` 作为模型输入。

## 实验环境

本项目当前已验证的服务器环境：

- GPU: NVIDIA RTX A6000 48GB
- NVIDIA Driver: 550.54.15
- CUDA: 12.4
- Python: 3.10
- PyTorch: 2.6.0+cu124
- vLLM: 0.8.5
- Model: Qwen3-4B

建议使用独立 Python 虚拟环境运行。

## 启动 vLLM

进入项目目录：

```bash
cd ~/cail-vllm
```

激活虚拟环境：

```bash
source .venv-vllm/bin/activate
```

启动前可以通过以下命令检查 GPU 显存：

```bash
nvidia-smi
```

在当前已验证环境中，可使用以下命令启动 Qwen3-4B：

```bash
TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
vllm serve ./models/Qwen3-4B \
  --served-model-name Qwen3-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.6 \
  --max-model-len 8192 \
  --enforce-eager
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--served-model-name Qwen3-4B` | API 中使用的模型名称 |
| `--host 0.0.0.0` | 监听服务器网络接口 |
| `--port 8000` | API 服务端口 |
| `--gpu-memory-utilization 0.6` | vLLM 可使用的 GPU 显存比例 |
| `--max-model-len 8192` | 最大上下文长度 |
| `--enforce-eager` | 使用 eager mode，避免当前服务器上的 TorchInductor/Triton 编译问题 |

出现类似以下信息即表示服务启动成功：

```text
INFO: Started server process [...]
INFO: Waiting for application startup.
INFO: Application startup complete.
```

## API 测试

### 查询模型

启动 vLLM 后，在另一个终端执行：

```bash
curl http://127.0.0.1:8000/v1/models
```

正常情况下会返回 `Qwen3-4B`。

### Chat Completions

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B",
    "messages": [
      {
        "role": "user",
        "content": "你好，请用中文简单介绍一下你自己。 /no_think"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Qwen3 支持 thinking 模式。对于不需要长推理的简单测试，可以在提示词中加入 `/no_think`，减少思考过程带来的额外 token 消耗。

## Python 调用示例

vLLM 提供 OpenAI-compatible API，因此也可以通过 Python 调用。

安装 OpenAI Python SDK：

```bash
pip install openai
```

示例：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen3-4B",
    messages=[
        {
            "role": "user",
            "content": "请分析下面的法律问题：…… /no_think",
        }
    ],
    temperature=0.7,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

## 推荐的测试流程

项目的基本实验流程如下：

```text
CAIL 测试数据
      ↓
读取 big_ques / small_ques
      ↓
构造 Prompt
      ↓
Qwen3-4B
      ↓
vLLM OpenAI-compatible API
      ↓
模型回答
      ↓
outputs/
      ↓
后续评测与分析
```

后续可以继续补充：

- 批量读取 `official_test.jsonl`
- 自动调用 vLLM API
- 保存 Qwen3-4B 的预测结果
- 根据题目 `score` 或参考答案设计评测方法
- 对比不同 Prompt
- 对比 thinking / non-thinking 模式
- 对比不同模型或微调模型

## 常见问题

### 1. CUDA Out of Memory

如果出现：

```text
torch.OutOfMemoryError: CUDA out of memory
```

首先执行：

```bash
nvidia-smi
```

检查 GPU 是否正在被其他用户或任务占用。共享 GPU 环境中，即使 GPU 总显存为 48GB，也必须以实际空闲显存为准。

### 2. TorchInductor / Triton 编译失败

当前服务器环境中曾出现 TorchInductor/Triton 调用 `gcc` 编译失败的问题，因此启动命令保留：

```bash
--enforce-eager
```

以优先保证推理服务稳定运行。

### 3. `torch_c_dlpack_ext` 兼容问题

当前环境使用：

```bash
TVM_FFI_DISABLE_TORCH_C_DLPACK=1
```

绕过可选 DLPack 扩展的二进制兼容问题。若未来升级 PyTorch、vLLM 或相关依赖，应重新验证该设置是否仍有必要。

## 注意事项

- 本项目目前主要用于实验、学习和模型评测。
- 法律问答结果由大语言模型生成，可能存在事实错误、法律理解偏差或幻觉。
- 模型输出不应被视为正式法律意见。
- 上传公开仓库前，请确认数据集、模型权重以及其他资源的许可证和再分发条件。
- 不要将 API Key、密码、SSH 私钥或其他敏感信息提交到 GitHub。

## Roadmap

- [x] 准备 CAIL 测试数据
- [x] 准备 Qwen3-4B
- [x] 在 RTX A6000 上启动 vLLM
- [x] 验证 `/v1/models`
- [x] 验证 `/v1/chat/completions`
- [ ] 编写批量推理脚本
- [ ] 保存完整预测结果
- [ ] 构建自动评测流程
- [ ] 对比不同推理参数
- [ ] 尝试 LoRA / QLoRA 微调
- [ ] 对比微调前后的法律问答表现

## License

本仓库中的代码、数据与模型可能分别适用不同许可证。

在使用或再分发前，请分别确认：

1. CAIL 数据集的使用条款；
2. Qwen3 模型的许可证；
3. 本仓库后续新增代码的许可证。

如果计划公开发布或用于科研成果，请补充明确的项目 `LICENSE` 文件及数据来源说明。

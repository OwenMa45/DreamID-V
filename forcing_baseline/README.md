# DreamID-V Causal-Forcing 后训练框架

把 **DreamID-V-Wan-1.3B-Faster**（双向注意力扩散换脸）通过 [Causal-Forcing](https://github.com/) 三阶段蒸馏后训练为**因果自回归流式换脸**模型。Stage2 采用 **Consistency Distillation (CD)**（非 ODE distillation）。新增 **Stage0** 用原版 DreamID-V 推理管线自蒸馏生成训练数据。

> 交付说明：本目录是一套**完整可运行的代码框架**（configs / scripts / README 就绪）。由于当前工作区缺少模型权重与数据集，未在本地执行大规模训练；请按下文填好权重/数据路径后再运行。

## 整体数据流

```
源视频 + 参考脸(语料)
   │  Stage0: 原版双向 DreamID-V 推理 + DWPose mask（自蒸馏）
   ▼
LMDB 数据集: clean换脸latent + y(源+mask) + img_ref
   │
dreamidv_faster.pth(双向) ──StageA: 权重转换──▶ causal_init.pt
   │                                              │
   └──────────────┐                               ▼
                  ▼                    Stage1 AR Diffusion (teacher forcing)
            (数据集 latent)                        │ ar_diffusion.pt
                  │                                ▼
                  ├───────────────────▶ Stage2 Causal CD（仅需 GT latent）
                  │                                │ causal_cd.pt
                  │                                ▼
                  └──(条件 y, img_ref)──▶ Stage3 DMD（real_score=双向 teacher 冻结）
                                                   │ causal_dmd.pt
                                                   ▼
                                       inference.py 逐块自回归流式换脸
```

## 核心设计

- **同基座**：DreamID-V-Faster 与 Causal-Forcing 同为 Wan2.1-1.3B（480p latent `[21,16,60,104]`，1560 tokens/帧）。差异只在条件注入。
- **条件注入**：`patch_embedding.in_dim=48`（噪声 16 + `y`= 源视频 16 + mask 16 通道拼接）；`ref_conv` 把参考脸编码成 1 帧 prefix token。
- **因果化**：以 Causal-Forcing 的 `CausalWanModel` 为骨架（KV cache、block-causal mask、`causal_rope_apply`、teacher-forcing mask），嫁接上述条件。`img_ref` 作为 **attention sink**：rollout 开始时写入 KV cache 第 0 帧，永不淘汰，后续所有视频帧都能注意到它；视频帧从第 1 帧开始索引。
- **CFG 语义**：guidance 作用在 `img_ref`（cond=真脸 / uncond=零脸），对齐 DreamID-V 的 CFG；`unconditional_dict` 把 `img_ref` 置零、保留 `y` 与 prompt。
- **chunkwise**：`num_frame_per_block=3`，更稳更省。

## 目录结构

```
forcing_baseline/
├── wan/modules/                # 模型骨架
│   ├── causal_dreamidv_model.py   # 因果 student（CausalWanModel + y/img_ref）
│   ├── dreamidv_model.py          # 双向 teacher/critic
│   └── attention.py vae.py t5.py ... (vendored from DreamID-V)
├── utils/                      # wrapper / scheduler / dataset / 数据编码 / 分布式 / lmdb
│   ├── dreamidv_wrapper.py        # DreamIDVDiffusionWrapper + VAE/Text 封装
│   ├── dataset.py swap_data.py    # SwapLatentLMDBDataset + Stage0 编码/LMDB 写入
│   └── scheduler.py distributed.py loss.py misc.py lmdb_.py
├── pipeline/                   # AR rollout
│   ├── self_forcing_training.py      # Stage3 DMD generator AR rollout
│   ├── causal_diffusion_inference.py # Stage2 CD teacher AR 1-step（inference_for_genuine_cd）
│   └── causal_inference.py           # few-step 流式推理
├── model/                      # base / diffusion(S1) / naive_consistency(S2) / dmd(S3)
├── trainer/                    # diffusion(S1) / naive_cd(S2) / distillation(S3)
├── tools/                      # syncid_generate_data.py(S0) / convert_dreamidv_to_causal.py(SA)
├── configs/                    # default + ar_diffusion + causal_cd + dmd + inference
├── scripts/                    # stage0/A/1/2/3 + infer
├── train.py inference.py       # 顶层入口
└── requirements.txt setup.py
```

## 准备权重

在 `configs/default_config.yaml` 填好：

- `t5_checkpoint` / `t5_tokenizer` / `vae_checkpoint`：Wan2.1 的 T5 与 VAE。
- 各 stage 配置里的 `*_ckpt`：见下。

## 完整流程

```bash
cd DreamID-V/forcing_baseline
pip install -r requirements.txt
```

### Stage0 自蒸馏生成数据

准备 `corpus.jsonl`，每行：

```json
{"ref_video": "/path/driving.mp4", "ref_image": "/path/ref.jpg", "mask": "(可选)", "prompt": "(可选)"}
```

```bash
DREAMIDV_ROOT=/mnt/nas/share/home/lzk/mrq/swap/DreamID-V \
CKPT_DIR=checkpoints/Wan2.1 DREAMIDV_CKPT=checkpoints/dreamidv_faster.pth \
MANIFEST=corpus.jsonl OUTPUT_LMDB=dataset/swap_latents \
bash scripts/stage0_gen_data.sh
```

产物：LMDB（`clean_latent` / `y` / `img_ref` / `prompts`）。

### StageA 权重转换（双向 → 因果初始化）

```bash
DREAMIDV_CKPT=checkpoints/dreamidv_faster.pth OUTPUT_CKPT=checkpoints/causal_init.pt \
bash scripts/stageA_convert.sh
```

### Stage1 AR Diffusion（teacher forcing）

```bash
# configs/ar_diffusion.yaml: generator_ckpt=checkpoints/causal_init.pt
bash scripts/stage1_ar.sh   # → checkpoints/chunkwise/stage1_ar/.../model.pt
```

flow-matching MSE + 块内共享 timestep + teacher forcing（`clean_x`=GT latent）。

### Stage2 Causal CD（仅需 GT latent）

```bash
# configs/causal_cd.yaml: generator_ckpt=Stage1 产物（student/teacher/EMA 均由此初始化）
bash scripts/stage2_cd.sh   # → causal_cd.pt
```

teacher 用因果 AR 1-step 把 `latent_t → latent_t_next`（img_ref CFG），
loss=`MSE(student(latent_t), EMA(latent_t_next))`，`discrete_cd_N=48`。

### Stage3 DMD

```bash
# configs/dmd.yaml: generator_ckpt=Stage2 产物;
#   real_score_ckpt / fake_score_ckpt = dreamidv_faster.pth（双向）
bash scripts/stage3_dmd.sh  # → causal_dmd.pt
```

generator=因果 student；real_score=双向 DreamID-V-Faster（冻结）；fake_score=双向 critic（可训）。
DMD loss 与 critic loss 交替（`dfake_gen_update_ratio=5`）；ID/ArcFace loss 默认关闭（`lambda_id=0`，留接口）。

### 推理（逐块自回归流式换脸）

```bash
DREAMIDV_ROOT=/mnt/nas/share/home/lzk/mrq/swap/DreamID-V \
REF_VIDEO=assets/driving.mp4 REF_IMAGE=assets/ref.jpg SAVE_FILE=outputs/swapped.mp4 \
bash scripts/infer.sh
```

`configs/inference.yaml` 里 `generator_ckpt` 指向 Stage3 产物；`denoising_step_list` 默认 4 步。

## 实现要点对照

| 阶段 | trainer | model | 关键点 |
| --- | --- | --- | --- |
| S1 | `diffusion` | `CausalDiffusion` | flow MSE + TF；`clean_x`=GT |
| S2 | `consistency_distillation` | `NaiveConsistency` | genuine-CD；teacher AR 1-step + img_ref CFG；EMA target |
| S3 | `distillation` | `DMD` | Self-Forcing rollout；双向 real/fake score；img_ref CFG |

- **条件 threading**：`conditional_dict={prompt_embeds, y, img_ref}`；rollout 中按当前 block 对 `y` 时间切片；`img_ref` 在开始 forward 一次写入 KV cache 当 sink。
- **VAE 一致性**：forcing_baseline 的 VAE 封装与原版 `WanVAE` 使用同一套 Wan2.1 latent 归一化（`scale=[mean, 1/std]`），故 Stage0 落地的 latent 与 student 训练/推理空间一致。
- **StageA**：DreamID-V 与因果模型层名一致（blocks q/k/v/o + norm_q/k、patch_embedding(48)、ref_conv、text/time embedding、head），按白名单 key-copy。
```

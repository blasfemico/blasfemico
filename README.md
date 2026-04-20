
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,2,20&height=280&section=header&text=Dante%20Villena&fontSize=75&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=CTO%20·%20Independent%20LLM%20Researcher%20·%20Senior%20Backend%20Engineer&descAlignY=55&descSize=18" width="100%" />

<br>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=26&duration=3000&pause=800&color=00F5FF&center=true&vCenter=true&multiline=false&repeat=true&width=900&height=60&lines=Building+Wraith+%E2%9A%A1+the+first+100%25+integer+LLM;Native+Pure+Quantized+Network+(NPQN);CUDA+Kernels+%7C+2.3x+faster+than+cuBLAS+fp16;501+tok%2Fs+%7C+114+MB+VRAM+%7C+64+mJ+per+token;Open+to+compute+sponsors+%26+seed+accelerators" alt="Typing SVG" />
</a>

<br><br>

<img src="https://komarev.com/ghpvc/?username=blasfemico&label=Profile%20Views&color=ff006e&style=for-the-badge" />
<img src="https://img.shields.io/github/followers/blasfemico?label=Followers&style=for-the-badge&color=00f5ff&logo=github" />
<img src="https://img.shields.io/badge/Status-Open%20to%20Sponsors-39ff14?style=for-the-badge" />
<img src="https://img.shields.io/badge/Based%20In-Argentina%20🇦🇷-ffffff?style=for-the-badge" />

</div>

<br>

<img src="https://raw.githubusercontent.com/platane/platane/output/github-contribution-grid-snake-dark.svg" width="100%" />

<br>

## <img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.png" width="32"> `whoami`

```yaml
name:         Dante Villena
location:     Buenos Aires / San Juan, Argentina 🇦🇷
role:         CTO · Independent LLM Researcher · Senior Backend Engineer
experience:   3+ años Backend mid/senior · multi-year CUDA research
mission:      Make LLMs run everywhere — phones, browsers, embedded — at the Shannon limit.

currently:
  [✓] Training Wraith 186M — NPQN LLM at 3.17 bits/weight (Shannon limit)
  [✓] Custom CUDA kernels — 501 tok/s on RTX 5070 consumer (2.38× cuBLAS fp16)
  [✓] Paper draft for NeurIPS / ICLR 2026
  [ ] 2B scale validation — seeking compute sponsors ($3k H100 credits)

contact:      programmingblas@gmail.com
```

<br>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%" />
</div>

<br>

## <img src="https://user-images.githubusercontent.com/74038190/216121967-7366c88e-6faf-4b21-8e61-afd8f5de0c6c.png" width="32"> Flagship — **Wraith**: Native Pure Quantized Network (NPQN)

> **The first LLM with 100% integer training pipeline.**
> No fp16 matmuls · No fp32 optimizer masters · No post-hoc quantization.
> Just integer arithmetic, end to end, from gradient accumulation to forward pass.

```diff
+ Native    → trained quantized from random initialization, not converted post-hoc
+ Pure      → zero float tensors anywhere in the weight pipeline
+ Quantized → 9-level Dualwire at Shannon limit (3.17 bits/weight)
+ LLM scale → multi-purpose transformer, 186M validated, 2B/100B projected
- No bf16 masters (unlike BitNet b1.58)
- No fp32 optimizer states (unlike standard mixed-precision)
- No post-hoc compression (unlike GPTQ, AWQ, BitsAndBytes)
```

### 📊 Benchmarks — Wraith 186M vs. fp16 baseline (same architecture)

<div align="center">

| Metric | Wraith 186M | cuBLAS fp16 baseline | Δ |
|:---|:---:|:---:|:---:|
| 🚀 Throughput (decode B=1, RTX 5070) | **501 tok/s** | 387 tok/s | **+29%** |
| 💾 Peak Inference VRAM | **114 MB** | 1,031 MB | **−89%** (9× less) |
| ⚡ Energy per Token | **64 mJ** | 84 mJ | **−24%** |
| 🎯 Kernel speedup vs cuBLAS fp16 | **2.38–2.59×** | 1.00× | 2–3× faster ops |
| 💿 Packed storage (5-trits/byte, Shannon-optimal) | **74.9 MB** | 372 MB | **4.97× smaller** |
| 📈 Perplexity WikiText-2 (same training budget) | **102** | 636 | **6.24× better** |

</div>

### 🧠 Technical highlights
- **Custom CUDA kernels**: packed 2-bit GEMV, fused QKV/GateUp, embedding lookup packed, CUDA Graphs compatible
- **Dualwire quantization**: 9 discrete levels (3.17 bits/weight, Shannon-optimal for 2 ternary channels)
- **Shadow int16 optimizer**: persistent integer state with stochastic rounding — distinct from transient matmul accumulators in NITI/Ghaffari
- **ASR** (Adaptive Saturation Relief): novel correction for the Derived-Scale Saturation Coupling (DSSC) failure mode
- Runs a complete LLM on a **\$650 consumer GPU** — same card used for training from scratch

> 📄 **Paper:** targeting NeurIPS / ICLR 2026 · draft in preparation
> 🔒 **Preview under NDA:** [programmingblas@gmail.com](mailto:programmingblas@gmail.com)

<br>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%" />
</div>

<br>

## <img src="https://user-images.githubusercontent.com/74038190/216122003-c855ad5e-1b41-4b4f-aa95-6087dd1e1cfd.png" width="32"> Tech Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=python,pytorch,cuda,cpp,c,linux,docker,postgres,mongodb,redis,fastapi,flask,git,github,vscode,bash,nodejs,ts&perline=9" />

</div>

### 🔬 Research / LLM stack
```python
Languages:    Python · C++ · CUDA · Bash
ML:           PyTorch · Triton · NumPy · CUDA Graphs
Kernels:      Packed ternary GEMV · Fused QKV/GateUp · dp4a · WMMA int8
Training:     Shadow int16 optimizer · Stochastic rounding · STE · ASR
Inference:    Custom end-to-end engine · CPU C++ AVX2 · WebGPU target
```

### ⚙️ Backend stack (3+ years production)
```python
Core:         Python · FastAPI · Flask · RESTful APIs
Databases:    PostgreSQL · MySQL · SQLite · MongoDB · Redis
Integration:  Modbus · Websockets · Selenium · NLP pipelines
DevOps:       Docker · Linux · Git · CI/CD · PostgreSQL tuning
Security:     Data sanitization · SQL hardening · Pentesting (junior)
```

<br>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%" />
</div>

<br>

## <img src="https://user-images.githubusercontent.com/74038190/216122058-4de2d56e-6b4c-48e7-a06f-a6e3bfec5f3d.png" width="32"> Stats

<div align="center">

<img height="180em" src="https://github-readme-stats.vercel.app/api?username=blasfemico&show_icons=true&theme=synthwave&bg_color=0D1117&title_color=00F5FF&icon_color=FF006E&text_color=C9D1D9&hide_border=true&count_private=true" />
<img height="180em" src="https://github-readme-stats.vercel.app/api/top-langs/?username=blasfemico&layout=compact&theme=synthwave&bg_color=0D1117&title_color=00F5FF&text_color=C9D1D9&hide_border=true&langs_count=8" />

<br><br>

<img src="https://streak-stats.demolab.com?user=blasfemico&theme=synthwave&hide_border=true&background=0D1117&stroke=00F5FF&ring=00F5FF&fire=FF006E&currStreakNum=FFFFFF&sideNums=00F5FF&currStreakLabel=FF006E&sideLabels=C9D1D9&dates=C9D1D9" width="60%" />

<br><br>

[![Trophy](https://github-profile-trophy.vercel.app/?username=blasfemico&theme=darkhub&no-frame=true&no-bg=true&column=7&margin-w=10)](https://github.com/ryo-ma/github-profile-trophy)

</div>

<br>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%" />
</div>

<br>

## <img src="https://user-images.githubusercontent.com/74038190/216121990-7b6ec373-4de7-4d43-a04b-7d49d56eb8d2.png" width="32"> What I'm Looking For

<table align="center">
<tr>
<td width="25%" align="center">
<img src="https://user-images.githubusercontent.com/74038190/213866269-5d00981c-7c98-46d7-8a8e-9a2c91f9cd74.gif" width="80"><br>
<b>🧪 Compute Sponsors</b><br>
<sub>Anthropic · Google TRC · Lambda · NVIDIA Inception · AWS Activate</sub><br><br>
<i>2B validation run: ~$3k in H100 credits (1 GPU × ~14 days)</i>
</td>
<td width="25%" align="center">
<img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="80"><br>
<b>🚀 Accelerators</b><br>
<sub>UTEC · Kaszek · NXTP · Antler · Y Combinator</sub><br><br>
<i>Seed round for scaling validation</i>
</td>
<td width="25%" align="center">
<img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="80"><br>
<b>🏢 Enterprise Pilots</b><br>
<sub>On-device · Edge · Air-gapped · LATAM compliance</sub><br><br>
<i>Sub-200 MB LLMs where privacy or hardware matters</i>
</td>
<td width="25%" align="center">
<img src="https://user-images.githubusercontent.com/74038190/213866269-5d00981c-7c98-46d7-8a8e-9a2c91f9cd74.gif" width="80"><br>
<b>🎓 Academic Collaboration</b><br>
<sub>For 20B / 100B scale-up experiments</sub><br><br>
<i>Co-authorship available · reproducible research commitment</i>
</td>
</tr>
</table>

<br>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%" />
</div>

<br>

## <img src="https://user-images.githubusercontent.com/74038190/216122069-5b8169d7-1d8e-4a13-b245-a8e4176c99f8.png" width="32"> Track Record

```
2026 Q2 ─────────────── Wraith v1 (186M) complete · kernel stack · paper draft
            │
2025 ──────────────────── Mid-Senior Backend · FastAPI · Postgres · pentesting (junior)
            │
2022-2023 ──────────── Mid Backend · Python · API design · data pipelines
            │
2022 ──────────────────── Junior Backend · first production systems
```

Public selected projects: [`NeuroADAN-PAPER`](https://github.com/blasfemico/NeuroADAN-PAPER) · [`SecureBrain`](https://github.com/blasfemico/SecureBrain) · [`SAC`](https://github.com/blasfemico/SAC)

<br>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%" />
</div>

<br>

## <img src="https://user-images.githubusercontent.com/74038190/216122029-6d334758-4f29-4d7f-b0bc-5dbcb0f55c13.png" width="32"> Connect

<div align="center">

<a href="mailto:programmingblas@gmail.com">
  <img src="https://img.shields.io/badge/Gmail-FF006E?style=for-the-badge&logo=gmail&logoColor=white&labelColor=FF006E" />
</a>
<a href="https://www.linkedin.com/in/dante-villena/">
  <img src="https://img.shields.io/badge/LinkedIn-00F5FF?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=00F5FF" />
</a>
<a href="https://github.com/blasfemico">
  <img src="https://img.shields.io/badge/GitHub-39FF14?style=for-the-badge&logo=github&logoColor=black&labelColor=39FF14" />
</a>
<img src="https://img.shields.io/badge/Discord-_blasfemia-7289DA?style=for-the-badge&logo=discord&logoColor=white&labelColor=7289DA" />

</div>

<br><br>

<blockquote align="center">
<i>"I think, therefore I compile — and the next LLM should run in 114 MB."</i>
</blockquote>

<br>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,2,20&height=120&section=footer" width="100%" />

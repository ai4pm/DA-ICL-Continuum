# Environment and hardware

Use `environment.yml` for CUDA systems or `requirements.txt` in an existing Python 3.12.3 environment. The intended CUDA combination is PyTorch 2.6.0 with CUDA 12.4. Confirm the driver supports CUDA 12.4 before running TabPFN.

The recommended GPU profile is one NVIDIA H100 with at least 80 GB framebuffer per process, 4 CPU cores, and 48 GB host RAM. Historical jobs ran on cluster nodes labeled H100 and V100S; their scheduler metadata did not retain a portable per-device framebuffer value. Record `nvidia-smi --query-gpu=name,memory.total --format=csv` on the target node rather than inferring memory from the Slurm GPU label. Avoid sharing one GPU between independent TabPFN workers unless memory has been measured for the target context/query load.

Approximate observed/planning ranges:

| Operation | Hardware | Approximate time |
|---|---|---:|
| Unit tests excluding TabPFN inference | 1 CPU core | under 1 minute |
| Synthetic smoke config | 8 CPU cores | 1–3 minutes |
| One 30-replicate synthetic sweep point | 8 CPU cores | 3–10 minutes |
| Full serial synthetic paper sweep | 8 CPU cores | several hours |
| One TCGA task/seed | H100 or V100S | tens of minutes, workload dependent |

Record `nvidia-smi`, package versions, checkpoint path/hash, estimator parameters, task name, split seed, wall time, and `last_context_report_` for each authorized run.

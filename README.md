# An Official Code Implementation of CLASP: General-purpose Clothes Manipulation with Semantic Keypoints

If you find this work useful, please cite:
```bibtex
@article{deng2026clasp,
  title={{CLASP}: General-purpose clothes manipulation with semantic keypoints},
  author={Deng, Yuhong and Tang, Chao and Yu, Cunjun and Li, Linfeng and Hsu, David},
  journal={The International Journal of Robotics Research},
  year={2026},
  doi={10.1177/02783649261478649},
  url={https://doi.org/10.1177/02783649261478649}
}
```

## Installation

CLASP uses three vision models: [OWL-ViT](https://huggingface.co/docs/transformers/en/model_doc/owlv2),
[Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything), and
[SD-DINO](https://github.com/Junyi42/geoaware-sc).

1. Create and activate a Python 3.10 environment. Python 3.10 matches the
   precompiled PyFlex binding included in this repository.

```bash
conda create --name clasp python=3.10
conda activate clasp
```

2. Install the core Python dependencies in the `clasp` environment. This list
   covers the current inference and evaluation pipeline but is not exhaustive.
   Install `torch` and `torchvision` using the command appropriate for your
   system from the [PyTorch installation
   guide](https://pytorch.org/get-started/locally/), then install the remaining
   packages:

```bash
python -m pip install numpy scipy opencv-python pillow matplotlib \
  scikit-learn requests pyyaml imageio tqdm pyquaternion
```

3. Install each model and its dependencies in a suitable environment. OWL-ViT,
   SAM, and SD-DINO may use separate conda environments; they do not need to be
   installed in the `clasp` environment.

4. Implement and launch an HTTP service for each model. Example service
   implementations and launch scripts are provided in
   [`cloud_services/service`](cloud_services/service).

5. Update [`cloud_services/config.yaml`](cloud_services/config.yaml) with the IP
   address and port of each service:

```yaml
services:
  owl_vit:
    scheme: http
    host: <OWL_VIT_SERVER_IP>
    port: <OWL_VIT_SERVER_PORT>
  sam:
    scheme: http
    host: <SAM_SERVER_IP>
    port: <SAM_SERVER_PORT>
  sd_dino:
    scheme: http
    host: <SD_DINO_SERVER_IP>
    port: <SD_DINO_SERVER_PORT>
```

6. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

For a single-image semantic keypoint extraction example, see the
[semantic keypoint guide](semantic_keypoints/README.md).

## Simulation benchmark evaluation

The simulation benchmark uses [SoftGym](https://github.com/Xingyu-Lin/softgym)
and requires a compiled PyFlex binding.
This repository includes a precompiled binding for Python 3.10 at
`PyFlex/bindings/build/pyflex.cpython-310-x86_64-linux-gnu.so`. If you use a
different Python version, compile PyFlex for that environment before running the
benchmark. See this [SoftGym installation and compilation
guide](https://danieltakeshi.github.io/2021/02/20/softgym/) for detailed
instructions.

Activate the `clasp` environment and configure the simulator paths:

```bash
conda activate clasp
source prepare_1.0.sh
```

### Command

The following command runs the complete perception, planning, simulation, and
evaluation pipeline for every case under `test_case`:

```bash
python inference.py --test-cases test_case --output eval_results
```

### Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--test-cases PATH` | `test_case` | Directory searched recursively for test-case pickles, or one specific `*_info.pkl` file. |
| `--output PATH` | `eval_results` | Root directory for evaluation artifacts. |
| `--task TASK` | All tasks | Evaluate only cases whose task name exactly matches `TASK`, such as `TshirtFold`. |
| `--resume` | Disabled | Keep completed results, remove incomplete result directories, and continue unfinished cases. |
| `--limit N` | No limit | Run at most `N` selected cases. With `--resume`, the limit is applied after completed cases are skipped. |

Currently, this repository supports 16 clothes manipulation tasks. The available
values for `--task` are:

```text
TowelFlatten    TowelFold    TowelHang    TowelPlace
TshirtFlatten   TshirtFold   TshirtHang   TshirtPlace
TrousersFlatten TrousersFold TrousersHang TrousersPlace
SkirtFlatten    SkirtFold    SkirtHang    SkirtPlace
```

### Examples

Evaluate only the `TshirtFold` task:

```bash
python inference.py --test-cases test_case --task TshirtFold \
  --output eval_results
```

Evaluate one test case by passing its pickle directly. The file below has case
ID `TshirtFold/24/0`:

```bash
python inference.py --test-cases test_case/TshirtFold/24_info.pkl \
  --output eval_results
```

Resume all unfinished cases in an existing output directory:

```bash
python inference.py --test-cases test_case --output eval_results --resume
```

### Results

Each evaluated case is placed in `success` or `fail` according to its task
metric:

```text
eval_results/
  <task>/
    success/
      <case_id>/
        initial.png
        final.png
        result.json
    fail/
      <case_id>/
        initial.png
        final.png
        failure_reason.txt
        result.json
```

`initial.png` and `final.png` show the scene before and after execution.
`result.json` records the instruction, generated plans, executed action count,
stopping condition, task metrics, errors, and artifact checksums. Failed cases
also include a concise explanation in `failure_reason.txt`.

After evaluation, the command prints the total number of cases, successes,
success rate, and errors, followed by the same statistics for each task.

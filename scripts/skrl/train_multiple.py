# sweep.py
import yaml, subprocess, tempfile, shutil
import itertools

BASE_CFG = "./source/TableTennis/TableTennis/tasks/direct/tabletennis/agents/skrl_ppo_cfg_simple.yaml"
OUTPUT_CFG = "./source/TableTennis/TableTennis/tasks/direct/tabletennis/agents/skrl_ppo_cfg_auto.yaml"
TASK = "Template-Tabletennis-Direct-v0"
TRAIN_SCRIPT = "./scripts/skrl/train.py"

# hyper-params to try
# rollout_values = [256, 375, 512]
rollout_values = [512]
lrs = [1e-4]
mini_batches_values = [16]
learning_epochs_val = [32]

entropy_loss_scales = [0.005, 0.01, 0.02]
ratio_clips = [0.05, 0.1, 0.2]
value_loss_scales = [0.5, 1, 2]
dir_name = "32envnew2"
# load once
with open(BASE_CFG) as f:
    base = yaml.safe_load(f)

i = 0
for entropy_loss_scale, ratio_clip, value_loss_scale in itertools.product(
    entropy_loss_scales, ratio_clips, value_loss_scales
):
    # if i==0:
    #     i+=1
    #     continue
    cfg = base.copy()
    cfg["agent"]["experiment"]["directory"] = dir_name
    cfg["agent"]["experiment"][
        "experiment_name"
    ] = f"{entropy_loss_scale=}_{ratio_clip=}_{value_loss_scale=}"

    cfg["agent"]["rollouts"] = rollout_values[0]
    cfg["agent"]["learning_rate"] = lrs[0]
    cfg["agent"]["mini_batches"] = mini_batches_values[0]
    cfg["agent"]["learning_epochs"] = learning_epochs_val[0]

    cfg["agent"]["entropy_loss_scale"] = entropy_loss_scale
    cfg["agent"]["ratio_clip"] = ratio_clip
    cfg["agent"]["value_loss_scale"] = value_loss_scale

    with open(OUTPUT_CFG, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    subprocess.run(["python", TRAIN_SCRIPT, f"--task={TASK}", "--headless"], check=True)

possibilities = [
    [32, 8, 5e-4, 0.999, 8],
    [32, 8, 5e-4, 0.999, 4],
    [32, 8, 5e-4, 0.99, 8],
    [32, 4, 1e-3, 0.999, 8],
    [32, 4, 1e-3, 0.999, 4],
    [32, 4, 1e-3, 0.99, 8],
    [64, 8, 5e-4, 0.999, 8],
    [64, 8, 5e-4, 0.999, 4],
    [64, 8, 5e-4, 0.99, 8],
    [128, 8, 5e-4, 0.999, 8],
    [128, 8, 5e-4, 0.999, 4],
    [128, 8, 5e-4, 0.99, 8],
    [32, 2, 5e-4, 0.999, 8],
    [32, 2, 5e-4, 0.999, 4],
    [32, 2, 5e-4, 0.99, 8],
]
# for r, mini_batches, lr, discount_factor, learning_epochs in possibilities:
#     cfg = base.copy()
#     cfg["agent"]["experiment"]["directory"] = dir_name
#     cfg["agent"]["experiment"][
#         "experiment_name"
#     ] = f"{r}_{lr}_{mini_batches}_{discount_factor}_{learning_epochs}"
#     cfg["agent"]["rollouts"] = r
#     cfg["agent"]["learning_rate"] = lr
#     cfg["agent"]["mini_batches"] = mini_batches
#     cfg["agent"]["discount_factor"] = discount_factor
#     cfg["agent"]["learning_epochs"] = learning_epochs

#     with open(OUTPUT_CFG, "w") as f:
#         yaml.dump(cfg, f, sort_keys=False)

#     print(f"/n=== rollouts={r}  lr={lr} ===")
#     subprocess.run(["python", TRAIN_SCRIPT, f"--task={TASK}", "--headless"], check=True)

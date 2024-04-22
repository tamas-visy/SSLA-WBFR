import subprocess

# import time
# import datetime
# hours = 1
# for i in range(hours * 60):
#     time.sleep(60)
#     print(i, datetime.datetime.now())
# print("Done")

task_name = "Fatigue"
pretrains = None

# task_name = "Fatigue"
# pretrains = [
#     "test_model.ckpt",
#     "test_model.ckpt",
#     "test_model.ckpt",
# ]

for i in range(3):
    seed = 2494 + i
    if pretrains is not None:
        extra = f"--model.pretrained_ckpt_path {pretrains[i]}"
    else:
        extra = ""

    c = f"python src/models/train.py fit -c configs/models/MyCNNtoTransformerClassifier.yaml -c configs/tasks/{task_name}.yaml -c configs/data_temporal_7_day.yaml -c configs/common.yaml --pl_seed {seed} {extra}"
    print(c)
    result = subprocess.run(c)  # , shell=True
    print(result)

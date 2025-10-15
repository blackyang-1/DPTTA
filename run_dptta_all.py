import subprocess

#code by black-y 2025.10.15 16:02
        
# Tuning Params
LR = [1,1,1,]
BETA1 = [1,1,1,]
BETA2 = [1,1,1,]
BS = [1]
Noise_level = [1]
for lr in LR:
    for bt1 in BETA1:
        for bt2 in BETA2:
          for bs in BS:
           for nl in Noise_level:
                subprocess.run(["python", "dptta.py",
                                "--lr", str(lr),
                                "--beta1", str(bt1),
                                "--beta2", str(bt2),
                                "--batch_size", str(bs),
                                "--noise_level", str(nl)])
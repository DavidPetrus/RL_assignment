# Reinforcement Learning - 2019 Assignment

# Environment
### Unity Obstacle Tower [[repo link](https://github.com/Unity-Technologies/obstacle-tower-env)]

# Packages
- torch
- torchvision
- gym
- matplotlib
- mlagents (v0.10.1)
- pillow
- opencv-python

# Create Conda Env from environment.yml

    conda env create -f environment.yml -n assignment

# Evaluation - MyAgent
- The MyAgent will be evaluated
- Initialize model within the `__init__` method
- Return action from `act(observation)` method

# Run Docker for Evaluation

- When initializing ObstacleTowerEvaluation env, set docker_training=True

        cd to_root_folder

        docker build -t assignment .

        docker run -it --rm assignment

# Record Agent
- Install ffmpeg [[Ubuntu](https://tecadmin.net/install-ffmpeg-on-linux/)] [[macOS](https://formulae.brew.sh/formula/ffmpeg#default)] [[Windows](https://www.wikihow.com/Install-FFmpeg-on-Windows)]

        python recorder.py
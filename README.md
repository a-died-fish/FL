# A Full Simulation Framework for Federated Learning Based on Cloud Services(基于云服务的联邦学习全仿真框架)
## Introduction
This repo is the homework of a course called "云计算技术" in Shanghai Jiaotong University, Shanghai, China. We made a simulation framework for federated learning based on cloud services, which greatly reduce the training time and GPU cost of federated learning. The framework is easy to extend to more federated algorithmns. If you have any questions, feel free to ask by raising an issue or email me at `zhuxinyu@sjtu.edu.cn`

## Setup
Clone the repo and install the required packages.

### Set the Environment
```
git clone https://github.com/a-died-fish/FL.git
cd FL
conda create -n py36 python=3.6
conda activate py36
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements_generation.txt
```

### Change ip, password and port
You should set you own client's information at `server.py` and set you server's information at `main.py`

### Change the directory to save and receive models
You should make and set the directory to save and receive models in both `server.py` and `main.py`.

## Run
Run the server first by (if you are a sever)
```
python server.py
```
Run the clients by (if you are a client)
```
sh run_scripts/example.sh
```



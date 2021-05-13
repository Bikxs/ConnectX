
#build docker image
docker build -t connectx/tf-gpu-jupyter .

#run docker jupyter notebook
docker run -it --rm --gpus all -v /home/bikxs/Kaggle/ConnectX:/tf -p 8888:8888 connectx/tf-gpu-jupyter:latest

#run docker with bash
docker run -it --rm --gpus all -v /home/bikxs/Kaggle/ConnectX:/tf -p 8888:8888 connectx/tf-gpu-jupyter:latest bash

#run docker and start training
docker run -it --rm --gpus all -v /home/bikxs/Kaggle/ConnectX:/tf -p 8888:8888 connectx/tf-gpu-jupyter:latest python strategy_deep_rl.py

#submit to kaggle
kaggle competitions submit -c connectx -f submission.py -m "Minimax with alpha-beta pruning with heuristics"
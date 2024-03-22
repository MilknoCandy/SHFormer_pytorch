###
 # @Description: create python environment for SHFormer_Pytorch Project
 # @version: 1.0
 # @Author: MilknoCandy
 # @Date: 2024-03-22 11:41:33
 # @LastEditTime: 2024-03-22 14:47:54
 # @FilePath: prepare_env
 # @Github: https://github.com/MilknoCandy
### 
# use command 'source' not 'sh'
conda create -n shformer_test python=3.8
conda activate shformer_test
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

python -m pip install -r requirements.txt
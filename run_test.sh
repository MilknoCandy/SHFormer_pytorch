###
 # @Description: test model with pretrained params
 # @version: 1.0
 # @Author: MilknoCandy
 # @Date: 2024-03-22 15:38:27
 # @LastEditTime: 2024-03-22 15:39:40
 # @FilePath: run_test
 # @Github: https://github.com/MilknoCandy
### 
python training/run_training.py --config configs/shformer_add.yaml \
    --pretrain_pth 'pretrained_params/shformer_add.pth
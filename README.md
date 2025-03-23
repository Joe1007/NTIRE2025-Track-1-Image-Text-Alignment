# NTIRE2025-Track-1-Image-Text-Alignment

1. Setup
---
The baseline code we use is from this [github](https://github.com/DYEvaLab/EvalMuse), so you should clone it first, then follow the steps to preprocessing the dataset, then run the code successfully .
Use `git clone https://github.com/DYEvaLab/EvalMuse`.  
And should download the final stage for [testing dataset](https://drive.google.com/file/d/1ZuCWg3-RLC8P82u6SbKtvGiWKt-BEHRT/view), and keep the path `EvalMuse/final_dataset`.  

2. Training Stage
---
Replace the `fga_blip2.py` with our `fga_blip2.py`, and run `train.sh` , you can reproduce the checkpoints under `EvalMuse\lavis\output\FGA-BLIP2`.  
(P.s. For convinienece, we offer the [checkpoints](https://drive.google.com/drive/u/0/folders/1fy_2KaHq_ygtSxB7FFzhwD3exjlsFDwP) we already trained before.  
Please choose `checkpoint8`, and you shouldn't run the code for training stage)

3. Inference Stage
---
If you get the checkpoints already, the you should modify the `main()` function of `eval.py`
Modify `parser.add_argument('--model_path', type=str, default='lavis/output/FGA-BLIP2/20250322203/checkpoint_8.pth')` 
Modify `parser.add_argument('--dataset_dir', type=str, default='dataset/images/')`-->`parser.add_argument('--dataset_dir', type=str, default='final_dataset/images/')`
Modify `parser.add_argument('--data_file', type=str, default='dataset/test.json')`-->`parser.add_argument('--data_file', type=str, default='final_dataset/test.json')`

And you should get the predicted .json under 'EvalMuse\results'

python train_deconv.py --savedir saved/train_deconv --sigma 7.65 --epochs 50 --bs 2 -tp 
python train_deconv_baseline.py --savedir saved/train_deconv_baseline --sigma 7.65 --epochs 50 --bs 2 -tp 


python train_deconv.py --savedir saved/train_deconv_div2k --sigma 7.65 --epochs 50 --bs 2 -tp --root data/div2k

python train_deconv_deq.py --savedir saved/train_deconv_deq --sigma 7.65 --epochs 50 --bs 2 -tp 
python train_deconv_joint_deq.py --savedir saved/train_deconv_joint_deq --sigma 7.65 --epochs 50 --bs 2 -tp 


python train_deconv_finetune.py --savedir saved/train_deconv_finetune --sigma 7.65 --epochs 50 --bs 2 -tp 


python train_deconv_joint.py --savedir saved/train_deconv_joint-train2 --sigma 7.65 --epochs 50 --bs 2 -tp 

python train_deconv_joint.py --savedir saved/train_deconv_joint_epoch_inter_train --sigma 7.65 --epochs 50 --bs 2 -tp 


python train_deconv.py --savedir saved/train_deconv_trainable_params --sigma 7.65 --epochs 50 --bs 2 -tp

python train_jd3.py --savedir saved/train_jd3 --sigma 7.65 --epochs 50 --bs 2 -tp

python train_jd3.py --savedir saved/train_jd3 --sigma 7.65 --epochs 50 --bs 2 -tp
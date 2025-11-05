# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

python3 bytetrack_inference.py \
image \
-f exps/bytetrack/yolox_x_mix_det.py \
-c exps/bytetrack/bytetrack_x_mot17.pth.tar \
--fp16 \
--fuse \
--save_result \
# --rmot_path "./datasets/refer-kitti \" #<- Open if wanted to test refer-kitti v1 
#&> log.txt 

# --resume exps/bytetrack/checkpoint0051.pth \
# --output_dir exps/default >"exps/bytetrack/test_log.txt" & echo $! >"exps/bytetrack/test_pid.txt"

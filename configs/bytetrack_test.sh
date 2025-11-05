# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

python3 bytetrack_inference.py \
-f exps/bytetrack/yolox_x_mix_det.py \
-c exps/bytetrack/bytetrack_x_mot17.pth.tar \
--fp16 \
--fuse \
--save_result \
&> log.txt \

# --resume exps/bytetrack/checkpoint0051.pth \
# --output_dir exps/default >"exps/bytetrack/test_log.txt" & echo $! >"exps/bytetrack/test_pid.txt"

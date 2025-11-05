import argparse
import os
import os.path as osp
import time
import cv2
import torch
import re
import json
import shutil
import subprocess
import sys
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as F

# --- Ensure project root on sys.path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import get_args_parser
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo")
    parser.add_argument("demo", default="image", help="demo type: image | video | webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # Path to images or video
    parser.add_argument("--path", default="./videos/palace.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam camera id")

    parser.add_argument("--save_result", action="store_true", help="save visualization results")

    # Exp / ckpt / device
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint for eval")
    parser.add_argument("--device", default="gpu", type=str, help="cpu | gpu")
    parser.add_argument("--conf", default=None, type=float, help="test confidence threshold")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test image size (square)")
    parser.add_argument("--fps", default=30, type=int, help="fallback FPS if video metadata missing")

    # Inference toggles
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="mixed precision")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="fuse conv+bn")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="use TensorRT")

    # Tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="frames to keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="filter overly tall boxes")
    parser.add_argument("--min_box_area", type=float, default=10, help="filter tiny boxes")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test MOT20 dataset")

    # end-to-end rmot settings
    parser.add_argument('--rmot_path', default='./datasets/refer-kitti-v2', type=str)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--exp_name', default='submit', type=str)
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1].lower()
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def load_label(label_path: str, img_size: tuple) -> dict:
    labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
    h, w = img_size
    # Normalized cewh to pixel xyxy format
    labels = labels0.copy()
    labels[:, 2] = w * (labels0[:, 2])
    labels[:, 3] = h * (labels0[:, 3])
    labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4])
    labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5])
    targets = {'boxes': [], 'labels': [], 'area': []}
    num_boxes = len(labels)

    visited_ids = set()
    for label in labels[:num_boxes]:
        obj_id = label[1]
        if obj_id in visited_ids:
            continue
        visited_ids.add(obj_id)
        targets['boxes'].append(label[2:6].tolist())
        targets['area'].append(label[4] * label[5])
        targets['labels'].append(0)
    targets['boxes'] = np.asarray(targets['boxes'])
    targets['area'] = np.asarray(targets['area'])
    targets['labels'] = np.asarray(targets['labels'])
    return targets

# New Added
class ListImgDataset(Dataset):
    def __init__(self, img_list, bytetrack=False) -> None:
        super().__init__()
        self.img_list = img_list
        self.bytetrack = bytetrack

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        label_path = f_path.replace('training', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
        # print(label_path)
        cur_img = cv2.imread(f_path)
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        targets = load_label(label_path, cur_img.shape[:2]) if os.path.exists(label_path) else None
        # img = draw_bboxes(torch.tensor(cur_img), targets['boxes'])
        return cur_img, targets

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img, targets = self.load_img_from_file(img_path)
        processed_img, ori_img = self.init_img(img)
        return processed_img, ori_img, img_path
    
class Detector(object):
    def __init__(self,
            args,
            model,
            exp,
            trt_file=None,
            decoder=None,
            device=torch.device("cpu"),
            fp16=False,
            seq_num=None
        ):
        # TODO: modify model to bytetrack
        
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.seq_num = seq_num
        self.args=args
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225) 
        
        img_list = os.listdir(os.path.join(self.args.rmot_path, 'KITTI/training/image_02', self.seq_num[0]))
        img_list = [os.path.join(self.args.rmot_path, 'KITTI/training/image_02', self.seq_num[0], _)
                    for _ in img_list if ('jpg' in _) or ('png' in _)]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.json_path = os.path.join(self.args.rmot_path, 'expression', seq_num[0], seq_num[1])
        with open(self.json_path, 'r') as f:
            json_info = json.load(f)
        self.json_info = json_info
        self.sentence = [json_info['sentence']]

        # TODO: modify tr_tracker to bytetrack
        # self.tr_tracker = TransRMOT()
        # self.tr_tracker_withouRef = TransRMOT()
        checkpoint_id = 51
        self.save_path = os.path.join(self.args.output_dir,
                                      'results_epoch{}/{}/{}'.format(checkpoint_id, seq_num[0], seq_num[1].split('.')[0]))
        os.makedirs(self.save_path, exist_ok=True)

        self.predict_path = os.path.join(self.args.output_dir, self.args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_path, f'{self.seq_num}.txt')):
            os.remove(os.path.join(self.predict_path, f'{self.seq_num}.txt'))

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
    

        return outputs, img_info

    def image_demo(self,predictor, vis_folder, current_time, args):
        if osp.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        
        
        loader = DataLoader(ListImgDataset(self.img_list, bytetrack=True), 1,num_workers=2)
        tracker = BYTETracker(args, frame_rate=args.fps)
        timer = Timer()
        results = []
        tracker_outputs=[]
        for frame_id, (cur_img, ori_img, img_path) in enumerate(tqdm(loader)):
            cur_img, ori_img, img_path = cur_img[0], ori_img[0], img_path[0]
            seq_h, seq_w, _ = ori_img.shape
            
            outputs, img_info = predictor.inference(img_path, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs, online_ids, online_scores = [], [], []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / max(1e-5, timer.average_time)
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']

            if args.save_result:
                timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                save_folder = osp.join(vis_folder, timestamp)
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        if args.save_result:
            res_file = osp.join(vis_folder, f"{timestamp}.txt")
            with open(res_file, 'w', encoding='utf-8') as f:
                f.writelines(results)
            
        self.write_results_bytetrack(
            txt_path = os.path.join(self.save_path, 'predict.txt'),
            results=results
        ) 
        gt_path = os.path.join(self.save_path, 'gt.txt')
        self.write_gt(gt_path, self.json_path,
                      os.path.join(self.args.rmot_path, 'KITTI/labels_with_ids/image_02', self.seq_num[0]), seq_h, seq_w)
        
    def write_results_bytetrack(self, txt_path, results):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,1,1\n'
        # Results are formatted strings like "frame_id,tid,x,y,w,h,score,-1,-1,-1\n"
        # We need to parse and reformat to "frame,id,x1,y1,w,h,1,1,1\n"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for result in results:
                # Parse the CSV string
                parts = result.strip().split(',')
                if len(parts) >= 6:
                    frame_id, tid, x, y, w, h = parts[0:6]
                    line = save_format.format(frame=int(frame_id)+1, id=int(tid), 
                                             x1=float(x), y1=float(y), w=float(w), h=float(h))
                    f.write(line)

    # write ground-truth for each expression in a text. The text includes gt of all frames
    def write_gt(self, txt_path, json_file, gt_txt_file, im_height, im_width):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1, 1, 1\n'

        with open(json_file) as f:
            json_info = json.load(f)

        with open(txt_path, 'w') as f:
            for k in json_info['label'].keys():
                frame_id = int(k)
                if not os.path.isfile(os.path.join(gt_txt_file, '{:06d}.txt'.format(frame_id))):
                    continue
                frame_gt = np.loadtxt(
                    os.path.join(gt_txt_file, '{:06d}.txt'.format(frame_id))).reshape(-1, 6)
                for frame_gt_line in frame_gt:
                    aa = json_info['label'][k]  # all gt from frame
                    aa = [int(a) for a in aa]
                    if int(frame_gt_line[1]) in aa:  # choose referent gt from all gt
                        track_id = int(frame_gt_line[1])
                        x1, y1, w, h = frame_gt_line[2:6] # KITTI -> [x1, y1, w, h]
                        line = save_format.format(frame=frame_id+1, id=track_id, x1=x1 * im_width, y1=y1 * im_height,
                                                w=w * im_width, h=h * im_height)
                        f.write(line)

        print('save gt to {}'.format(txt_path))



def _extract_last_json(text: str, fallback=-1) -> int:
    """
    Extract the last JSON object in streaming text and return target_id.
    Expected shape: {"target_id": <int>}
    """
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    if not matches:
        return fallback
    for candidate in reversed(matches):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "target_id" in obj:
                return int(obj.get("target_id", fallback))
        except Exception:
            continue
    return fallback


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_meta = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_meta if (fps_meta and 0 < fps_meta <= 120) else args.fps

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    base_name = osp.splitext(osp.basename(args.path if args.demo == "video" else "webcam"))[0]
    avi_path = osp.join(save_folder, base_name + ".avi")
    mp4_path = osp.join(save_folder, base_name + ".mp4")

    logger.info(f"video temp_save_path is {avi_path}")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vid_writer = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    tracker = BYTETracker(args, frame_rate=fps)
    timer = Timer()
    frame_id = 0
    results = []

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        outputs, img_info = predictor.inference(frame, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs, online_ids = [], []

            # --- On first frame (and no pre-selected target), crop and call LLM to pick target_id ---
            if frame_id == 0 and args.target_id == -1:
                os.makedirs("crops", exist_ok=True)
                for t in online_targets:
                    x1, y1, w, h = t.tlwh
                    x2, y2 = int(x1 + w), int(y1 + h)
                    crop = img_info['raw_img'][int(y1):int(y2), int(x1):int(x2)]
                    save_path = f"crops/i{t.track_id}_f{frame_id}.jpg"
                    cv2.imwrite(save_path, crop)

                try:
                    raw = subprocess.check_output(
                        ["python", "LLM.py", "--crops", "crops", "--prompt", args.prompt],
                        stderr=subprocess.STDOUT
                    )
                    text = raw.decode("utf-8", errors="ignore")
                    print("=== LLM Raw Output ===")
                    print(text)
                    args.target_id = _extract_last_json(text, fallback=-1)
                except subprocess.CalledProcessError as e:
                    logger.warning("LLM.py returned non-zero exit, proceeding without target filter.")
                    try:
                        text = e.output.decode("utf-8", errors="ignore")
                        print("=== LLM Raw Output (error path) ===")
                        print(text)
                        args.target_id = _extract_last_json(text, fallback=-1)
                    except Exception:
                        args.target_id = -1
                except Exception as e:
                    logger.warning(f"LLM invocation failed: {e}")
                    args.target_id = -1

                logger.info(f"LLM response target_id = {args.target_id}")

            # --- Keep only the selected target_id if provided ---
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                # 支援 allow_ids 多目標
                if args.allow_ids:
                    allow_ids = set(int(x) for x in args.allow_ids.split(",") if x.strip())
                    if tid not in allow_ids:
                        continue
                elif args.target_id != -1:
                    if tid != args.target_id:
                        continue


                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids,
                frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
            )
        else:
            online_im = img_info['raw_img']

        if args.save_result:
            vid_writer.write(online_im)

        frame_id += 1

    cap.release()
    vid_writer.release()

    if args.save_result:
        # Save txt results
        res_file = osp.join(save_folder, f"{timestamp}.txt")
        with open(res_file, 'w', encoding='utf-8') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

        # Convert avi -> mp4 using ffmpeg
        logger.info(f"Converting {avi_path} to {mp4_path} using ffmpeg...")

        # Prefer PATH ffmpeg; else allow override via FFMPEG_BIN
        ffmpeg_bin = shutil.which("ffmpeg") or os.environ.get("FFMPEG_BIN")
        if not ffmpeg_bin or not osp.exists(ffmpeg_bin):
            logger.warning(
                "ffmpeg not found in PATH. Set env var FFMPEG_BIN to your ffmpeg.exe path, e.g.\n"
                'set FFMPEG_BIN=D:\\CSProject\\ByteTrackLLM_HOTA\\ffmpeg-8.0-full_build\\bin\\ffmpeg.exe'
            )
            # Still try PATH name to keep old behavior; may fail silently on some shells
            ffmpeg_bin = "ffmpeg"

        try:
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", avi_path, "-c:v", "libx264", "-r", str(int(fps)), mp4_path],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
        finally:
            logger.info(f"final video saved at {mp4_path}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt if args.ckpt is not None else osp.join(output_dir, "best_ckpt.pth.tar")
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()

    if args.trt:
        assert not args.fuse, "TensorRT model does not support fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(trt_file), "TensorRT model not found! Run tools/trt.py first."
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT for inference")
    else:
        trt_file = None
        decoder = None

    current_time = time.localtime()
    if args.demo == "image":
        for seq_num in seq_nums:
            predictor = Detector(
                args=args,
                model=model,
                exp=exp,
                trt_file=trt_file, 
                decoder=decoder, 
                device=args.device,
                fp16= args.fp16,
                seq_num=seq_num
            )
            predictor.image_demo(predictor, vis_folder, current_time, args)
    # elif args.demo in ("video", "webcam"):
    #     imageflow_demo(predictor, vis_folder, current_time, args)
        
    
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    
    expressions_root = os.path.join(args.rmot_path, 'expression')
    if "refer-kitti-v2" in args.rmot_path:
        video_ids = ['0005', '0011', '0013','0019']
    else:
        video_ids = ['0005', '0011', '0013']

    seq_nums = [] 
    for video_id in video_ids:  
        expression_jsons = sorted(os.listdir(os.path.join(expressions_root, video_id)))
        for expression_json in expression_jsons:
            seq_nums.append([video_id, expression_json])

    expression_num = len(seq_nums)
    
    print("Start inference")
    main(exp=exp, args=args)
    
    
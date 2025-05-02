import argparse
import csv
import os
import platform
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import *
from tkinter import filedialog

import cv2
import torch
from PIL import Image, ImageTk

LOG_LINE_NUM = 0

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


# 检测函数
@smart_inference_mode()
def run(
        weights=ROOT / "fall.pt",  # model path or triton URL
        source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / "fall_detection.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # 判断数据源类型
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # 目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 加载模型 -> 根目录fall.pt
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # 数据加载
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
        write_log_to_text("长按q键结束摄像头检测")
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # 创建或追加到 CSV 文件
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # 过程预测
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化处理
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 将框从 img_size 重新调整为 im0 大小
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # 写入文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # 流式传输结果
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

                cv2image1 = cv2.cvtColor(imc, cv2.COLOR_BGR2RGBA)  # 转换颜色格式
                img1 = Image.fromarray(cv2image1)  # 转换为PIL可用的图片格式
                img1 = img1.resize((400, 300), Image.LANCZOS)  # 调整图片大小
                photo1 = ImageTk.PhotoImage(img1)  # 转换为Tkinter可用的图片格式
                raw_data.configure(image=photo1)  # 更新Label显示的图片
                raw_data.image = photo1
                raw_data.update()  # 更新GUI界面
                if webcam:
                    # cv2.imshow(str(p), im0) #显示图像
                    cv2.imshow("Detection window", im0)  # 显示图像
                    cv2.waitKey(1)  # 1 millisecond
                else:
                    # 监测数据
                    cv2image2 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGBA)  # 转换颜色格式
                    img2 = Image.fromarray(cv2image2)  # 转换为PIL可用的图片格式
                    img2 = img2.resize((400, 300), Image.LANCZOS)  # 调整图片大小
                    photo2 = ImageTk.PhotoImage(img2)  # 转换为Tkinter可用的图片格式
                    process_results.configure(image=photo2)  # 更新Label显示的图片
                    process_results.image = photo2
                    process_results.update()  # 更新GUI界面

            # 保存结果（带有检测的图像）
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
            write_log_to_text('检测到跌倒' if len(det) else '')
        # 打印日志（仅推理）
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 打印结果
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # 更新模型（用于修复 SourceChangeWarning）

    # print(increment_path(Path(project)))

    # print(os.path.abspath(save_dir))
    # print(save_path)
    # return Path(project)
    return os.path.abspath(save_dir)


def parse_opt_source(source):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "fall.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=source, help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "fall_detection.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", default="true", action="store_true",
                        help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def detect(source):
    opt = parse_opt_source(source)
    abspath = run(**vars(opt))
    return abspath


# 选择文件
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path=='':
        return
    file_name = file_path.split("/")[-1]
    path = detect(file_path)
    set_image_by_path(file_path, raw_data)
    set_image_by_path(path + "/" + file_name, process_results)
    write_log_to_text("检测完成！文件目录：" + path + "\\" + file_name)


# 拍照
def take_picture():
    ret, frame = cap.read()  # 读取摄像头数据
    set_image(frame, raw_data)

    cv2.imwrite("temp.jpg", frame)
    path = detect("temp.jpg")
    set_image_by_path(path + "/temp.jpg", process_results)
    write_log_to_text("检测完成！文件目录：" + path + "\\temp.jpg")


# 根据路径设置图片
def set_image_by_path(file_path, label):
    img = Image.open(file_path)
    # cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)  # 转换颜色格式
    # img = Image.fromarray(cv2image)  # 转换为PIL可用的图片格式
    img = img.resize((400, 300), Image.LANCZOS)  # 调整图片大小
    photo = ImageTk.PhotoImage(img)  # 转换为Tkinter可用的图片格式
    label.configure(image=photo)  # 更新Label显示的图片
    label.image = photo


# 设置图片
def set_image(img, label):
    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色格式
    img = Image.fromarray(cv2image)  # 转换为PIL可用的图片格式
    img = img.resize((400, 300), Image.LANCZOS)  # 调整图片大小
    photo = ImageTk.PhotoImage(img)  # 转换为Tkinter可用的图片格式
    label.configure(image=photo)  # 更新Label显示的图片
    label.image = photo


# 相机检测
def camera_detection():
    detect(0)
    write_log_to_text("检测完成！文件目录：" + "E:\\Code\\PythonCode\\yolov5-full-detect\\runs\\detect\\exp")


# 退出
def esc():
    sys.exit(0)


# 记录日志
def write_log_to_text(logMsg):
    if ''==logMsg:
        return
    global LOG_LINE_NUM
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    logmsg_in = str(current_time) + " " + str(logMsg) + "\n"  # 换行
    if LOG_LINE_NUM <= 30:  # 最大行数30
        log_data_label.insert(END, logmsg_in)
        LOG_LINE_NUM = LOG_LINE_NUM + 1
    else:
        log_data_label.delete(1.0, 2.0)
        log_data_label.insert(END, logmsg_in)

# 主界面
if __name__ == '__main__':
    # 创建主窗口
    root = tk.Tk()
    root.title("Fall detect")
    root.geometry("1000x600")
    cap = cv2.VideoCapture(0)

    # 创建左侧工具栏
    tool_frame = tk.Frame(root, width=160, height=600, bg='lightgrey')
    tool_frame.pack(side=tk.LEFT, fill=tk.Y)
    # 添加按钮
    # 按钮-打开摄像头-拍照
    open_button1 = tk.Button(tool_frame, text="选择文件", command=select_file)
    open_button1.pack()
    # 按钮-打开摄像头-拍照
    open_button2 = tk.Button(tool_frame, text="拍照", command=take_picture)
    open_button2.pack()

    # 按钮-打开摄像头-检测
    open_button3 = tk.Button(tool_frame, text="摄像头-检测", command=camera_detection)
    open_button3.pack()

    # 按钮-退出程序
    open_button4 = tk.Button(tool_frame, text="退出", command=esc)
    open_button4.pack(side=tk.BOTTOM)

    # 创建右侧区域
    right_frame = tk.Frame(root, width=640, height=600)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH)

    # 创建上部分展示区域
    display_frame = tk.Frame(right_frame, width=640, height=450)
    display_frame.pack(side=tk.TOP, fill=tk.BOTH)

    # 创建图片展示区
    raw_data = tk.Label(display_frame, text="原数据")
    raw_data.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 创建视频展示区
    process_results = tk.Label(display_frame, text="处理结果")
    process_results.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # 创建下部分日志区域
    log_frame = tk.Frame(right_frame, width=640, height=350)
    log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH)

    # 创建日志显示框
    log_data_title = tk.Label(log_frame, text="日志")
    log_data_title.pack(fill=tk.BOTH)

    # 日志框
    log_data_label = tk.Text(log_frame)
    log_data_label.pack(fill=tk.BOTH)

    root.mainloop()

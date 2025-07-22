import os
import cv2
import subprocess
from uuid import uuid4
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from ultralytics import YOLO
import torch

# Путь к локальному шрифту
FONT_PATH = "DejaVuSans.ttf"
# if not FONT_PATH.exists():
#     raise FileNotFoundError(f"[ERROR] Шрифт DejaVuSans.ttf не найден: {FONT_PATH}")

# Функции для детекции

def has_audio_stream(video_path: str) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-loglevel", "error",
             "-select_streams", "a",
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return 'audio' in result.stdout
    except:
        return False


def wrap_text(text, font, max_width):
    words, lines, line = text.split(), [], ''
    for word in words:
        test = f"{line} {word}".strip()
        if font.getbbox(test)[2] <= max_width:
            line = test
        else:
            if line: lines.append(line)
            line = word
    if line: lines.append(line)
    return lines


def draw_legend(img_np, dets, font_size, min_font_size=10):
    h, w = img_np.shape[:2]
    legend_w, pad = 400, 10
    max_text_w = legend_w - 2*pad
    size = font_size
    while size >= min_font_size:
        font = ImageFont.truetype(str(FONT_PATH), size)
        line_h = size + 6
        lines = []
        for i, (lbl, conf) in enumerate(dets):
            lines.extend(wrap_text(f"{i+1}. {lbl} ({conf:.2f})", font, max_text_w))
        if len(lines)*line_h + 2*pad <= h:
            break
        size -= 1
    else:
        font = ImageFont.truetype(str(FONT_PATH), min_font_size)
        line_h = min_font_size + 6
        lines = [l for i,(lbl,conf) in enumerate(dets) for l in wrap_text(f"{i+1}. {lbl} ({conf:.2f})", font, max_text_w)]
    canvas_h = max(h, len(lines)*line_h + 2*pad)
    canvas = Image.new('RGB', (w+legend_w, canvas_h), (255,255,255))
    canvas.paste(Image.fromarray(img_np), (0,0))
    draw = ImageDraw.Draw(canvas)
    y = pad
    for line in lines:
        draw.text((w+pad, y), line, fill='black', font=font)
        y += line_h
    return np.array(canvas)


def process_frame(img_pil, model, conf_thresh, thickness, font_size):
    thickness = int(thickness)
    font_size = int(font_size)
    img_np = np.array(img_pil)
    results = model(img_np, conf=conf_thresh, verbose=False)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(str(FONT_PATH), font_size)
    except:
        font = ImageFont.load_default()
    dets = []
    for idx, box in enumerate(results[0].boxes):
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        cid,conf = int(box.cls[0]), float(box.conf[0])
        lbl = model.model.names.get(cid, f"Class {cid}")
        draw.rectangle([x1,y1,x2,y2], outline='red', width=thickness)
        txt = str(idx+1)
        tb = draw.textbbox((0,0), txt, font=font)
        tw,th = tb[2]-tb[0], tb[3]-tb[1]
        draw.rectangle([x1, y1-th-4, x1+tw+4, y1], fill='red')
        draw.text((x1+2, y1-th-2), txt, fill='white', font=font)
        dets.append((lbl, conf))
    return draw_legend(np.array(img_pil), dets, font_size)


def detect_objects(model, source, is_video, conf_thresh, thickness, font_size, save, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")
    if is_video:
        cap = cv2.VideoCapture(str(src))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret: raise RuntimeError("Cannot read video")
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ann0 = process_frame(pil, model, conf_thresh, thickness, font_size)
        h0, w0 = ann0.shape[:2]
        raw = Path(out_dir)/f"raw_{uuid4().hex}.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w0,h0))
        writer.write(cv2.cvtColor(ann0, cv2.COLOR_RGB2BGR))
        for _ in range(total-1):
            ret, frame = cap.read()
            if not ret: break
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ann = process_frame(pil, model, conf_thresh, thickness, font_size)
            writer.write(cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
        cap.release(); writer.release()
        final = Path(out_dir)/f"annot_{src.stem}.mp4"
        if has_audio_stream(str(src)):
            cmd = ['ffmpeg','-y',
                   '-i',str(raw),
                   '-i',str(src),
                   '-c:v','libx264',
                   '-pix_fmt','yuv420p',
                   '-c:a','aac','-map',
                   '0:v','-map','1:a',
                   '-shortest',
                   '-movflags','+faststart',
                   str(final)
                ]
        else:
            cmd = ['ffmpeg','-y',
                   '-i',str(raw),
                   '-f','lavfi',
                   '-i','anullsrc=channel_layout=stereo:sample_rate=44100',
                   '-c:v','libx264',
                   '-pix_fmt','yuv420p',
                   '-preset','fast','-c:a',
                   'aac','-shortest',
                   '-movflags','+faststart',
                   str(final)
                ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw.unlink()
        if save: return str(final)
        else: return str(final)
    else:
        pil = Image.open(str(src)).convert('RGB')
        ann = process_frame(pil, model, conf_thresh, thickness, font_size)
        out_file = Path(out_dir)/f"annot_{src.name}"
        if save:
            cv2.imwrite(str(out_file), cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
            return str(out_file)
        return ann

# Загружаем модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("RTSD_YOLO11s/yolo11s/weights/best.pt")
model.to(device)


css = """
.gradio-container {
    max-width: 80% !important;
    margin: auto;
}

.tab-content {
    max-width: 1000px;
    margin: auto;
    padding: 20px;
}

.fixed-size {
    width: 480px !important;
    height: 300px !important;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid #ccc;
}
.fixed-size img, .fixed-size video {
    max-width: 100% !important;
    max-height: 100% !important;
    object-fit: contain;
}
.fixed-save {
    width: 480px !important;
    height: 80px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #ccc;
    overflow: hidden;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Traffic Sign Detection With YOLO")
    with gr.Tabs():
        with gr.TabItem("Readme"):
            with gr.Column(elem_classes="tab-content"):
                with open("README_FOR_APP.md", encoding="utf-8") as f:
                    gr.Markdown(f.read())
        
        with gr.TabItem("Image"):
            with gr.Column(elem_classes="tab-content"):
                with gr.Row():
                    with gr.Column(elem_classes="fixed-size"):
                        img_input = gr.Image(label="Upload Image", type="filepath")
                    with gr.Column(elem_classes="fixed-size"):
                        img_preview = gr.Image(label="Preview")
            ex_img = gr.Examples(
                examples=["example_3.jpg"],
                inputs=[img_input],
                label="Example Images"
            )
            thr_i = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Confidence Threshold")
            thickness_i = gr.Slider(1, 10, value=2, step=1, label="Box Thickness")
            font_i = gr.Slider(8, 72, value=18, step=1, label="Font Size")
            with gr.Column(elem_classes="fixed-save"):
                img_file = gr.File(label="Save Image File")
            with gr.Row():
                btn_preview_i = gr.Button("Run Image")
                btn_save_i = gr.Button("Save Image")
                btn_clear_i = gr.Button("Clear")
        with gr.TabItem("Video"):
            with gr.Column(elem_classes="tab-content"):
                with gr.Row():
                    with gr.Column(elem_classes="fixed-size"):
                        vid_input = gr.Video(label="Upload Video")
                    with gr.Column(elem_classes="fixed-size"):
                        vid_preview = gr.Video(label="Preview")
            ex_vid = gr.Examples(
                examples=["mini_demo_1.mp4"],
                inputs=[vid_input],
                label="Example Videos"
            )
            thr_v = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Confidence Threshold")
            thickness_v = gr.Slider(1, 10, value=2, step=1, label="Box Thickness")
            font_v = gr.Slider(8, 72, value=18, step=1, label="Font Size")
            with gr.Column(elem_classes="fixed-save"):
                vid_file = gr.File(label="Save Video File")
            with gr.Row():
                btn_preview_v = gr.Button("Run Video")
                btn_save_v = gr.Button("Save Video")
                btn_clear_v = gr.Button("Clear")

    # Callbacks
    btn_preview_i.click(lambda f, t, th, fs: detect_objects(model, f, False, t, th, fs, False, 'output'),
                        [img_input, thr_i, thickness_i, font_i], img_preview)
    btn_save_i.click(lambda f, t, th, fs: detect_objects(model, f, False, t, th, fs, True, 'output'),
                     [img_input, thr_i, thickness_i, font_i], img_file)
    btn_clear_i.click(lambda: (None, 0.5, 2, 18, None, None), [], 
                      [img_input, thr_i, thickness_i, font_i, img_preview, img_file])

    btn_preview_v.click(lambda f, t, th, fs: detect_objects(model, f, True, t, th, fs, False, 'output'),
                        [vid_input, thr_v, thickness_v, font_v], vid_preview)
    btn_save_v.click(lambda f, t, th, fs: detect_objects(model, f, True, t, th, fs, True, 'output'),
                     [vid_input, thr_v, thickness_v, font_v], vid_file)
    btn_clear_v.click(lambda: (None, 0.5, 2, 18, None, None), [],
                      [vid_input, thr_v, thickness_v, font_v, vid_preview, vid_file])
    
if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860, share=True)

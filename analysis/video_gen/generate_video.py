import cv2
import numpy as np
import random
from pathlib import Path

# ==========================
# Konfiguration
# ==========================
FRAME_SIZE = (1280, 720)         # (Breite, Höhe)
FPS = 30
DISPLAY_TIME_S = 2.0             # Dauer pro Schild
GRID = (10, 10)                  # Hintergrund-Kacheln (Spalten, Zeilen)
COLOR_RANGE = ((30, 30, 30), (180, 180, 180))  # RGB-Unter-/Obergrenzen
PATH_IMAGES = Path("analysis/video_gen/images")
PATH_OUTPUT = Path("analysis/video_gen/output")
OUTFILE = "generated_video.mp4"
CODEC = "mp4v"
ORDER = "shuffle"                # 'shuffle' oder 'sequential'
MAX_IMAGES = None                # z.B. 20 für Test
SEED = 42
FADE_IN_OUT = True
SOLID_BG = True
BG_COLOR = (255, 255, 255)  # Weiß (B,G,R in OpenCV)

random.seed(SEED)
PATH_OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================
# Hilfsfunktionen
# ==========================
def generate_background(size, grid, color_range):
    if SOLID_BG:
        w, h = size
        return np.full((h, w, 3), BG_COLOR, dtype=np.uint8)

    w, h = size
    cols, rows = grid
    tile_w, tile_h = w // cols, h // rows
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(rows):
        for x in range(cols):
            color = [random.randint(color_range[0][c], color_range[1][c]) for c in range(3)]
            bg[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w] = color
    return bg

def overlay_image(background, img, x, y):
    """Kopiert img auf background an Position (x,y), sicher bei Randaustritt."""
    h, w = img.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background  # komplett außerhalb

    x1_bg = max(x, 0)
    y1_bg = max(y, 0)
    x2_bg = min(x + w, bg_w)
    y2_bg = min(y + h, bg_h)

    x1_img = x1_bg - x
    y1_img = y1_bg - y
    x2_img = x1_img + (x2_bg - x1_bg)
    y2_img = y1_img + (y2_bg - y1_bg)

    background[y1_bg:y2_bg, x1_bg:x2_bg] = img[y1_img:y2_img, x1_img:x2_img]
    return background


# ==========================
# Hauptlogik
# ==========================
def main():
    images = list(PATH_IMAGES.glob("*.*"))
    if ORDER == "shuffle":
        random.shuffle(images)
    if MAX_IMAGES:
        images = images[:MAX_IMAGES]

    num_frames_per_image = int(DISPLAY_TIME_S * FPS)
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    out_path = PATH_OUTPUT / OUTFILE
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, FRAME_SIZE)

    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Überspringe ungültiges Bild: {img_path}")
            continue
        ih, iw = img.shape[:2]
        bg = generate_background(FRAME_SIZE, GRID, COLOR_RANGE)

        # zufällige Bewegung
        start_x = random.randint(-iw//2, FRAME_SIZE[0] - iw//2)
        start_y = random.randint(-ih//2, FRAME_SIZE[1] - ih//2)
        dx = random.uniform(-200, 200) / FPS
        dy = random.uniform(-200, 200) / FPS

        for f in range(num_frames_per_image):
            frame = bg.copy()
            x = int(start_x + dx * f)
            y = int(start_y + dy * f)

            # optionale Fade-In/Out
            if FADE_IN_OUT:
                alpha = min(1.0, max(0.0, f / (FPS*0.3)))  # Fade-In 0.3 s
                beta = min(1.0, max(0.0, (num_frames_per_image - f) / (FPS*0.3)))
                fade = min(alpha, beta)
                img_mod = (img.astype(np.float32) * fade).astype(np.uint8)
            else:
                img_mod = img

            frame = overlay_image(frame, img_mod, x, y)
            writer.write(frame)

        print(f"[{idx+1}/{len(images)}] verarbeitet: {img_path.name}")

    writer.release()
    print(f"Fertig! Gespeichert unter {out_path}")

if __name__ == "__main__":
    main()

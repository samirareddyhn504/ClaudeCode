"""Video de prueba: '배 (bae)' estilo plastilina, jerga peruana. 1080x1920, 30fps."""
import subprocess, wave, os, sys, math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "video_bae.mp4"
TMP = os.path.join(os.path.dirname(os.path.abspath(OUT)), "_tmp")
os.makedirs(TMP, exist_ok=True)
W, H, FPS, SR = 1080, 1920, 30, 22050
rng = np.random.default_rng(7)
random.seed(7)

# ---------------- AUDIO ----------------
def tts(text, voice="es-419", speed=150, pitch=55, name="x"):
    p = os.path.join(TMP, name + ".wav")
    subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch), "-w", p, text], check=True)
    with wave.open(p) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        if w.getframerate() != SR:
            a = np.interp(np.arange(0, len(a), w.getframerate() / SR), np.arange(len(a)), a)
    return a

def tone(freqs, dur=0.12):
    out = []
    for f in freqs:
        t = np.arange(int(SR * dur)) / SR
        env = np.exp(-t * 9)
        out.append((np.sin(2*np.pi*f*t) + 0.4*np.sin(4*np.pi*f*t)) * env * 0.35)
    return np.concatenate(out)

def sil(s): return np.zeros(int(SR * s), np.float32)

# Guion: (tipo, texto pantalla, texto voz, voz, objeto, romanización)
S = [
    ("intro", "¡HABLA, CAUSA!\n¿Cómo se dice...\nen coreano?", "¡Habla, causa! ¿Cómo se dice... en coreano?", None, "mascot", ""),
    ("q", "¿PERA?", "¿Pera?", None, "pear", ""),
    ("a", "배", "배", None, "pear", "bae"),
    ("q", "¿Y BOTE, PE?", "¿Y bote, pe?", None, "boat", ""),
    ("a", "배", "배", None, "boat", "bae"),
    ("q", "¿Y LA PANZA?", "¿Y la panza?", None, "belly", ""),
    ("a", "배", "배", None, "belly", "bae"),
    ("q", "¿EL DOBLE?", "¿El doble?", None, "two", ""),
    ("a", "두 배", "두 배", None, "two", "du bae"),
    ("q", "¿ESTOY\nLLENAZO?", "¿Estoy llenazo?", None, "full", ""),
    ("a", "배불러", "배불러", None, "full", "bae-bul-leo"),
    ("fast", "¡Subí al 배,|comí mi 배,|mi 배 creció 두 배...|배불러, causa,|¡ya no entro ni al micro!",
     "¡Subí al be, comí mi be, mi be creció du be! ¡Be bul ló, causa, ya no entro ni al micro!", None, "full", ""),
    ("end", "Asu...\n¡todo es 배!", "Asu... ¡todo es be!", None, "mascot", ""),
]

voice = []; events = []; t = 0.0
for i, (kind, scr, say, _, obj, rom) in enumerate(S):
    if kind == "a":
        sting = tone([523, 659, 784])
        v = tts(say, "ko", 120, 75, f"s{i}")
        seg = np.concatenate([sting, sil(0.05), v, sil(0.35)])
        vstart = len(sting) / SR + 0.05
    elif kind == "fast":
        v = tts(say, "es-419", 235, 70, f"s{i}")
        seg = np.concatenate([v, sil(0.3)]); vstart = 0
    else:
        v = tts(say, "es-419", 150, 55, f"s{i}")
        seg = np.concatenate([v, sil(0.25)]); vstart = 0
    events.append(dict(kind=kind, scr=scr, obj=obj, rom=rom, t0=t, t1=t + len(seg)/SR, vs=t+vstart))
    voice.append(seg); t += len(seg)/SR
voice = np.concatenate([sil(0.3), np.concatenate(voice), sil(0.6)])
for e in events:
    for k in ("t0", "t1", "vs"): e[k] += 0.3
DUR = len(voice) / SR

# Pista de ritmo (bombo + hi-hat), más rápida en la parte "fast"
beat = np.zeros_like(voice)
fast = next(e for e in events if e["kind"] == "fast")
def kick(): tt = np.arange(int(SR*.15))/SR; return np.sin(2*np.pi*(60+90*np.exp(-tt*30))*tt)*np.exp(-tt*18)*.5
def hat(): n = int(SR*.04); return rng.normal(0, 1, n)*np.exp(-np.arange(n)/SR*90)*.08
tb = 0.3
while tb < DUR - 0.3:
    bpm = 150 if fast["t0"] <= tb < fast["t1"] else 100
    step = 60 / bpm / 2
    i0 = int(tb*SR); k = kick(); h = hat()
    n = int(round((tb - 0.3) / step))
    if n % 2 == 0: beat[i0:i0+len(k)] += k[:len(beat)-i0]
    beat[i0:i0+len(h)] += h[:len(beat)-i0]
    tb += step
mix = voice * 1.0 + beat * 0.7
mix = mix / max(1e-6, np.abs(mix).max()) * 0.9
with wave.open(os.path.join(TMP, "mix.wav"), "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
# envolvente de voz para mover la boca
hop = SR // FPS
env = np.array([np.abs(voice[i*hop:(i+1)*hop]).mean() if i*hop < len(voice) else 0 for i in range(int(DUR*FPS)+2)])
env = env / max(env.max(), 1e-6)

# ---------------- PLASTILINA ----------------
NOISE = Image.fromarray((rng.normal(128, 22, (H//2, W//2))).clip(0, 255).astype(np.uint8)).resize((W, H)).filter(ImageFilter.GaussianBlur(1.2))

def clay(mask, color, size):
    """Convierte una máscara en una pieza de plastilina: textura, brillo y sombra."""
    w, h = size
    m = mask.filter(ImageFilter.GaussianBlur(3))
    base = Image.new("RGB", (w, h), color)
    # textura (huellas)
    n = np.asarray(NOISE.crop((0, 0, w, h)), np.float32) - 128
    arr = np.asarray(base, np.float32) + n[..., None] * 0.35
    # sombreado: claro arriba-izquierda, oscuro abajo-derecha
    inner = np.asarray(m.filter(ImageFilter.GaussianBlur(18)), np.float32) / 255
    sh = np.roll(np.roll(inner, -14, 0), -14, 1) - np.roll(np.roll(inner, 14, 0), 14, 1)
    arr += sh[..., None] * 70
    arr *= (0.75 + 0.25 * inner)[..., None]
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).convert("RGBA")
    img.putalpha(m)
    # sombra proyectada
    out = Image.new("RGBA", (w + 60, h + 60), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (w, h), (60, 30, 20, 0)); shadow.putalpha(m.point(lambda v: v * 0.45))
    out.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)), (30 + 12, 30 + 22))
    out.alpha_composite(img, (30, 30))
    return out

def piece(size, draw_fn, color):
    m = Image.new("L", size, 0); dr = ImageDraw.Draw(m); dr.ink = 255; dr.fill = True; draw_fn(dr); return clay(m, color, size)

def eyes_mouth(d, cx, cy, s, open_=0.0):
    for dx in (-0.28, 0.28):
        d.ellipse([cx+dx*s-0.09*s, cy-0.16*s, cx+dx*s+0.09*s, cy+0.04*s], fill=(30, 20, 20))
        d.ellipse([cx+dx*s-0.03*s, cy-0.12*s, cx+dx*s+0.02*s, cy-0.07*s], fill=(255, 255, 255))
    mh = 0.03*s + open_*0.18*s
    d.ellipse([cx-0.12*s, cy+0.15*s, cx+0.12*s, cy+0.15*s+mh], fill=(120, 30, 40))
    for dx in (-0.42, 0.42):
        d.ellipse([cx+dx*s-0.08*s, cy+0.08*s, cx+dx*s+0.08*s, cy+0.16*s], fill=(240, 120, 120))

def pear_img(s=420, face=False):
    def f(d):
        d.ellipse([s*0.15, s*0.42, s*0.85, s*1.0]); d.ellipse([s*0.28, s*0.18, s*0.72, s*0.68])
    body = piece((s, s), f, (214, 190, 70))
    stem = piece((s, s), lambda d: d.rounded_rectangle([s*0.47, s*0.04, s*0.53, s*0.24], 8), (110, 70, 40))
    leaf = piece((s, s), lambda d: d.ellipse([s*0.52, s*0.06, s*0.78, s*0.17]), (90, 160, 70))
    out = Image.new("RGBA", body.size); out.alpha_composite(stem); out.alpha_composite(leaf); out.alpha_composite(body)
    return out

def boat_img(s=520):
    hull = piece((s, s), lambda d: d.polygon([(s*0.05, s*0.62), (s*0.95, s*0.62), (s*0.78, s*0.85), (s*0.22, s*0.85)]), (190, 80, 60))
    mast = piece((s, s), lambda d: d.rounded_rectangle([s*0.48, s*0.12, s*0.53, s*0.64], 6), (120, 80, 50))
    sail = piece((s, s), lambda d: d.polygon([(s*0.55, s*0.14), (s*0.55, s*0.58), (s*0.85, s*0.56)]), (245, 240, 225))
    sail2 = piece((s, s), lambda d: d.polygon([(s*0.46, s*0.2), (s*0.46, s*0.58), (s*0.2, s*0.56)]), (90, 160, 210))
    out = Image.new("RGBA", hull.size)
    for p in (mast, sail, sail2, hull): out.alpha_composite(p)
    return out

def char_img(s=620, belly=1.0, mouth=0.0):
    out = Image.new("RGBA", (s+60, s+60))
    bw = 0.36 * belly
    body = piece((s, s), lambda d: d.ellipse([s*(0.5-bw), s*0.48, s*(0.5+bw), s*0.98]), (90, 150, 200))
    head = piece((s, s), lambda d: d.ellipse([s*0.25, s*0.06, s*0.75, s*0.56]), (240, 195, 160))
    hair = piece((s, s), lambda d: (d.chord([s*0.23, s*0.03, s*0.77, s*0.45], 180, 360), d.ellipse([s*0.22, s*0.12, s*0.36, s*0.3]), d.ellipse([s*0.64, s*0.12, s*0.78, s*0.3])), (70, 45, 35))
    hands = piece((s, s), lambda d: (d.ellipse([s*0.3, s*0.7, s*0.42, s*0.8]), d.ellipse([s*0.58, s*0.7, s*0.7, s*0.8])), (240, 195, 160))
    for p in (body, hands, head, hair): out.alpha_composite(p)
    eyes_mouth(ImageDraw.Draw(out), 30 + s*0.5, 30 + s*0.34, s*0.42, mouth)
    return out

# fondo de plastilina
def background():
    bg = Image.new("RGB", (W, H), (245, 190, 140))
    d = ImageDraw.Draw(bg)
    for i in range(12):  # rayas suaves estilo "aplanado con rodillo"
        y = i * H // 12
        d.rectangle([0, y, W, y + H // 24], fill=(240, 180, 128))
    arr = np.asarray(bg, np.float32) + (np.asarray(NOISE, np.float32)[..., None] - 128) * 0.25
    bg = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).convert("RGBA")
    floor = piece((W, 700), lambda d: d.rounded_rectangle([-40, 120, W + 40, 760], 120), (200, 120, 90))
    bg.alpha_composite(floor, (-30, H - 620))
    for (x, y, r, c) in [(140, 260, 60, (255, 230, 120)), (930, 420, 40, (255, 150, 170)), (90, 980, 34, (150, 210, 160)), (980, 1100, 50, (170, 170, 240))]:
        bg.alpha_composite(piece((r*2+20, r*2+20), lambda d: d.ellipse([10, 10, r*2+10, r*2+10]), c), (x - r, y - r))
    return bg

BG = background()
FES = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FKO = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
_fc = {}
def font(path, size):
    k = (path, size)
    if k not in _fc: _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]

def text_block(d, txt, y, size, fill, ko=False):
    lines = txt.split("\n")
    for ln in lines:
        f = font(FKO if ko or any('가' <= ch <= '힣' for ch in ln) else FES, size)
        tw = d.textlength(ln, font=f)
        d.text(((W - tw) / 2, y), ln, font=f, fill=fill, stroke_width=max(6, size // 10), stroke_fill=(70, 35, 25))
        y += int(size * 1.2)
    return y

print("Pre-renderizando piezas...")
OBJ = {
    "pear": pear_img(460), "boat": boat_img(600),
    "two": None, "mascot": None, "belly": None, "full": None,
}
two = Image.new("RGBA", (900, 560)); p = pear_img(440)
two.alpha_composite(p, (0, 40)); two.alpha_composite(p, (420, 40)); OBJ["two"] = two
CHAR = {(b, m): char_img(620, b, m) for b in (1.0, 1.35) for m in (0.0, 0.5, 1.0)}

def ease_pop(x):
    x = min(max(x, 0), 1); return 1 + 0.25 * math.sin(x * math.pi) * (1 - x) if x < 1 else 1

# ---------------- RENDER ----------------
NF = int(DUR * FPS)
print(f"Duración {DUR:.1f}s, {NF} cuadros")
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                       "-i", os.path.join(TMP, "mix.wav"), "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
for fi in range(NF):
    t = fi / FPS
    e = next((e for e in events if e["t0"] <= t < e["t1"]), events[-1])
    lt = t - e["t0"]
    frame = BG.copy()
    d = ImageDraw.Draw(frame)
    mo = env[min(fi, len(env)-1)]
    mq = 1.0 if mo > 0.55 else 0.5 if mo > 0.18 else 0.0
    wob = math.sin(t * 6) * 6  # temblor de stop-motion
    jit = (fi // 3) % 2 * 3      # "a saltitos" como stop motion (12 fps aprox)

    if e["kind"] in ("intro", "end"):
        text_block(d, e["scr"], 230 + jit, 96 if e["kind"] == "intro" else 120, (255, 245, 230))
        c = CHAR[(1.0, mq)]
        sc = ease_pop(lt * 3)
        c2 = c.resize((int(c.width*1.25*sc), int(c.height*1.25*sc)))
        frame.alpha_composite(c2, ((W - c2.width)//2, H - c2.height - 120 + int(wob)))
    elif e["kind"] in ("q", "a"):
        if e["kind"] == "q":
            text_block(d, e["scr"], 260 + jit, 130, (255, 245, 230))
        else:
            sc = ease_pop(lt * 2.5)
            fsz = int(260 * sc) if len(e["scr"]) <= 2 else int(200 * sc)
            y = text_block(d, e["scr"], 170 + jit, fsz, (255, 220, 90), ko=True)
            text_block(d, e["rom"], y + 10, 80, (255, 255, 255))
        obj = e["obj"]
        if obj in ("belly", "full"):
            c = CHAR[(1.35 if obj == "full" and e["kind"] == "a" else 1.0, mq)]
            c2 = c.resize((int(c.width*1.3), int(c.height*1.3)))
            frame.alpha_composite(c2, ((W - c2.width)//2, H - c2.height - 110 + int(wob)))
            if obj == "belly":  # flecha señalando la panza
                d.polygon([(220, 1420), (330, 1470), (230, 1520)], fill=(255, 245, 230), outline=(70, 35, 25), width=6)
        else:
            o = OBJ[obj]
            ang = math.sin(t * 3) * 4
            o2 = o.rotate(ang, resample=Image.BICUBIC, expand=True)
            frame.alpha_composite(o2, ((W - o2.width)//2, 820 + int(wob)))
            c = CHAR[(1.0, mq)]; c2 = c.resize((int(c.width*0.7), int(c.height*0.7)))
            frame.alpha_composite(c2, (W - c2.width - 10, H - c2.height - 60))
    elif e["kind"] == "fast":
        parts = e["scr"].split("|")
        k = min(len(parts) - 1, int(lt / ((e["t1"] - e["t0"] - 0.3) / len(parts))))
        y = 200
        for j, ln in enumerate(parts[:k+1]):
            col = (255, 220, 90) if j == k else (255, 245, 230)
            y = text_block(d, ln, y, 84 if j == k else 70, col)
        c = CHAR[(1.35, mq)]
        bounce = abs(math.sin(t * math.pi * 2.5)) * 40
        c2 = c.resize((int(c.width*1.2), int(c.height*1.2)))
        frame.alpha_composite(c2, ((W - c2.width)//2, H - c2.height - 110 - int(bounce)))
    ff.stdin.write(frame.convert("RGB").tobytes())
    if fi % 90 == 0: print(f"  {fi}/{NF}", flush=True)
ff.stdin.close(); ff.wait()
print("Listo:", OUT)

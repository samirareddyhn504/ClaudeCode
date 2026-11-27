"""
La Jerga Catracha - video estilo comic book (vertical 1080x1920, 30 fps, 60 s).

Uso:
    python3 make_video.py            # genera las dos versiones (TikTok y Facebook/KOOL)
    python3 make_video.py --preview  # solo guarda algunos cuadros PNG de prueba

Archivos en assets/: cancion.mp3, logo_tiktok.png, logo_kool_original.jpg, fuentes .ttf
"""
import math, os, random, subprocess, sys, unicodedata
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda *p: os.path.join(HERE, "assets", *p)
OUTDIR = os.path.join(HERE, "salida")
os.makedirs(OUTDIR, exist_ok=True)

W, H, FPS, DUR = 1080, 1920, 30, 60.0
WW, WH = 2160, 3840            # tamaño del "mundo" (escenario) que recorre la cámara
BLACK = (20, 16, 24)
OL = 10                        # grosor de contorno en el mundo
random.seed(4)

# ------------------------------------------------------------------ fuentes
FB = A("Bangers-Regular.ttf")
FC = A("ComicNeue-Bold.ttf")
FD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_fc = {}
def font(p, s):
    k = (p, int(s))
    if k not in _fc:
        _fc[k] = ImageFont.truetype(p, int(s))
    return _fc[k]

# ------------------------------------------------------------------ guion (tiempos confirmados)
KEYS = {"maje", "cipote", "pisto", "pulperia", "fresco", "baleada", "rapidito",
        "chamba", "chambeando", "chambea", "ito"}

def norm(w):
    w = "".join(c for c in unicodedata.normalize("NFD", w.lower()) if unicodedata.category(c) != "Mn")
    return "".join(c for c in w if c.isalpha())

# Diálogo: (inicio, fin_en_pantalla, quien, texto, estilo, habla_hasta, voz_real)
LINES = [
    (4.0, 7.6, "G", "La jerga catracha es fácil, ¿no?", "speech", 7.0, True),
    (9.0, 11.4, "G", "¿Cómo se dice «amigo» en Honduras?", "speech", 11.2, True),
    (11.5, 12.95, "C", "¡Maje!", "shout", 12.4, True),
]
verse = [("G", "¿Un niño?"), ("C", "¡Cipote!"), ("G", "¿Dinero?"), ("C", "¡Pisto!"),
         ("G", "¿Tienda de barrio?"), ("C", "¡Pulpería!"), ("G", "¿Refresco?"), ("C", "¡Fresco!"),
         ("G", "¿Comida?"), ("C", "¡Baleada! ¡Baleada! ¡Baleada!"),
         ("G", "¿Un bus?"), ("C", "¡Rapidito!"), ("G", "¿Trabajo?"), ("C", "¡Chamba!"),
         ("G", "¿Qué hace?"), ("C", "¡Chambeando!")]
vt = [13.0 + i * (18.5 - 13.0) / 9 for i in range(10)]           # 13.0 ... Baleada 18.5
vt += [19.7 + i * (22.0 - 19.7) / 5 for i in range(1, 6)]         # ... Chambeando 22.0
vt = vt[:10] + [19.7] + vt[10:]
vt = [round(x, 3) for x in vt][:16]
for i, (who, txt) in enumerate(verse):
    t0 = vt[i]
    t1 = vt[i + 1] if i + 1 < len(vt) else 23.0
    LINES.append((t0, t1, who, txt, "shout" if who == "C" else "speech", t1 - 0.05, True))
LINES += [
    (26.4, 28.65, "G", "Intenta decir esto:", "speech", 28.2, True),
    (28.7, 32.9, "G", "El niño de mi amigo trabaja en la tienda para comprar su refresco, "
                      "su comida y pagar el bus.", "speech", 30.5, True),
]
RAP_WORDS = ("El cipote del maje chambea en la pulpería pa' ajustar el pisto "
             "de su fresco, su baleada").split()
RAP_TIMES = [33.0 + i * 4.0 / len(RAP_WORDS) for i in range(len(RAP_WORDS))]
RAP_WORDS += ["y", "el", "¡rapidito!"]
RAP_TIMES += [37.0, 37.15, 37.3]
RAP = (33.0, 39.5)
ITO_T = [37.75 + i * 0.36 for i in range(5)]
LINES += [
    (39.5, 46.0, "C", "♪ ¡Pam pam papapapam! ♪", "speech", 46.0, True),
    (42.4, 44.6, "G", "¿Pa... pa-pam?", "small", 44.3, False),
    (46.4, 48.3, "G", "¿Maje...?", "speech", 47.4, False),
    (48.35, 50.3, "C", "¡Jajaja!", "shout", 50.0, False),
]
POW_T = 51.0
BANNER_T = 52.0

def speaking(who, t):
    for (t0, t1, w, txt, st, s1, real) in LINES:
        if w == who and t0 <= t < s1:
            return (t0, s1, real)
    if who == "C" and RAP[0] <= t < RAP[1]:
        return (RAP[0], RAP[1], True)
    return None

# ------------------------------------------------------------------ audio: ritmo y voz
print("Analizando audio...")
import librosa
_y, _sr = librosa.load(A("cancion.mp3"), sr=22050, mono=True)
_, BEATS = librosa.beat.beat_track(y=_y, sr=_sr, units="time")
BEATS = np.asarray(BEATS)
_S = np.abs(librosa.stft(_y, n_fft=1024, hop_length=256))
_f = librosa.fft_frequencies(sr=_sr, n_fft=1024)
_band = _S[(_f > 300) & (_f < 3400)].sum(0)
_bt = librosa.frames_to_time(np.arange(len(_band)), sr=_sr, hop_length=256)
_bn = _band / (np.convolve(_band, np.ones(120) / 120, mode="same") + 1e-6)
def voice_level(t):
    i = min(len(_bn) - 1, max(0, int(np.searchsorted(_bt, t))))
    return float(np.clip((_bn[i] - 0.8) / 0.9, 0, 1))

def beat_phase(t, shift=0.0):
    t = t + shift
    i = int(np.searchsorted(BEATS, t)) - 1
    if i < 0:
        return (t / 0.81) % 1.0
    if i >= len(BEATS) - 1:
        return ((t - BEATS[-1]) / 0.81) % 1.0
    return (t - BEATS[i]) / (BEATS[i + 1] - BEATS[i])

# ------------------------------------------------------------------ ayudas de dibujo
def shade_part(img, draw_fn, color, shade=None, d=(28, 16), ol=OL):
    """Pinta una forma con relleno, sombra tipo cómic (lado derecho/abajo) y contorno negro."""
    dr = ImageDraw.Draw(img)
    draw_fn(dr, dict(fill=color))
    if shade:
        m = Image.new("L", img.size, 0)
        draw_fn(ImageDraw.Draw(m), dict(fill=255))
        off = ImageChops.offset(m, -d[0], -d[1])
        sh = ImageChops.subtract(m, off)
        img.paste(Image.new("RGBA", img.size, shade + (255,)), (0, 0), sh)
    draw_fn(dr, dict(outline=BLACK, width=ol))

def E(box):  return lambda d, k: d.ellipse(box, **k)
def P(pts):  return lambda d, k: d.polygon(pts, **k)
def R(box, r): return lambda d, k: d.rounded_rectangle(box, r, **k)

def limb(img, pts, w, color, shade=None):
    d = ImageDraw.Draw(img)
    d.line(pts, fill=BLACK, width=w + 2 * OL, joint="curve")
    for x, y in (pts[0], pts[-1]):
        r = (w + 2 * OL) / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=BLACK)
    d.line(pts, fill=color, width=w, joint="curve")
    for x, y in (pts[0], pts[-1]):
        r = w / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)
    if shade:
        d.line([(x + w * 0.22, y + w * 0.1) for x, y in pts], fill=shade, width=int(w * 0.35), joint="curve")

def hand(img, x, y, skin, sh, r=46):
    shade_part(img, E([x - r, y - r, x + r, y + r]), skin, sh, d=(12, 8))

# ------------------------------------------------------------------ personajes
SW, SH = 900, 1700           # sprite; pies en (450, 1660)
G = dict(skin=(250, 214, 186), skin_sh=(226, 170, 142), hair=(255, 214, 70), hair_sh=(214, 160, 30),
         shirt=(56, 178, 120), shirt_sh=(32, 128, 86), pants=(70, 100, 170), pants_sh=(48, 70, 128),
         shoe=(245, 245, 245))
C = dict(skin=(190, 132, 92), skin_sh=(150, 98, 66), hair=(40, 30, 28), hair_sh=(20, 14, 12),
         shirt=(250, 250, 252), shirt_sh=(196, 210, 230), pants=(52, 52, 64), pants_sh=(30, 30, 40),
         shoe=(220, 50, 60), cap=(24, 86, 200), cap_sh=(14, 56, 150))

def arms_for(pose, who):
    """Devuelve dos listas de puntos (brazo izq., brazo der.) en coords del sprite."""
    Ls, Rs = (230, 820), (670, 820)
    if pose == "idle":
        return [Ls, (200, 1010), (230, 1180)], [Rs, (700, 1010), (670, 1180)]
    if pose == "ask":       # mano derecha arriba con la palma abierta
        return [Ls, (200, 1010), (230, 1180)], [Rs, (790, 930), (800, 760)]
    if pose == "cross":
        return [Ls, (300, 1040), (590, 1000)], [Rs, (600, 1060), (310, 1010)]
    if pose == "rap":       # mano izq. señalando hacia arriba/cámara
        return [Ls, (130, 860), (150, 650)], [Rs, (700, 1010), (670, 1180)]
    if pose == "laugh":
        return [Ls, (230, 1040), (380, 1130)], [Rs, (670, 1040), (520, 1130)]
    if pose == "danceA":
        return [Ls, (110, 700), (170, 520)], [Rs, (760, 960), (840, 1100)]
    if pose == "danceB":
        return [Ls, (140, 960), (60, 1100)], [Rs, (790, 700), (730, 520)]
    if pose == "fist":      # brazo extendido hacia el otro personaje
        if who == "G":
            return [Ls, (200, 1010), (230, 1180)], [Rs, (760, 900), (830, 900)]
        return [Ls, (140, 900), (70, 900)], [Rs, (700, 1010), (670, 1180)]
    return arms_for("idle", who)

def draw_mouth(img, who, mouth):
    d = ImageDraw.Draw(img)
    cx, cy = 450, 548
    if mouth == "closed":
        if who == "C":
            d.line([(cx - 55, cy), (cx + 55, cy - 4)], fill=BLACK, width=12)
        else:
            d.arc([cx - 60, cy - 40, cx + 60, cy + 20], 20, 160, fill=BLACK, width=12)
    elif mouth in ("mid", "open"):
        h = 46 if mouth == "mid" else 92
        d.ellipse([cx - 52, cy - 18, cx + 52, cy - 18 + h], fill=(120, 24, 40), outline=BLACK, width=10)
        d.ellipse([cx - 28, cy - 18 + h * 0.55, cx + 28, cy - 18 + h - 8], fill=(236, 92, 110))
    elif mouth == "smile":
        d.chord([cx - 80, cy - 50, cx + 80, cy + 50], 0, 180, fill=(120, 24, 40), outline=BLACK, width=10)
        d.rectangle([cx - 60, cy, cx + 60, cy + 14], fill=(255, 255, 255))
    elif mouth == "laugh":
        d.chord([cx - 100, cy - 70, cx + 100, cy + 90], 0, 180, fill=(120, 24, 40), outline=BLACK, width=10)
        d.rectangle([cx - 80, cy + 10, cx + 80, cy + 30], fill=(255, 255, 255))
        d.ellipse([cx - 40, cy + 40, cx + 40, cy + 85], fill=(236, 92, 110))

def make_char(who, pose="idle", mouth="closed"):
    c = G if who == "G" else C
    img = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    # piernas y zapatos
    shade_part(img, R([318, 1230, 446, 1610], 30), c["pants"], c["pants_sh"])
    shade_part(img, R([454, 1230, 582, 1610], 30), c["pants"], c["pants_sh"])
    shade_part(img, E([280, 1580, 450, 1670]), c["shoe"], (180, 180, 180) if who == "G" else (160, 30, 40), d=(14, 10))
    shade_part(img, E([450, 1580, 620, 1670]), c["shoe"], (180, 180, 180) if who == "G" else (160, 30, 40), d=(14, 10))
    # brazos traseros se dibujan después del torso salvo los cruzados
    la, ra = arms_for(pose, who)
    # torso
    torso = [(200, 790), (700, 790), (680, 1290), (220, 1290)]
    shade_part(img, P(torso), c["shirt"], c["shirt_sh"], d=(40, 0))
    d = ImageDraw.Draw(img)
    if who == "C":   # camiseta blanca con franjas azules (Honduras) y número 5
        for y in (930, 1010):
            d.rectangle([214, y, 686, y + 26], fill=(24, 86, 200))
        d.polygon(torso, outline=BLACK, width=OL)
        d.text((450, 1150), "5", font=font(FB, 150), fill=(24, 86, 200), anchor="mm",
               stroke_width=6, stroke_fill=BLACK)
    else:            # camiseta verde con un rayo (casual)
        d.polygon([(470, 960), (400, 1090), (460, 1090), (420, 1200), (540, 1050), (475, 1050), (520, 960)],
                  fill=(255, 220, 60), outline=BLACK)
    # cuello de la camiseta
    d.chord([370, 740, 530, 850], 0, 180, fill=c["skin"], outline=BLACK, width=OL)
    for arm in (la, ra):
        sleeve = arm[:2]
        limb(img, arm, 92, c["skin"], c["skin_sh"])
        limb(img, [arm[0], ((arm[0][0] + arm[1][0]) / 2, (arm[0][1] + arm[1][1]) / 2)], 120, c["shirt"], c["shirt_sh"])
        hand(img, arm[-1][0], arm[-1][1], c["skin"], c["skin_sh"], 52 if pose == "fist" else 46)
    if pose == "fist":
        x, y = (la if who == "C" else ra)[-1]
        d.line([(x - 20, y - 30), (x - 20, y + 30)], fill=BLACK, width=6)
        d.line([(x + 8, y - 30), (x + 8, y + 30)], fill=BLACK, width=6)
    # cuello y cabeza
    shade_part(img, R([395, 600, 505, 790], 30), c["skin"], c["skin_sh"])
    shade_part(img, E([168, 360, 238, 470]), c["skin"], c["skin_sh"], d=(10, 6))
    shade_part(img, E([662, 360, 732, 470]), c["skin"], c["skin_sh"], d=(10, 6))
    shade_part(img, E([200, 130, 700, 670]), c["skin"], c["skin_sh"], d=(44, 24))
    if who == "G":
        # pelo rubio despeinado
        hair = [(190, 340), (200, 210), (260, 140), (330, 80), (400, 100), (430, 40), (500, 90), (560, 50),
                (600, 120), (680, 150), (712, 260), (712, 350), (650, 260), (560, 230), (470, 270), (420, 220),
                (330, 250), (260, 300)]
        shade_part(img, P(hair), c["hair"], c["hair_sh"], d=(30, 18))
        # ojos azules mirando a la derecha
        for ex in (355, 545):
            d.ellipse([ex - 50, 360, ex + 50, 470], fill=(255, 255, 255), outline=BLACK, width=8)
            d.ellipse([ex - 14, 382, ex + 46, 450], fill=(40, 120, 220), outline=BLACK, width=5)
            d.ellipse([ex + 2, 398, ex + 32, 432], fill=BLACK)
            d.ellipse([ex + 4, 392, ex + 18, 406], fill=(255, 255, 255))
        d.line([(300, 330), (400, 320)], fill=c["hair_sh"], width=14)
        d.line([(500, 320), (600, 330)], fill=c["hair_sh"], width=14)
        for fx, fy in ((290, 500), (315, 520), (275, 525), (610, 500), (585, 520), (625, 525)):
            d.ellipse([fx - 6, fy - 6, fx + 6, fy + 6], fill=(210, 130, 90))
        d.arc([420, 440, 480, 500], 200, 340, fill=c["skin_sh"], width=8)       # nariz
    else:
        # pelo negro corto + gorra azul hacia atrás
        shade_part(img, P([(200, 330), (210, 230), (690, 230), (700, 330), (650, 280), (250, 280)]), c["hair"], c["hair_sh"])
        shade_part(img, E([196, 90, 704, 400]), c["cap"], c["cap_sh"], d=(36, 20))
        d.rectangle([200, 236, 700, 280], fill=c["cap"], outline=BLACK, width=OL)        # banda
        d.rounded_rectangle([395, 236, 505, 300], 20, fill=c["hair"], outline=BLACK, width=8)  # abertura
        shade_part(img, E([420, 80, 480, 120]), c["cap"], c["cap_sh"], d=(6, 4))         # botón
        # cejas y lentes oscuros
        d.line([(280, 330), (410, 340)], fill=BLACK, width=20)
        d.line([(490, 340), (620, 330)], fill=BLACK, width=20)
        d.rounded_rectangle([230, 365, 440, 470], 40, fill=(26, 26, 36), outline=BLACK, width=OL)
        d.rounded_rectangle([460, 365, 670, 470], 40, fill=(26, 26, 36), outline=BLACK, width=OL)
        d.line([(440, 395), (460, 395)], fill=BLACK, width=14)
        d.line([(270, 400), (330, 385)], fill=(170, 200, 255), width=12)
        d.line([(500, 400), (560, 385)], fill=(170, 200, 255), width=12)
        d.arc([420, 450, 480, 510], 200, 340, fill=c["skin_sh"], width=8)
        d.polygon([(420, 610), (480, 610), (450, 640)], fill=c["hair"])                  # barbita
    draw_mouth(img, who, mouth)
    return img

# ------------------------------------------------------------------ escenario: esquina del barrio
def halftone(size, spacing=18, r=4, color=(0, 0, 0, 40)):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for j, y in enumerate(range(0, size[1], spacing)):
        for x in range(-spacing if j % 2 else 0, size[0], spacing):
            d.ellipse([x + spacing / 2 - r, y - r, x + spacing / 2 + r, y + r], fill=color)
    return img

def make_world():
    print("Dibujando el barrio...")
    sky = np.zeros((WH, WW, 3), np.float32)
    top, bot = np.array([40, 150, 240]), np.array([255, 196, 120])
    yy = np.clip(np.arange(WH) / 2300.0, 0, 1)[:, None]
    sky[:] = (top * (1 - yy) + bot * yy)[:, None, :]
    world = Image.fromarray(sky.astype(np.uint8)).convert("RGBA")
    world.alpha_composite(halftone((WW, 2400), 30, 6, (255, 255, 255, 40)))
    d = ImageDraw.Draw(world)
    # sol y nubes de cómic
    shade_part(world, E([1500, 260, 1900, 660]), (255, 226, 90), (255, 180, 60), d=(30, 20))
    for cx, cy, s in ((420, 420, 1.0), (1250, 860, 0.8), (300, 1300, 0.7), (1700, 1350, 0.6)):
        blobs = [(-160, 0, 120), (-40, -60, 150), (110, -20, 130), (220, 30, 100), (0, 40, 120)]
        m = Image.new("L", world.size, 0); md = ImageDraw.Draw(m)
        for bx, by, br in blobs:
            md.ellipse([cx + (bx - br) * s, cy + (by - br) * s, cx + (bx + br) * s, cy + (by + br) * s], fill=255)
        edge = m.filter(ImageFilter.MaxFilter(2 * (OL // 2) + 1))
        world.paste(Image.new("RGBA", world.size, BLACK + (255,)), (0, 0), edge)
        world.paste(Image.new("RGBA", world.size, (255, 255, 255, 255)), (0, 0), m.filter(ImageFilter.MinFilter(5)))
    # cerros verdes de fondo
    hills = [(0, 2100), (300, 1820), (650, 1960), (1000, 1760), (1400, 1900), (1800, 1740), (2160, 1880), (2160, 2500), (0, 2500)]
    shade_part(world, P(hills), (110, 180, 90), (80, 140, 70), d=(0, -40))
    rng = random.Random(3)
    for i in range(90):  # casitas lejanas en el cerro
        x = rng.randint(20, 2140); y = rng.randint(1950, 2300)
        col = rng.choice([(255, 140, 160), (120, 210, 230), (255, 230, 110), (250, 170, 90), (240, 240, 240)])
        d.rectangle([x, y, x + 34, y + 26], fill=col, outline=BLACK, width=3)
    # casas de la cuadra
    houses = [
        (-20, 560, 1520, (255, 128, 160), (210, 90, 120), "casa"),
        (560, 1260, 1620, (255, 214, 64), (220, 170, 30), "pulperia"),
        (1260, 1780, 1480, (80, 205, 215), (40, 160, 175), "casa"),
        (1780, 2180, 1680, (255, 150, 70), (220, 110, 40), "casa"),
    ]
    BASE = 3160
    for x0, x1, ytop, col, sh, kind in houses:
        shade_part(world, R([x0, ytop, x1, BASE], 6), col, sh, d=(70, 0))
        world.alpha_composite(halftone((x1 - x0, BASE - ytop), 22, 4, (0, 0, 0, 22)), (max(0, x0), ytop))
        if kind == "casa":   # techo de teja
            shade_part(world, P([(x0 - 40, ytop + 10), (x0 + 60, ytop - 150), (x1 - 60, ytop - 150), (x1 + 40, ytop + 10)]),
                       (196, 80, 50), (150, 56, 34), d=(0, 30))
            for tx in range(x0, x1, 60):
                d.arc([tx, ytop - 60, tx + 60, ytop + 20], 0, 180, fill=BLACK, width=5)
            # ventana con rejas y puerta
            wx0, wx1 = x0 + 80, x0 + 80 + (x1 - x0) * 0.42
            shade_part(world, R([wx0, ytop + 380, wx1, ytop + 820], 8), (150, 210, 255), (110, 170, 220))
            for bx in np.linspace(wx0, wx1, 6)[1:-1]:
                d.line([(bx, ytop + 380), (bx, ytop + 820)], fill=BLACK, width=8)
            d.line([(wx0, ytop + 600), (wx1, ytop + 600)], fill=BLACK, width=8)
            dx0 = x1 - 80 - (x1 - x0) * 0.3
            shade_part(world, R([dx0, BASE - 820, x1 - 80, BASE], 8), (140, 80, 50), (100, 56, 36))
            d.ellipse([dx0 + 20, BASE - 440, dx0 + 50, BASE - 410], fill=(250, 210, 80), outline=BLACK, width=4)
        else:                # pulpería con rótulo y toldo
            shade_part(world, R([x0 + 40, ytop - 120, x1 - 40, ytop + 260], 20), (255, 255, 255), (220, 220, 230))
            d.text(((x0 + x1) / 2, ytop + 30), "PULPERÍA", font=font(FB, 190), fill=(220, 40, 60),
                   anchor="mm", stroke_width=8, stroke_fill=BLACK)
            d.text(((x0 + x1) / 2, ytop + 185), "«LA ESQUINA»", font=font(FB, 100), fill=(24, 86, 200),
                   anchor="mm", stroke_width=5, stroke_fill=BLACK)
            ay = ytop + 520
            stripes = 9; sw = (x1 - x0 + 80) / stripes
            for i in range(stripes):
                c = (230, 50, 60) if i % 2 == 0 else (255, 255, 255)
                bx = x0 - 40 + i * sw
                d.polygon([(bx, ay), (bx + sw, ay), (bx + sw, ay + 200), (bx + sw / 2, ay + 250), (bx, ay + 200)],
                          fill=c, outline=BLACK, width=6)
            # ventanilla con rejas y producto
            shade_part(world, R([x0 + 80, ay + 330, x1 - 80, BASE - 260], 8), (60, 40, 40), (40, 28, 28))
            for sy in (ay + 520, ay + 760):
                d.rectangle([x0 + 90, sy, x1 - 90, sy + 18], fill=(170, 120, 80), outline=BLACK, width=4)
                for k, px in enumerate(range(x0 + 120, x1 - 120, 70)):
                    col = [(230, 60, 60), (60, 160, 230), (250, 210, 60), (80, 200, 120)][k % 4]
                    d.rounded_rectangle([px, sy - 120, px + 50, sy], 10, fill=col, outline=BLACK, width=4)
            for bx in np.linspace(x0 + 80, x1 - 80, 9)[1:-1]:
                d.line([(bx, ay + 330), (bx, BASE - 260)], fill=(90, 90, 100), width=10)
            shade_part(world, R([x0 + 60, BASE - 280, x1 - 60, BASE - 200], 8), (150, 100, 60), (110, 70, 40))
    # poste de luz y cables
    shade_part(world, R([1960, 700, 2030, 3200], 10), (150, 120, 100), (110, 86, 70), d=(24, 0))
    shade_part(world, R([1840, 780, 2150, 830], 6), (150, 120, 100), (110, 86, 70))
    for k, (yy0, yy1) in enumerate(((800, 1040), (800, 1160), (790, 980))):
        pts = [(1860 + 120 * k, 800)]
        for i in range(1, 21):
            x = 1860 + 120 * k - i * (1860 + 120 * k) / 20
            sag = math.sin(i / 20 * math.pi) * 160
            pts.append((x, yy0 + (yy1 - yy0) * i / 20 + sag))
        d.line(pts, fill=BLACK, width=7)
    # acera y calle
    shade_part(world, R([-20, BASE, WW + 20, 3790], 0), (205, 205, 212), (170, 170, 180), d=(0, -30))
    for x in range(0, WW, 240):
        d.line([(x, BASE), (x - 60, 3790)], fill=(150, 150, 160), width=5)
    d.rectangle([-20, 3790, WW + 20, WH], fill=(70, 70, 80))
    d.line([(-20, 3790), (WW + 20, 3790)], fill=BLACK, width=12)
    return world.convert("RGB")

# ------------------------------------------------------------------ cámara
HEAD = {"G": (700, 2400), "C": (1460, 2400)}
FEET_Y = 3660
POS_X = {"G": 700, "C": 1460}

def rect_at(cx, head_y, w, frac=0.42):
    h = w * 16 / 9
    top = head_y - frac * h
    top = min(max(top, 0), WH - h)
    cx = min(max(cx, w / 2), WW - w / 2)
    return [cx - w / 2, top, w, h]

WIDE = lambda: rect_at(1080, 2400, 1960, 0.30)
TWO = lambda: rect_at(1080, 2400, 1640, 0.36)
CG = lambda w=1060, f=0.52: rect_at(790, 2400, w, f)
CC = lambda w=1060, f=0.52: rect_at(1370, 2400, w, f)

def ease(x):
    x = min(max(x, 0), 1)
    return x * x * (3 - 2 * x)

def lerp_rect(a, b, k):
    return [a[i] + (b[i] - a[i]) * k for i in range(4)]

def camera(t):
    if t < 1.0:
        return None
    if t < 4.0:
        start = [1080 - 750, 0, 1500, 1500 * 16 / 9]
        return lerp_rect(start, WIDE(), ease((t - 1.0) / 2.8))
    for (t0, t1, who, txt, st, s1, real) in LINES:
        if 13.0 <= t0 < 23.0 and t0 <= t < t1:
            k = ease((t - t0) / 0.25)
            return CG(1100 - 80 * k) if who == "G" else CC(1100 - 110 * k)
    if t < 7.0:  return CG(1060)
    if t < 9.0:  return lerp_rect(TWO(), WIDE(), ease((t - 7.0) / 2.0))
    if t < 11.5: return CG(1060 - 60 * ease((t - 9.0) / 2.5))
    if t < 13.0:
        k = ease((t - 11.5) / 0.2)
        return CC(1200 - 260 * k)
    if t < 26.4: return lerp_rect(WIDE(), TWO(), ease((t - 23.0) / 3.4))
    if t < 30.6: return lerp_rect(TWO(), rect_at(980, 2400, 1500, 0.36), ease((t - 26.4) / 4.0))
    if t < 33.0: return lerp_rect(rect_at(980, 2400, 1500, 0.36), CC(1000), ease((t - 30.6) / 2.4))
    if t < 39.5:
        ph = beat_phase(t)
        w = 1180 - 70 * (1 - ph) ** 3
        r = CC(w, 0.68)
        sx = math.sin(t * 53) * 14 + math.sin(t * 31) * 10
        sy = math.cos(t * 47) * 12
        return [r[0] + sx, r[1] + sy, r[2], r[3]]
    if t < 46.3: return WIDE()
    if t < 48.3: return CG(1000)
    if t < 50.3:
        r = CC(1000)
        return [r[0] + math.sin(t * 40) * 8, r[1] + math.cos(t * 37) * 8, r[2], r[3]]
    if t < 52.0: return TWO()
    return lerp_rect(TWO(), WIDE(), ease((t - 52.0) / 6.0))

# ------------------------------------------------------------------ globos de diálogo
def tokens(text, big=False):
    out = []
    for w in text.split(" "):
        n = norm(w)
        key = n in KEYS or (n.startswith("chamb") and n != "")
        if "♪" in w:
            out.append((w, font(FD, 64), False))
        elif key:
            out.append((w.upper(), font(FB, 92), True))
        else:
            out.append((w.upper(), font(FB, 86) if big else font(FC, 60), False))
    return out

def layout(toks, maxw):
    lines, cur, cw = [], [], 0
    sp = 18
    for tk in toks:
        tw = tk[1].getlength(tk[0])
        if cur and cw + sp + tw > maxw:
            lines.append(cur); cur, cw = [], 0
        cur.append((tk, tw)); cw += (sp if cw else 0) + tw
    if cur: lines.append(cur)
    return lines, sp

def text_block(text, maxw=780, revealed=None, big=False):
    """Devuelve (img_texto, img_brillo) con palabras clave resaltadas."""
    toks = tokens(text, big)
    lines, sp = layout(toks, maxw)
    lh = [max(tk[1].size for tk, _ in ln) * 1.08 for ln in lines]
    widths = [sum(w for _, w in ln) + sp * (len(ln) - 1) for ln in lines]
    bw, bh = int(max(widths)) + 40, int(sum(lh)) + 30
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    glow = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d, gd = ImageDraw.Draw(img), ImageDraw.Draw(glow)
    y, idx = 10, 0
    for ln, h, wd in zip(lines, lh, widths):
        x = (bw - wd) / 2
        base = y + h * 0.82
        for (txt, f, key), tw in ln:
            show = revealed is None or idx < revealed
            if show:
                if key:
                    gd.text((x, base), txt, font=f, fill=(255, 120, 0, 255), anchor="ls", stroke_width=14, stroke_fill=(255, 60, 160, 255))
                    d.text((x, base), txt, font=f, fill=(255, 214, 0), anchor="ls", stroke_width=5, stroke_fill=BLACK)
                else:
                    d.text((x, base), txt, font=f, fill=BLACK, anchor="ls")
            x += tw + sp; idx += 1
        y += h
    return img, glow.filter(ImageFilter.GaussianBlur(10))

_tb_cache = {}
def balloon_text(key, text, maxw, revealed=None, big=False):
    k = (key, revealed)
    if k not in _tb_cache:
        _tb_cache[k] = text_block(text, maxw, revealed, big)
    return _tb_cache[k]

def draw_balloon(frame, t, t0, text, style, target, side, revealed=None, key=None, maxw=780, top_min=340):
    ti, gl = balloon_text(key or text, text, maxw, revealed, big=(style == "shout"))
    pop = ease((t - t0) / 0.14)
    s = 0.6 + 0.4 * pop + 0.06 * math.sin(min(1, (t - t0) / 0.3) * math.pi)
    pad_x, pad_y = (70, 50) if style != "shout" else (110, 90)
    bw, bh = (ti.width + 2 * pad_x) * s, (ti.height + 2 * pad_y) * s
    tx, ty = target
    cx = tx + (170 if side == "G" else -170)
    cx = min(max(cx, bw / 2 + 40), W - bw / 2 - 40)
    cy = max(top_min + bh / 2, min(ty - 260 - bh / 2, 980))
    box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
    d = ImageDraw.Draw(frame)
    # cola del globo
    bx = min(max(tx, box[0] + 80), box[2] - 80)
    tail = [(bx - 40 * s, box[3] - 20), (bx + 40 * s, box[3] - 20),
            (tx + (bx - tx) * 0.25, min(ty - 40, box[3] + 160))]
    o = 9
    if style == "shout":
        pts = []
        n = 22
        for i in range(n):
            a = 2 * math.pi * i / n
            r = 1.0 if i % 2 == 0 else 0.84
            pts.append((cx + math.cos(a) * bw / 2 * r * 1.06, cy + math.sin(a) * bh / 2 * r * 1.06))
        inner = [(cx + (x - cx) * (1 - o / max(1, math.hypot(x - cx, y - cy))),
                  cy + (y - cy) * (1 - o / max(1, math.hypot(x - cx, y - cy)))) for x, y in pts]
        d.polygon(tail, fill=BLACK); d.line(tail + [tail[0]], fill=BLACK, width=2 * o)
        d.polygon(pts, fill=BLACK)
        d.polygon(tail, fill=(255, 255, 255))
        d.polygon(inner, fill=(255, 250, 220))
    else:
        r = min(bh / 2, 90 * s)
        d.polygon(tail, fill=BLACK); d.line(tail + [tail[0]], fill=BLACK, width=2 * o)
        d.rounded_rectangle([box[0] - o, box[1] - o, box[2] + o, box[3] + o], r + o, fill=BLACK)
        d.polygon(tail, fill=(255, 255, 255))
        d.rounded_rectangle(box, r, fill=(255, 255, 255))
    tw, th = int(ti.width * s), int(ti.height * s)
    if tw < 4 or th < 4:
        return
    ti2 = ti.resize((tw, th)) if s != 1 else ti
    gl2 = gl.resize((tw, th)) if s != 1 else gl
    pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * 9))
    a = gl2.getchannel("A").point(lambda v: int(v * pulse))
    gl2 = gl2.copy(); gl2.putalpha(a)
    px, py = int(cx - tw / 2), int(cy - th / 2)
    frame.alpha_composite(gl2, (px, py))
    frame.alpha_composite(ti2, (px, py))

def sfx(frame, text, cx, cy, size, t, t0, rot=0, fill=(255, 214, 0), shadow=(230, 40, 60)):
    k = ease((t - t0) / 0.12)
    s = size * (0.4 + 0.6 * k + 0.25 * math.sin(min(1, (t - t0) / 0.35) * math.pi))
    f = font(FB, max(10, s))
    tw = f.getlength(text)
    img = Image.new("RGBA", (int(tw + 120), int(s * 1.5)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((70, s * 0.12 + 14), text, font=f, fill=shadow, stroke_width=int(s / 12) + 2, stroke_fill=BLACK)
    d.text((56, s * 0.12), text, font=f, fill=fill, stroke_width=int(s / 12) + 2, stroke_fill=BLACK)
    if rot:
        img = img.rotate(rot, resample=Image.BICUBIC, expand=True)
    frame.alpha_composite(img, (int(cx - img.width / 2), int(cy - img.height / 2)))

def burst(frame, cx, cy, r, t, t0, col=(255, 230, 60)):
    k = ease((t - t0) / 0.1)
    d = ImageDraw.Draw(frame)
    pts = []
    for i in range(28):
        a = 2 * math.pi * i / 28 + 0.2
        rr = r * k * (1.0 if i % 2 == 0 else 0.6)
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    d.polygon(pts, fill=col, outline=BLACK, width=10)

# ------------------------------------------------------------------ portada y efectos de pantalla
def make_cover(chars):
    img = Image.new("RGBA", (W, H), (255, 200, 40, 255))
    d = ImageDraw.Draw(img)
    cx, cy = W / 2, 760
    for i in range(36):
        a0, a1 = 2 * math.pi * i / 36, 2 * math.pi * (i + 0.5) / 36
        d.polygon([(cx, cy), (cx + math.cos(a0) * 3000, cy + math.sin(a0) * 3000),
                   (cx + math.cos(a1) * 3000, cy + math.sin(a1) * 3000)], fill=(255, 150, 30))
    img.alpha_composite(halftone((W, H), 26, 6, (200, 40, 40, 60)))
    d.rectangle([0, 0, W, 300], fill=(24, 86, 200))
    d.line([(0, 300), (W, 300)], fill=BLACK, width=12)
    d.text((60, 70), "Nº 1", font=font(FB, 120), fill=(255, 255, 255), stroke_width=6, stroke_fill=BLACK)
    for i, (sx, sy) in enumerate(((800, 110), (900, 70), (900, 160), (1000, 110), (900, 115))):
        pass
    d.text((W - 60, 150), "¡EDICIÓN\nHONDURAS!", font=font(FB, 66), fill=(255, 255, 255), anchor="rm",
           align="right", stroke_width=5, stroke_fill=BLACK)
    for i, txt in enumerate(("LA JERGA", "CATRACHA")):
        f = font(FB, 230 if i == 0 else 250)
        y = 470 + i * 260
        d.text((W / 2 + 14, y + 16), txt, font=f, fill=(230, 40, 60), anchor="mm", stroke_width=16, stroke_fill=BLACK)
        d.text((W / 2, y), txt, font=f, fill=(255, 255, 255) if i == 0 else (255, 230, 50), anchor="mm",
               stroke_width=16, stroke_fill=BLACK)
    # los dos amigos en la portada
    g = chars[("G", "idle", "smile")]; c = chars[("C", "cross", "closed")]
    for spr, x in ((g, 300), (c, 780)):
        s2 = spr.resize((int(spr.width * 0.62), int(spr.height * 0.62)), Image.LANCZOS)
        img.alpha_composite(s2, (int(x - s2.width / 2), H - s2.height + 40))
    return img.convert("RGB")

def speed_lines(seed):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rnd = random.Random(seed)
    cx, cy = W / 2, H * 0.42
    for i in range(70):
        a = rnd.uniform(0, 2 * math.pi)
        r0 = rnd.uniform(620, 820); r1 = 2000
        wdt = rnd.uniform(0.004, 0.012)
        d.polygon([(cx + math.cos(a) * r0, cy + math.sin(a) * r0),
                   (cx + math.cos(a - wdt) * r1, cy + math.sin(a - wdt) * r1),
                   (cx + math.cos(a + wdt) * r1, cy + math.sin(a + wdt) * r1)], fill=(255, 255, 255, 200))
    return img

DOTS = None
def comic_finish(frame):
    global DOTS
    if DOTS is None:
        DOTS = Image.new("RGB", (W, H), (255, 255, 255))
        DOTS.paste(Image.new("RGB", (W, H), (232, 232, 232)), (0, 0), halftone((W, H), 9, 1.6, (0, 0, 0, 255)).getchannel("A"))
    out = ImageChops.multiply(frame.convert("RGB"), DOTS)
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, W - 1, H - 1], outline=(255, 255, 255), width=16)
    d.rectangle([16, 16, W - 17, H - 17], outline=BLACK, width=10)
    return out

# ------------------------------------------------------------------ logos
def logo_kool():
    src = Image.open(A("logo_kool_original.jpg")).convert("RGB")
    src = ImageEnhance.Color(src).enhance(1.18)
    src = ImageEnhance.Contrast(src).enhance(1.08)
    src = src.filter(ImageFilter.UnsharpMask(radius=3, percent=120, threshold=2))
    n = 1000
    src = src.resize((n, n), Image.LANCZOS)
    m = Image.new("L", (n, n), 0)
    ImageDraw.Draw(m).ellipse([int(n * 0.045), int(n * 0.045), int(n * 0.955), int(n * 0.955)], fill=255)
    out = Image.new("RGBA", (n + 120, n + 120), (0, 0, 0, 0))
    ring = Image.new("RGBA", out.size, (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    rd.ellipse([60 + n * 0.03, 60 + n * 0.03, 60 + n * 0.97, 60 + n * 0.97], outline=(0, 240, 255, 255), width=26)
    out.alpha_composite(ring.filter(ImageFilter.GaussianBlur(22)))
    out.alpha_composite(ring.filter(ImageFilter.GaussianBlur(8)))
    circ = src.convert("RGBA"); circ.putalpha(m)
    out.alpha_composite(circ, (60, 60))
    rd2 = ImageDraw.Draw(out)
    rd2.ellipse([60 + n * 0.04, 60 + n * 0.04, 60 + n * 0.96, 60 + n * 0.96], outline=(255, 90, 230, 255), width=10)
    rd2.ellipse([60 + n * 0.03, 60 + n * 0.03, 60 + n * 0.97, 60 + n * 0.97], outline=(255, 255, 255, 255), width=6)
    return out

def overlay_tiktok():
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    lg = Image.open(A("logo_tiktok.png")).convert("RGBA").resize((150, 150), Image.LANCZOS)
    m = Image.new("L", (150, 150), 0); ImageDraw.Draw(m).ellipse([0, 0, 149, 149], fill=255)
    lg.putalpha(ImageChops.multiply(lg.getchannel("A"), m))
    d = ImageDraw.Draw(ov)
    x, y = 44, 170
    d.ellipse([x - 7, y - 7, x + 157, y + 157], fill=(255, 255, 255, 255), outline=BLACK, width=5)
    ov.alpha_composite(lg, (x, y))
    d.text((x + 172, y + 75), "@themememachine504", font=font(FB, 54), fill=(255, 255, 255, 255), anchor="lm",
           stroke_width=6, stroke_fill=BLACK)
    return ov

def overlay_kool(kool):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    s = 190
    lg = kool.resize((s, s), Image.LANCZOS)
    ov.alpha_composite(lg, (W - s - 34, 150))
    return ov

# ------------------------------------------------------------------ render
def main(preview=False):
    print("Preparando personajes...")
    chars = {}
    for who, poses in (("G", ("idle", "ask", "danceA", "danceB", "fist")),
                       ("C", ("idle", "cross", "rap", "fist", "laugh"))):
        for p in poses:
            for m in ("closed", "mid", "open", "smile", "laugh"):
                chars[(who, p, m)] = make_char(who, p, m)
    world = make_world()
    cover = make_cover(chars)
    speed = [speed_lines(i) for i in range(3)]
    kool = logo_kool()
    kool.save(os.path.join(OUTDIR, "logo_kool_mejorado.png"))
    ov_tt, ov_fb = overlay_tiktok(), overlay_kool(kool)

    def pose_mouth(who, t):
        sp = speaking(who, t)
        mouth = "closed"
        if sp:
            t0, s1, real = sp
            lvl = voice_level(t) if real else (0.5 + 0.5 * math.sin((t - t0) * 22))
            mouth = "open" if lvl > 0.6 else "mid" if lvl > 0.25 else "closed"
        if who == "G":
            pose = "idle"
            for (t0, t1, w, txt, st, s1, real) in LINES:
                if w == "G" and t0 <= t < t1 and txt.startswith("¿"):
                    pose = "ask"
            if 39.5 <= t < 46.3:
                pose = "danceA" if int((t + 0.31) / 0.53) % 2 else "danceB"
            if 50.4 <= t < 52.0: pose = "fist"
            if t >= 52.0: pose, mouth = "idle", ("smile" if mouth == "closed" else mouth)
            if 48.4 <= t < 50.4 and mouth == "closed": mouth = "smile"
        else:
            pose = "idle"
            for (t0, t1, w, txt, st, s1, real) in LINES:
                if w == "C" and t0 <= t < t1 and 13 <= t0 < 23:
                    pose = "rap"
            if 23.0 <= t < 33.0: pose = "cross"
            if 33.0 <= t < 39.5: pose = "rap"
            if 48.35 <= t < 50.4: pose, mouth = "laugh", ("laugh" if mouth != "closed" else "smile")
            if 50.4 <= t < 52.0: pose = "fist"
            if t >= 52.0: pose, mouth = "idle", "smile"
        return pose, mouth

    def bob(who, t):
        amp = 14
        shift = 0.0
        if 13 <= t < 23 or 33 <= t < 39.5: amp = 24
        if 39.5 <= t < 46.3 or t >= 52: amp = 46
        if who == "G" and 39.5 <= t < 46.3:
            shift = 0.37 + 0.12 * math.sin(t * 3.1)      # bailando a destiempo
        ph = beat_phase(t, shift)
        return -abs(math.sin(math.pi * ph)) * amp, ph

    def render(fi):
        t = fi / FPS
        if t < 1.0:
            z = 1.0 + 0.05 * t
            cw, ch = int(W / z), int(H / z)
            fr = cover.crop(((W - cw) // 2, (H - ch) // 2, (W + cw) // 2, (H + ch) // 2)).resize((W, H))
            return comic_finish(fr)
        rx, ry, rw, rh = camera(t)
        sc = W / rw
        fr = world.crop((int(rx), int(ry), int(rx + rw), int(ry + rh))).resize((W, H), Image.BILINEAR).convert("RGBA")
        heads = {}
        for who in ("G", "C"):
            pose, mouth = pose_mouth(who, t)
            by, ph = bob(who, t)
            spr = chars[(who, pose, mouth)]
            # al chocar los puños se acercan un poquito
            dx = 0
            if 50.4 <= t < 52.0:
                k = ease((t - 50.4) / 0.5)
                dx = (60 if who == "G" else -60) * k
            wx = POS_X[who] + dx
            sw, sh = int(SW * sc), int(SH * sc)
            s2 = spr.resize((sw, sh), Image.BILINEAR)
            if who == "G" and 39.5 <= t < 46.3:
                s2 = s2.rotate(math.sin(t * 4.3) * 7, resample=Image.BILINEAR, expand=False)
            px = (wx - 450 - rx) * sc
            py = (FEET_Y - 1660 + by - ry) * sc
            fr.alpha_composite(s2, (int(px), int(py)))
            heads[who] = ((wx - rx) * sc, (2400 + by - ry) * sc, 270 * sc)
        if 33.0 <= t < 39.5:
            fr.alpha_composite(speed[(fi // 2) % 3])
        # globos
        for i, (t0, t1, who, txt, st, s1, real) in enumerate(LINES):
            if t0 <= t < t1:
                hx, hy, hr = heads[who]
                target = (hx + (60 if who == "G" else -60), hy - hr * 1.0)
                if st == "small":
                    draw_balloon(fr, t, t0, txt, "speech", target, who, maxw=420, key=("s", i), top_min=980)
                else:
                    draw_balloon(fr, t, t0, txt, st, target, who, key=("l", i))
        if RAP[0] <= t < RAP[1]:
            n = sum(1 for x in RAP_TIMES if x <= t)
            hx, hy, hr = heads["C"]
            draw_balloon(fr, t, RAP[0], " ".join(RAP_WORDS), "speech", (hx - 40, hy - hr * 1.0), "C",
                         revealed=n, key="rap", maxw=860)
            if t >= 37.3:
                sfx(fr, "¡RAPIDITO!", W / 2, 1180, 150, t, 37.3, rot=-6)
            for k, ti in enumerate(ITO_T):
                if t >= ti:
                    rr = random.Random(k)
                    sfx(fr, "¡ITO!", rr.uniform(220, 860), rr.uniform(1350, 1650), 120, t, ti,
                        rot=rr.uniform(-18, 18), fill=[(255, 214, 0), (90, 220, 255), (255, 120, 200)][k % 3])
        if 50.9 <= t < 52.6:
            hxg = heads["G"][0]; hxc = heads["C"][0]
            fx = (hxg + hxc) / 2
            fy = heads["G"][1] + 480 * (W / camera(t)[2])
            burst(fr, fx, fy, 200, t, POW_T)
            sfx(fr, "¡POW!", fx, fy, 130, t, POW_T, rot=-8, fill=(255, 255, 255), shadow=(24, 86, 200))
        if t >= BANNER_T:
            k = ease((t - BANNER_T) / 0.25)
            d = ImageDraw.Draw(fr)
            y0 = 360 - (1 - k) * 300
            d.polygon([(60, y0), (W - 60, y0 + 20), (W - 80, y0 + 330), (80, y0 + 310)], fill=(255, 230, 50), outline=BLACK, width=12)
            sfx(fr, "¡ASÍ SE HABLA,", W / 2, y0 + 95, 118, t, BANNER_T, fill=(255, 255, 255), shadow=(24, 86, 200))
            sfx(fr, "MAJE!", W / 2, y0 + 225, 160, t, BANNER_T + 0.1, fill=(230, 40, 60), shadow=(24, 86, 200))
        return comic_finish(fr)

    if preview:
        for t in (0.5, 2.5, 5.0, 11.8, 14.0, 18.7, 27.0, 29.5, 35.0, 38.5, 42.0, 47.0, 49.0, 51.2, 55.0):
            render(int(t * FPS)).save(os.path.join(OUTDIR, f"preview_{t:05.1f}.png"))
        print("Previews en", OUTDIR)
        return

    base = os.path.join(OUTDIR, "_base.mp4")
    nf = int(DUR * FPS)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                           "-pix_fmt", "yuv420p", base], stdin=subprocess.PIPE)
    for fi in range(nf):
        ff.stdin.write(render(fi).tobytes())
        if fi % 150 == 0:
            print(f"  cuadro {fi}/{nf}", flush=True)
    ff.stdin.close(); ff.wait()

    for name, ov in (("jerga_catracha_TIKTOK.mp4", ov_tt), ("jerga_catracha_FACEBOOK_KOOL.mp4", ov_fb)):
        op = os.path.join(OUTDIR, "_ov.png"); ov.save(op)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", base, "-i", op, "-i", A("cancion.mp3"),
                        "-filter_complex", "[0:v][1:v]overlay=0:0[v]", "-map", "[v]", "-map", "2:a:0",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k", "-t", str(DUR), "-movflags", "+faststart",
                        os.path.join(OUTDIR, name)], check=True)
        os.remove(op)
        print("Listo:", name)
    os.remove(base)

if __name__ == "__main__":
    main(preview="--preview" in sys.argv)

"""
Boxing Ring 3D  (full-body webcam tracking, fight a bot in a ring)
-------------------------------------------------------------------
Install:   pip install ursina opencv-python mediapipe
           (if you get "module 'mediapipe' has no attribute 'solutions'":
            pip install mediapipe==0.10.14)

SETUP:     Stand ~2 m from the webcam so your head, shoulders and both arms are
           visible (the preview at the bottom-right shows the skeleton).
           Good light + plain background = much better tracking.

CONTROLS (body)
  Punch ........ throw real punches (jab / cross / hook / uppercut are detected)
  Block ........ hold both fists up next to your face
  Dodge ........ lean left/right (sidestep your head)  or  duck down
  C ............ recalibrate your neutral stance (stand in guard, press C)
CONTROLS (keyboard fallback / testing)
  F / left-click = left punch      J / right-click = right punch
  A / D = lean     S = duck        SPACE = guard
  ENTER = start / restart          ESC = quit

HOW TO WIN:  Bot telegraphs punches (its glove turns YELLOW while winding up).
  - Jab (straight)  : lean sideways OR duck OR block
  - Hook (wide arc) : duck (or block)
  - Body shot (low) : lean sideways (blocking only helps a little)
  Dodge a punch and the bot is off balance -> your next hits do +50% damage.
  3 rounds, 60 s each. Knock the bot out or have more health at the end.
"""
from ursina import *
from ursina.shaders import lit_with_shadows_shader, unlit_shader
from panda3d.core import Texture as PandaTexture
import math
import os
import colorsys
import threading
import random as rnd
import time as pytime          # (ursina's own `time.dt` is used for delta time)
import cv2
import mediapipe as mp

# ----------------------------------------------------------------------------
# TUNING
# ----------------------------------------------------------------------------
QUALITY = 'high'               # 'high' = lights + shadows, 'low' = flat/fast (use if it looks black or crashes)
CAM_INDEX = 0
CAM_W, CAM_H = 1280, 720       # asked from the webcam (it picks the closest it supports)
POSE_COMPLEXITY = 1            # 0 fast, 1 good, 2 most accurate (slower)
PREVIEW_W, PREVIEW_H = 480, 270
PREVIEW_SCALE = 0.5

CAM_FOV = 80
GLOVE_SCALE = (2.3, 2.3, 3.2)  # how far your real arm movement is stretched in the game (x, y, z)
PUNCH_SPEED = 1.8              # m/s of the wrist relative to the shoulder to count as a punch (lower = easier)
PUNCH_MIN_EXT = 0.58           # arm must be at least this straight (0..1)
PUNCH_COOLDOWN = 0.28
GUARD_DIST = 0.38              # both wrists this close to the nose (metres) = guarding
LEAN_T = 0.42                  # lean needed to dodge (in shoulder widths)
DUCK_T = 0.30                  # duck needed to dodge (in shoulder widths)
HEAD_R, BODY_R = 0.36, 0.50    # hit radius around bot's head / body

ROUNDS = 3
ROUND_TIME = 60
CALIB_TIME = 3.0


# ----------------------------------------------------------------------------
# SMALL HELPERS
# ----------------------------------------------------------------------------
def rgb(r, g, b, a=1.0):
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return color.hsv(h * 360, s, v, a)


def length(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


class OneEuro:
    """One-Euro filter: removes jitter when slow, stays fast when you punch."""
    def __init__(self, min_cutoff=1.2, beta=3.0, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = 0.0

    @staticmethod
    def _a(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        if self.x is None:
            self.x = x
            return x
        dt = max(dt, 1e-3)
        d = (x - self.x) / dt
        self.dx += self._a(self.dc, dt) * (d - self.dx)
        cutoff = self.mc + self.beta * abs(self.dx)
        self.x += self._a(cutoff, dt) * (x - self.x)
        return self.x


# ----------------------------------------------------------------------------
# BODY TRACKING (own thread): MediaPipe Pose + One-Euro smoothing + punch detection
# ----------------------------------------------------------------------------
class PoseTracker:
    USED = (0, 11, 12, 13, 14, 15, 16, 23, 24)
    ARMS = (('L', (11, 13, 15)), ('R', (12, 14, 16)))     # (shoulder, elbow, wrist) - anatomical left/right

    def __init__(self):
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(CAM_INDEX, backend)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.error = None if self.cap.isOpened() else 'Could not open the webcam - keyboard mode only'
        self.pose = mp.solutions.pose.Pose(
            model_complexity=POSE_COMPLEXITY, smooth_landmarks=True, enable_segmentation=False,
            min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.lock = threading.Lock()
        self.data = None
        self.events = []
        self.preview = None
        self.frame_id = 0
        self.running = True
        self.fw = {i: [OneEuro(1.5, 6.0) for _ in range(3)] for i in self.USED}
        self.fi = {i: [OneEuro(1.2, 3.0) for _ in range(2)] for i in self.USED}
        self.arm_len = {'L': 0.62, 'R': 0.62}
        self.prev_rel = {'L': None, 'R': None}
        self.last_punch = {'L': -9.0, 'R': -9.0}
        self.last_t = None
        threading.Thread(target=self._loop, daemon=True).start()

    # -- public (called from the game thread) -------------------------------------
    def get(self):
        with self.lock:
            d = self.data
        if d is None or pytime.perf_counter() - d['t'] > 0.4:
            return None
        return d

    def pop_events(self):
        with self.lock:
            ev, self.events = self.events, []
        return ev

    def stop(self):
        self.running = False

    # -- internals ------------------------------------------------------------------
    def _digest(self, res, aspect, t):
        img = res.pose_landmarks.landmark
        wor = res.pose_world_landmarks.landmark
        if img[11].visibility < 0.5 or img[12].visibility < 0.5:
            return False
        dt = 1 / 30 if self.last_t is None else clamp(t - self.last_t, 0.012, 0.2)
        self.last_t = t

        G, I = {}, {}                     # G: game-axes metres (x right, y up, z forward) ; I: mirrored image coords
        for i in self.USED:
            fw, fi = self.fw[i], self.fi[i]
            wx = fw[0](wor[i].x, dt)
            wy = fw[1](wor[i].y, dt)
            wz = fw[2](wor[i].z, dt)
            G[i] = (-wx, -wy, -wz)        # webcam looks at you: toward the camera = forward in the game
            I[i] = ((1 - fi[0](img[i].x, dt)) * aspect, fi[1](img[i].y, dt))

        sw = max(0.05, math.hypot(I[11][0] - I[12][0], I[11][1] - I[12][1]))
        sh_mid = ((I[11][0] + I[12][0]) / 2, (I[11][1] + I[12][1]) / 2)

        arms, evs = {}, []
        for side, (s, e, w) in self.ARMS:
            rel = sub(G[w], G[s])
            cur_len = length(sub(G[e], G[s])) + length(sub(G[w], G[e]))
            self.arm_len[side] = clamp(self.arm_len[side] * 0.97 + cur_len * 0.03, 0.3, 0.9)
            ext = min(1.15, length(rel) / self.arm_len[side])
            prev = self.prev_rel[side]
            v, speed = (0, 0, 0), 0.0
            if prev is not None:
                v = tuple((rel[k] - prev[k]) / dt for k in range(3))
                speed = length(v)
            self.prev_rel[side] = rel
            ok = min(img[e].visibility, img[w].visibility) > 0.35
            if ok and speed > PUNCH_SPEED and ext > PUNCH_MIN_EXT and t - self.last_punch[side] > PUNCH_COOLDOWN:
                outward = (v[0] * rel[0] + v[1] * rel[1] + v[2] * rel[2]) / max(length(rel), 1e-4)
                if outward > 0.45 * speed:                       # fist is moving AWAY from the shoulder
                    ax, ay, az = abs(v[0]), abs(v[1]), abs(v[2])
                    if v[1] > 0 and ay > ax * 1.1 and ay > az * 0.9:
                        kind = 'uppercut'
                    elif v[2] > 0 and az >= max(ax, ay) * 0.8:
                        kind = 'straight'
                    else:
                        kind = 'hook'
                    evs.append(dict(side=side, kind=kind, speed=speed, t=t))
                    self.last_punch[side] = t
            arms[side] = dict(rel=rel, el=sub(G[e], G[s]), ext=ext, speed=speed, ok=ok)

        guard = length(sub(G[15], G[0])) < GUARD_DIST and length(sub(G[16], G[0])) < GUARD_DIST
        data = dict(t=t, arms=arms, head=I[0], sh_mid=sh_mid, sw=sw, guard=guard)
        with self.lock:
            self.data = data
            self.events.extend(evs)
        return True

    def _loop(self):
        draw = mp.solutions.drawing_utils
        conns = mp.solutions.pose.POSE_CONNECTIONS
        try:
            while self.running and self.error is None:
                ok, frame = self.cap.read()
                if not ok:
                    pytime.sleep(0.01)
                    continue
                h, w = frame.shape[:2]
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_img.flags.writeable = False
                res = self.pose.process(rgb_img)       # NOT mirrored: left/right stay anatomically correct
                good = False
                if res.pose_landmarks and res.pose_world_landmarks:
                    good = self._digest(res, w / h, pytime.perf_counter())
                    draw.draw_landmarks(frame, res.pose_landmarks, conns)
                show = cv2.flip(frame, 1)               # mirror for the preview only
                label, col = ('BODY TRACKED', (0, 220, 0)) if good else ('NO BODY - step back, show arms', (0, 0, 255))
                cv2.putText(show, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                small = cv2.resize(show, (PREVIEW_W, PREVIEW_H))
                small = cv2.cvtColor(cv2.flip(small, 0), cv2.COLOR_BGR2RGB)     # panda is bottom-up
                self.preview = small.tobytes()
                self.frame_id += 1
        finally:
            self.cap.release()
            self.pose.close()


# ----------------------------------------------------------------------------
# ENGINE + LOOK
# ----------------------------------------------------------------------------
app = Ursina(title='Boxing Ring 3D')
window.color = rgb(6, 6, 10)
window.fps_counter.enabled = False
window.exit_button.visible = False
tracker = PoseTracker()

HIGH = QUALITY == 'high'
if HIGH:
    try:
        Entity.default_shader = lit_with_shadows_shader
    except Exception:
        HIGH = False

WHITE = rgb(235, 235, 240)
BLACK = rgb(15, 15, 18)
RED = rgb(200, 25, 30)
BLUE = rgb(30, 70, 210)
STEEL = rgb(120, 125, 135)


def box(pos, scale, col, **kw):
    return Entity(model='cube', position=pos, scale=scale, color=col, **kw)


def neon(pos, scale, col):
    return Entity(model='cube', position=pos, scale=scale, color=col, shader=unlit_shader)


def ui_quad(**kw):
    return Entity(parent=camera.ui, model='quad', shader=unlit_shader, **kw)


def build_arena():
    # --- ring floor, apron, logo -------------------------------------------------
    box((0, -0.15, 0), (7.2, 0.3, 7.2), rgb(176, 178, 190))                      # canvas
    box((0, -0.70, 0), (7.3, 0.8, 7.3), rgb(20, 28, 90))                         # apron (blue skirt)
    box((0, -0.55, 0), (7.32, 0.08, 7.32), WHITE)                                # skirt stripe
    Entity(model='circle', rotation_x=90, scale=3.6, y=0.012, color=RED)
    Entity(model='circle', rotation_x=90, scale=3.0, y=0.014, color=rgb(176, 178, 190))
    Entity(model='circle', rotation_x=90, scale=1.2, y=0.016, color=BLUE)
    for s in (-1, 1):                                                           # corner paint
        Entity(model='quad', rotation_x=90, scale=(2.0, 0.12), position=(0, 0.013, s * 1.9),
               color=RED if s < 0 else BLUE)
    # --- posts, pads, ropes --------------------------------------------------------
    corners = {(-3, -3): RED, (3, 3): BLUE, (3, -3): WHITE, (-3, 3): WHITE}
    for (px, pz), col in corners.items():
        box((px, 0.8, pz), (0.14, 1.6, 0.14), STEEL)
        for ry in (0.55, 0.95, 1.35):
            box((px, ry, pz), (0.30, 0.30, 0.30), col)
    for ry, col in ((0.55, WHITE), (0.95, RED), (1.35, WHITE)):
        box((0, ry, -3), (6.0, 0.055, 0.055), col)
        box((0, ry, 3), (6.0, 0.055, 0.055), col)
        box((-3, ry, 0), (0.055, 0.055, 6.0), col)
        box((3, ry, 0), (0.055, 0.055, 6.0), col)
    # --- hall floor + barrier boards with neon -------------------------------------
    box((0, -1.12, 0), (80, 0.04, 80), rgb(12, 12, 18))
    for s in (-1, 1):
        box((0, -0.55, s * 5.6), (12, 1.1, 0.14), rgb(18, 18, 26))
        box((s * 5.6, -0.55, 0), (0.14, 1.1, 12), rgb(18, 18, 26))
        neon((0, -0.12, s * 5.52), (12, 0.07, 0.05), rgb(0, 220, 255) if s < 0 else rgb(255, 40, 200))
        neon((s * 5.52, -0.12, 0), (0.05, 0.07, 12), rgb(255, 40, 200) if s < 0 else rgb(0, 220, 255))
    # --- overhead lights + truss -----------------------------------------------------
    for sx in (-1, 1):
        box((sx * 3.4, 7.6, 0), (0.18, 0.18, 8.0), rgb(30, 30, 36))
        box((0, 7.6, sx * 3.4), (8.0, 0.18, 0.18), rgb(30, 30, 36))
    for sx in (-1, 1):
        for sz in (-1, 1):
            neon((sx * 3.0, 7.4, sz * 3.0), (0.9, 0.12, 0.9), rgb(255, 250, 235))
            neon((sx * 3.0, 7.0, sz * 3.0), (0.5, 0.05, 0.5), rgb(255, 240, 200))
    # --- crowd ------------------------------------------------------------------------
    crowd = Entity()
    tiers = [(8.0, 0.2), (9.4, 1.0), (10.8, 1.8)]
    for d, top in tiers:
        for side in range(4):
            def at(u, dd):
                return ((u, dd), (u, -dd), (dd, u), (-dd, u))[side]
            sx, sz = at(0, d)
            long_x = side < 2
            box((sx, (top - 1.1) / 2 - 0.0, sz), (24 if long_x else 1.4, top + 1.1, 1.4 if long_x else 24),
                rgb(14, 14, 20))
            u = -11.0
            while u < 11.0:
                px, pz = at(u + rnd.uniform(-0.15, 0.15), d + rnd.uniform(-0.2, 0.2))
                body = color.hsv(rnd.uniform(0, 360), rnd.uniform(0.25, 0.7), rnd.uniform(0.18, 0.5))
                skin = color.hsv(rnd.uniform(14, 34), rnd.uniform(0.3, 0.55), rnd.uniform(0.35, 0.65))
                h = rnd.uniform(0.0, 0.12)
                Entity(parent=crowd, model='cube', position=(px, top + 0.32 + h, pz), scale=(0.46, 0.62, 0.32), color=body)
                Entity(parent=crowd, model='sphere', position=(px, top + 0.82 + h, pz), scale=0.24, color=skin)
                u += rnd.uniform(0.62, 0.9)
    try:
        crowd.combine()
    except Exception:
        pass
    # --- lighting -----------------------------------------------------------------------
    if HIGH:
        try:
            AmbientLight(color=color.hsv(225, 0.25, 0.55))
            sun = DirectionalLight(position=Vec3(-2, 10, -6), shadows=True, shadow_map_resolution=Vec2(2048, 2048))
            sun.look_at(Vec3(0, 0, 1))
            try:
                sun.update_bounds(scene)
            except Exception:
                pass
        except Exception:
            pass
    try:
        scene.fog_color = rgb(6, 6, 10)
        scene.fog_density = (14, 48)
    except Exception:
        pass


build_arena()


# ----------------------------------------------------------------------------
# LIMB HELPER (stretched sphere between two points)
# ----------------------------------------------------------------------------
def place(e, a, b, t):
    d = b - a
    e.position = (a + b) * 0.5
    e.rotation = (-math.degrees(math.atan2(d.y, math.hypot(d.x, d.z))),
                  math.degrees(math.atan2(d.x, d.z)), 0)
    e.scale = (t, t, d.length() + t * 0.7)


# ----------------------------------------------------------------------------
# THE PLAYER (first person: camera, red gloves, arms, lean / duck / guard)
# ----------------------------------------------------------------------------
P_SKIN = rgb(232, 178, 140)
P_GLOVE = rgb(205, 25, 30)
P_GLOVE_DARK = rgb(150, 15, 22)


class Player:
    def __init__(self):
        self.rig = Entity(position=(0, 1.65, -1.7))
        camera.parent = self.rig
        camera.position = (0, 0, 0)
        camera.rotation = (0, 0, 0)
        camera.fov = CAM_FOV
        camera.clip_plane_near = 0.04
        self.g, self.up, self.fore = {}, {}, {}
        self.gpos, self.epos = {}, {}
        self.anchor = {'L': Vec3(-0.22, -0.28, 0.10), 'R': Vec3(0.22, -0.28, 0.10)}
        for s, inward in (('L', 1), ('R', -1)):
            g = Entity(parent=camera)
            Entity(parent=g, model='sphere', color=P_GLOVE, scale=(0.21, 0.19, 0.25))
            Entity(parent=g, model='sphere', color=P_GLOVE_DARK, scale=(0.09, 0.09, 0.15), position=(inward * 0.10, 0.02, -0.02))
            Entity(parent=g, model='cube', color=WHITE, scale=(0.17, 0.17, 0.10), position=(0, 0, -0.16))
            self.g[s] = g
            self.up[s] = Entity(parent=camera, model='sphere', color=P_SKIN)
            self.fore[s] = Entity(parent=camera, model='sphere', color=P_SKIN)
            self.gpos[s] = Vec3(-inward * 0.18, -0.02, 0.5)
            self.epos[s] = Vec3(-inward * 0.28, -0.35, 0.2)
        self.pt = {'L': 0.0, 'R': 0.0}                 # time left in the "punch is live" window
        self.pspeed = {'L': 0.0, 'R': 0.0}
        self.pkind = {'L': 'straight', 'R': 'straight'}
        self.assist = {'L': 0.0, 'R': 0.0}
        self.last_punch_time = -9.0
        self.base = None                               # neutral stance (head x, head y, shoulder y)
        self.cal = []
        self.lean = self.duck = 0.0
        self.rl = self.rd = 0.0                        # "recent" lean / duck (so a slightly early dodge still counts)
        self.rg = 0.0
        self.guard_vis = False
        self.shake = 0.0
        self.combo = 0
        self.combo_t = 0.0

    def calibrate(self):
        self.cal = []

    def fake_punch(self, side):
        self.on_punch(side, 5.0, 'straight')

    def on_punch(self, side, speed, kind):
        self.pt[side] = 0.30
        self.pspeed[side] = speed
        self.pkind[side] = kind
        self.assist[side] = 1.0
        self.last_punch_time = pytime.perf_counter()

    @property
    def guarding(self):
        return self.rg > 0

    def update(self, dt, data, fight, game, bot):
        for ev in tracker.pop_events():
            if fight:
                self.on_punch(ev['side'], ev['speed'], ev['kind'])

        # --- lean / duck / guard ------------------------------------------------------
        t_lean, t_duck, guard = 0.0, 0.0, False
        if data:
            hx, hy = data['head']
            sy = data['sh_mid'][1]
            sw = data['sw']
            if self.cal is not None:
                self.cal.append((hx, hy, sy))
                if len(self.cal) >= 20:
                    n = len(self.cal)
                    self.base = (sum(c[0] for c in self.cal) / n, sum(c[1] for c in self.cal) / n,
                                 sum(c[2] for c in self.cal) / n)
                    self.cal = None
            if self.base:
                t_lean = (hx - self.base[0]) / sw
                t_duck = ((hy - self.base[1]) + (sy - self.base[2])) / 2 / sw
            guard = data['guard']
        t_lean += (held_keys['d'] - held_keys['a']) * 0.9
        t_duck += held_keys['s'] * 0.7
        guard = guard or bool(held_keys['space'])
        k = min(1, dt * 14)
        self.lean += (clamp(t_lean, -1.2, 1.2) - self.lean) * k
        self.duck += (clamp(t_duck, -0.3, 1.2) - self.duck) * k
        self.rl = max(abs(self.lean), self.rl * max(0.0, 1 - dt * 6))
        self.rd = max(self.duck, self.rd * max(0.0, 1 - dt * 6))
        self.rg = 0.2 if guard else self.rg - dt
        if game.state != 'over' or game.player_hp > 0:
            self.rig.x += (clamp(self.lean, -1, 1) * 0.55 - self.rig.x) * min(1, dt * 12)
            self.rig.y += ((1.65 - clamp(self.duck, 0, 1) * 0.8) - self.rig.y) * min(1, dt * 12)

        # --- gloves + arms ---------------------------------------------------------------
        sx, sy_, sz = GLOVE_SCALE
        for s in ('L', 'R'):
            side_sign = -1 if s == 'L' else 1
            a = data['arms'][s] if data else None
            self.assist[s] = max(0.0, self.assist[s] - dt * 3.5)
            if a and a['ok']:
                rel, el = a['rel'], a['el']
                tg = self.anchor[s] + Vec3(rel[0] * sx, rel[1] * sy_, rel[2] * sz)
                te = self.anchor[s] + Vec3(el[0] * sx, el[1] * sy_, el[2] * sz)
            elif a:
                tg, te = self.gpos[s], self.epos[s]
            else:
                tg = Vec3(side_sign * 0.2, -0.02, 0.5)
                te = Vec3(side_sign * 0.3, -0.40, 0.15)
            tg = Vec3(clamp(tg.x, -1.4, 1.4), clamp(tg.y, -1.2, 1.1), clamp(tg.z, 0.12, 2.3))
            tg = tg + Vec3(0, 0, self.assist[s] * 0.6)
            kk = min(1, dt * 30)
            self.gpos[s] += (tg - self.gpos[s]) * kk
            self.epos[s] += (te - self.epos[s]) * kk
            self.g[s].position = self.gpos[s]
            place(self.up[s], self.anchor[s], self.epos[s], 0.10)
            place(self.fore[s], self.epos[s], self.gpos[s] - Vec3(0, 0, 0.05), 0.09)

            # --- did this punch land? ---------------------------------------------------
            if self.pt[s] > 0:
                self.pt[s] -= dt
                if fight and bot.state != 'ko':
                    gw = self.g[s].world_position
                    head = (gw - bot.head_w).length() < HEAD_R
                    body = (gw - bot.body_w).length() < BODY_R
                    if head or body:
                        self.land(s, head, game, bot)
                        self.pt[s] = 0

        self.combo_t -= dt
        if self.combo_t <= 0:
            self.combo = 0
        self.shake = max(0.0, self.shake - dt)
        if self.shake > 0:
            camera.position = Vec3(rnd.uniform(-1, 1), rnd.uniform(-1, 1), 0) * self.shake * 0.12
        else:
            camera.position = Vec3(0, 0, 0)

    def land(self, side, head, game, bot):
        kind, speed = self.pkind[side], self.pspeed[side]
        dmg = {'straight': 7.0, 'hook': 9.0, 'uppercut': 10.0}[kind]
        if not head:
            dmg *= 0.6
        dmg *= clamp(speed / 5.0, 0.7, 1.5)
        txt = ('HEAD SHOT!' if head else 'BODY SHOT!')
        if bot.state == 'block':
            dmg *= 0.2
            txt = 'BLOCKED'
        elif bot.vuln > 0:
            dmg *= 1.5
            txt = 'COUNTER!'
        self.combo += 1
        self.combo_t = 1.6
        if self.combo >= 3:
            txt += f'  x{self.combo} COMBO'
        bot.take_hit(dmg, head)
        game.popup(txt, rgb(255, 210, 60) if bot.state != 'block' else rgb(180, 180, 200))


# ----------------------------------------------------------------------------
# THE BOT
# ----------------------------------------------------------------------------
B_SKIN = rgb(120, 78, 56)
B_GLOVE = rgb(30, 80, 230)
B_WARN = rgb(255, 215, 30)
B_TRUNK = rgb(20, 60, 190)
B_FLASH = rgb(240, 90, 80)


class Bot:
    def __init__(self):
        R = self.root = Entity()

        def S(**kw):
            return Entity(parent=R, model='sphere', **kw)

        self.torso = S(color=B_SKIN, scale=(0.62, 0.74, 0.38))
        self.trunks = S(color=B_TRUNK, scale=(0.56, 0.38, 0.38))
        self.waist = Entity(parent=R, model='cube', color=WHITE, scale=(0.52, 0.07, 0.34))
        self.neck = S(color=B_SKIN, scale=(0.15, 0.20, 0.15))
        self.head = S(color=B_SKIN, scale=(0.27, 0.33, 0.30))
        self.hair = S(color=BLACK, scale=(0.29, 0.20, 0.32))
        self.eyes = [S(color=WHITE, scale=0.055) for _ in range(2)]
        self.pupils = [S(color=BLACK, scale=0.03) for _ in range(2)]
        self.sh = [S(color=B_SKIN, scale=0.2) for _ in range(2)]
        self.up = [S(color=B_SKIN) for _ in range(2)]
        self.fore = [S(color=B_SKIN) for _ in range(2)]
        self.glove = [S(color=B_GLOVE, scale=(0.23, 0.23, 0.26)) for _ in range(2)]
        self.cuff = [S(color=WHITE, scale=0.14) for _ in range(2)]
        self.thigh = [S(color=B_SKIN) for _ in range(2)]
        self.shin = [S(color=B_SKIN) for _ in range(2)]
        self.shoe = [S(color=WHITE, scale=(0.15, 0.10, 0.30)) for _ in range(2)]
        self._gcol = [None, None]
        self._hflash = False
        self.head_w = Vec3(0, 1.7, 1)
        self.body_w = Vec3(0, 1.3, 1)
        self.reset()

    def reset(self):
        self.hp = 100.0
        self.state = 'idle'
        self.timer = 1.2
        self.dur = 1.0
        self.t = rnd.uniform(0, 6)
        self.p = self.w = 0.0
        self.kind, self.hand = 'jab', 1
        self.landed = False
        self.hit_t = 0.0
        self.vuln = 0.0
        self.ko_t = 0.0
        self.dist = 2.0
        self.x = 0.0
        self.root.rotation_x = 0
        self.root.y = 0

    def set(self, state, dur):
        self.state, self.timer, self.dur = state, dur, max(dur, 1e-3)

    def begin_attack(self, game, fast=False):
        r = rnd.random()
        self.kind = 'jab' if r < 0.45 else 'hook' if r < 0.75 else 'body'
        self.hand = rnd.choice((-1, 1))
        base = max(0.26, 0.62 - 0.09 * (game.round - 1))
        self.set('windup', base * (0.65 if fast else 1.0) * rnd.uniform(0.9, 1.15))

    def take_hit(self, dmg, head):
        if self.state == 'ko':
            return
        self.hp = max(0.0, self.hp - dmg)
        if self.state != 'block':
            self.hit_t = 0.25
            if self.state in ('windup', 'idle', 'recover') and (head or dmg > 6):
                self.set('stagger', 0.30)
        if self.hp <= 0:
            self.state = 'ko'
            self.ko_t = 0.0

    # -- brain ---------------------------------------------------------------------------
    def ai(self, dt, game, player):
        self.timer -= dt
        st = self.state
        if st == 'idle':
            if self.timer <= 0:
                recent = pytime.perf_counter() - player.last_punch_time < 0.6
                if recent and rnd.random() < 0.35:
                    self.set('block', rnd.uniform(0.6, 1.1))
                else:
                    self.begin_attack(game)
        elif st == 'windup':
            self.w = clamp(1 - self.timer / self.dur, 0, 1)
            if self.timer <= 0:
                self.set('punch', 0.17)
                self.landed = False
        elif st == 'punch':
            self.p = clamp(1 - self.timer / self.dur, 0, 1)
            if not self.landed and self.p >= 0.55:
                self.landed = True
                game.bot_punch_lands(self.kind)
            if self.timer <= 0:
                if rnd.random() < 0.12 + 0.10 * game.round:
                    self.begin_attack(game, fast=True)
                else:
                    self.set('recover', 0.45)
        elif st in ('recover', 'block', 'stagger'):
            if self.timer <= 0:
                self.set('idle', rnd.uniform(0.5, 1.2) / (1 + 0.2 * (game.round - 1)))

    # -- body -----------------------------------------------------------------------------
    def update(self, dt, game, player):
        self.t += dt
        self.hit_t = max(0.0, self.hit_t - dt)
        self.vuln = max(0.0, self.vuln - dt)
        st = self.state
        if st == 'ko':
            self.ko_t += dt
        elif game.state == 'fight':
            self.ai(dt, game, player)
        elif st != 'idle':
            self.set('idle', 1.0)
        st = self.state

        dist_t = {'idle': 1.8 + 0.18 * math.sin(self.t * 0.9), 'windup': 1.55, 'punch': 1.25, 'recover': 1.9,
                  'block': 1.7, 'stagger': 2.1, 'ko': 2.7}.get(st, 2.0)
        self.dist += (dist_t - self.dist) * min(1, dt * (9 if st == 'punch' else 3))
        x_t = 0.55 * math.sin(self.t * 0.55)
        if st in ('windup', 'punch'):
            x_t = x_t * 0.3 + player.rig.x * 0.6
        self.x += (x_t - self.x) * min(1, dt * 3)
        self.root.position = Vec3(self.x, self.root.y, player.rig.z + self.dist)
        if st == 'ko':
            ang = min(90.0, self.ko_t * 170)
            self.root.rotation_x = ang
            self.root.y = 0.15 * (ang / 90)
        self.pose(player)

    def pose(self, player):
        t, st = self.t, self.state
        R = self.root.position
        bob = math.sin(t * 6.0) * 0.022
        q = math.sin(math.pi * clamp(self.p, 0, 1)) if st == 'punch' else 0.0
        lunge = -0.12 * q + (0.05 * self.w if st == 'windup' else 0.0)
        hk = self.hit_t / 0.25
        sway = math.sin(t * 2.1) * 0.03
        H = Vec3(sway, 1.72 + bob, lunge * 0.8 + 0.18 * hk)
        C = Vec3(sway * 0.6, 1.28 + bob, lunge * 0.5 + 0.05 * hk)
        self.head_w, self.body_w = R + H, R + C
        self.torso.position = C
        self.trunks.position = Vec3(sway * 0.3, 0.98 + bob * 0.5, 0)
        self.waist.position = Vec3(sway * 0.3, 1.12 + bob * 0.5, 0)
        self.neck.position = Vec3(H.x * 0.6, 1.52 + bob, H.z * 0.5)
        self.head.position = H
        self.hair.position = H + Vec3(0, 0.10, 0.03)
        for i, s in enumerate((-1, 1)):
            self.eyes[i].position = H + Vec3(s * 0.07, 0.03, -0.135)
            self.pupils[i].position = H + Vec3(s * 0.07, 0.03, -0.158)
        flash = self.hit_t > 0
        if flash != self._hflash:
            self._hflash = flash
            self.head.color = B_FLASH if flash else B_SKIN

        aim_head = player.rig.position - R
        for i, s in enumerate((-1, 1)):
            sho = Vec3(s * 0.28 + sway * 0.6, 1.5 + bob, lunge * 0.5)
            guard = Vec3(s * 0.17 + sway, 1.55 + bob, -0.30 + lunge)
            pull = guard + Vec3(s * 0.10, -0.12, 0.28)
            g = Vec3(s * 0.10 + sway, 1.72 + bob, -0.34) if st == 'block' else guard
            active = (s == self.hand and st in ('windup', 'punch'))
            if active:
                if st == 'windup':
                    g = guard + (pull - guard) * self.w
                else:
                    tgt = (Vec3(player.rig.x, 1.0, player.rig.z) - R) if self.kind == 'body' else aim_head
                    dv = tgt - pull
                    E = tgt - dv * (0.35 / max(dv.length(), 1e-3))
                    if self.kind == 'hook':
                        P1 = Vec3(s * 0.95, E.y, (pull.z + E.z) * 0.5)
                        g = pull * ((1 - q) ** 2) + P1 * (2 * (1 - q) * q) + E * (q * q)
                    else:
                        g = pull + (E - pull) * q
            el = (sho + g) * 0.5 + Vec3(s * (0.14 - 0.10 * q), -0.16, 0.04)
            self.sh[i].position = sho
            place(self.up[i], sho, el, 0.11)
            place(self.fore[i], el, g, 0.10)
            self.glove[i].position = g
            self.cuff[i].position = el + (g - el) * 0.78
            col = B_WARN if active else B_GLOVE
            if col != self._gcol[i]:
                self._gcol[i] = col
                self.glove[i].color = col

            hip = Vec3(s * 0.12, 0.95, 0)
            fz = math.sin(t * 3.0 + (0 if s < 0 else math.pi)) * 0.07 - (0.10 if st == 'punch' else 0.0)
            foot = Vec3(s * 0.17, 0.07, fz)
            knee = (hip + foot) * 0.5 + Vec3(0, 0.02, -0.08)
            place(self.thigh[i], hip, knee, 0.17)
            place(self.shin[i], knee, foot, 0.13)
            self.shoe[i].position = foot + Vec3(0, 0, -0.05)


player = Player()
bot = Bot()


# ----------------------------------------------------------------------------
# HUD
# ----------------------------------------------------------------------------
Text(text='Punch with your body | fists up = block | lean / duck = dodge | C recalibrate | ENTER restart | ESC quit',
     position=window.top_left + Vec2(0.02, -0.02), scale=0.8)
status = Text(text='', position=window.bottom_left + Vec2(0.02, 0.10), scale=0.9)

ui_quad(position=(-0.82, 0.43), origin=(-0.5, 0), scale=(0.57, 0.045), color=rgb(15, 15, 20))
ui_quad(position=(0.82, 0.43), origin=(0.5, 0), scale=(0.57, 0.045), color=rgb(15, 15, 20))
bar_p = ui_quad(position=(-0.815, 0.43), origin=(-0.5, 0), scale=(0.56, 0.032), color=rgb(60, 220, 90), z=-0.01)
bar_b = ui_quad(position=(0.815, 0.43), origin=(0.5, 0), scale=(0.56, 0.032), color=rgb(60, 120, 255), z=-0.01)
Text(text='YOU', position=(-0.82, 0.475), origin=(-0.5, 0), scale=1.3)
Text(text='IRON BOT', position=(0.82, 0.475), origin=(0.5, 0), scale=1.3)
timer_text = Text(text='', position=(0, 0.46), origin=(0, 0), scale=1.6)
msg_text = Text(text='', position=(0, 0.14), origin=(0, 0), scale=3)
pop_text = Text(text='', position=(0, -0.08), origin=(0, 0), scale=2.0)
guard_text = Text(text='', position=(0, -0.40), origin=(0, 0), scale=1.4)
red_flash = ui_quad(scale=(3, 2), color=color.hsv(0, 0.9, 0.8, 0), z=1)
panel = ui_quad(scale=(3, 2), color=color.hsv(0, 0, 0, 0.72), z=0.5)
title_a = Text(text='BOXING RING 3D', origin=(0, 0), position=(0, 0.14), scale=4)
title_b = Text(text='Stand ~2 m from the webcam (head, shoulders and both arms visible)\n'
                    'Throw real punches | fists up = block | lean or duck = dodge\n\n'
                    'Press ENTER to start', origin=(0, 0), position=(0, -0.05), scale=1.4)

# live webcam preview (bottom-right)
_ptex = PandaTexture()
_ptex.setup2dTexture(PREVIEW_W, PREVIEW_H, PandaTexture.TUnsignedByte, PandaTexture.FRgb)
_ptex.setRamImageAs(bytes(PREVIEW_W * PREVIEW_H * 3), 'RGB')
_pw, _ph = PREVIEW_SCALE, PREVIEW_SCALE * PREVIEW_H / PREVIEW_W
_pv_pos = window.bottom_right + Vec2(-(_pw / 2 + 0.02), _ph / 2 + 0.02)
ui_quad(position=_pv_pos, scale=(_pw + 0.01, _ph + 0.01), color=color.black)
preview_quad = ui_quad(position=_pv_pos, scale=(_pw, _ph))
preview_quad.texture = Texture(_ptex)
_last_pv = [-1]


# ----------------------------------------------------------------------------
# GAME FLOW
# ----------------------------------------------------------------------------
class Game:
    def __init__(self):
        self.state = 'wait'            # wait -> calib -> fight -> between -> ... -> over
        self.round = 1
        self.timer = ROUND_TIME
        self.sub = 0.0
        self.player_hp = 100.0
        self.msg_t = 0.0
        self.pop_t = 0.0
        self.flash = 0.0
        self.over_t = 0.0

    def show_msg(self, text, dur=1.6, col=None):
        msg_text.text = text
        msg_text.color = col or WHITE
        self.msg_t = dur

    def popup(self, text, col):
        pop_text.text = text
        pop_text.color = col
        self.pop_t = 0.8

    def start(self):
        self.round = 1
        self.player_hp = 100.0
        bot.reset()
        player.calibrate()
        player.rig.rotation_z = 0
        self.state, self.sub = 'calib', CALIB_TIME
        panel.enabled = title_a.enabled = title_b.enabled = False
        self.show_msg('GET IN YOUR STANCE\nhold still...', CALIB_TIME)

    def begin_round(self):
        self.state = 'fight'
        self.timer = ROUND_TIME
        bot.state, bot.timer = 'idle', 1.0
        self.show_msg(f'ROUND {self.round}\nFIGHT!', 1.4, rgb(255, 210, 60))

    def bot_punch_lands(self, kind):
        lean_ok = player.rl >= LEAN_T
        duck_ok = player.rd >= DUCK_T
        dodged = (lean_ok or duck_ok) if kind == 'jab' else duck_ok if kind == 'hook' else lean_ok
        base = {'jab': 6.0, 'hook': 10.0, 'body': 7.0}[kind] * (1 + 0.1 * (self.round - 1))
        if dodged:
            self.popup('DODGED!  counter now!', rgb(90, 200, 255))
            bot.vuln = 1.0
            bot.set('recover', 0.8)
            return
        if player.guarding:
            base *= 0.3 if kind != 'body' else 0.7
            self.popup('BLOCKED', rgb(180, 180, 200))
            self.flash = max(self.flash, 0.12)
            player.shake = 0.12
        else:
            self.popup('OUCH!', rgb(255, 70, 60))
            self.flash = 0.45
            player.shake = 0.3
        self.player_hp = max(0.0, self.player_hp - base)

    def end_fight(self, text, col):
        self.state = 'over'
        self.over_t = 0.0
        self.show_msg(text + '\nENTER = rematch', 999, col)

    def update(self, dt):
        if self.state == 'calib':
            self.sub -= dt
            if self.sub <= 0:
                self.begin_round()
        elif self.state == 'fight':
            self.timer -= dt
            if bot.hp <= 0:
                if bot.ko_t > 1.8:
                    self.end_fight('K.O.!  YOU WIN!', rgb(255, 210, 60))
            elif self.player_hp <= 0:
                self.end_fight('YOU WERE KNOCKED OUT', rgb(255, 70, 60))
            elif self.timer <= 0:
                if self.round >= ROUNDS:
                    if self.player_hp > bot.hp:
                        self.end_fight('YOU WIN BY DECISION!', rgb(255, 210, 60))
                    elif self.player_hp < bot.hp:
                        self.end_fight('YOU LOSE BY DECISION', rgb(255, 70, 60))
                    else:
                        self.end_fight('DRAW', WHITE)
                else:
                    self.state, self.sub = 'between', 4.0
                    self.player_hp = min(100.0, self.player_hp + 20)
                    bot.hp = min(100.0, bot.hp + 20)
                    self.show_msg(f'END OF ROUND {self.round}', 3.5)
        elif self.state == 'between':
            self.sub -= dt
            if self.sub <= 0:
                self.round += 1
                self.begin_round()
        elif self.state == 'over':
            self.over_t += dt
            if self.player_hp <= 0:                        # fall over
                player.rig.rotation_z += (35 - player.rig.rotation_z) * min(1, dt * 3)
                player.rig.y += (0.7 - player.rig.y) * min(1, dt * 3)

        # HUD
        self.msg_t -= dt
        if self.msg_t <= 0:
            msg_text.text = ''
        self.pop_t -= dt
        if self.pop_t <= 0:
            pop_text.text = ''
        self.flash = max(0.0, self.flash - dt)
        red_flash.color = color.hsv(0, 0.9, 0.8, min(0.5, self.flash))
        bar_p.scale_x = max(0.001, 0.56 * self.player_hp / 100)
        bar_b.scale_x = max(0.001, 0.56 * bot.hp / 100)
        bar_p.color = rgb(60, 220, 90) if self.player_hp > 35 else rgb(240, 70, 50)
        bar_b.color = rgb(60, 120, 255) if bot.hp > 35 else rgb(240, 70, 50)
        tl = max(0, int(self.timer))
        timer_text.text = f'ROUND {self.round}/{ROUNDS}   {tl // 60}:{tl % 60:02d}' if self.state in ('fight', 'between') else ''
        guard_text.text = 'GUARD' if player.guarding else ''


game = Game()


def update():
    dt = min(time.dt, 1 / 30)
    if tracker.preview is not None and tracker.frame_id != _last_pv[0]:
        _ptex.setRamImageAs(tracker.preview, 'RGB')
        _last_pv[0] = tracker.frame_id

    data = tracker.get()
    player.update(dt, data, game.state == 'fight', game, bot)
    bot.update(dt, game, player)
    game.update(dt)

    if tracker.error:
        line = tracker.error
    elif data:
        line = 'Body tracked' + ('' if player.base else ' (calibrating...)')
    else:
        line = 'NO BODY DETECTED - step back so your arms are visible'
    status.text = f'{line}\nlean {player.lean:+.2f}  duck {player.duck:+.2f}'


def input(key):
    if key == 'enter' and game.state in ('wait', 'over'):
        game.start()
    if key == 'c':
        player.calibrate()
    if key in ('left mouse down', 'f'):
        if game.state == 'fight':
            player.fake_punch('L')
    if key in ('right mouse down', 'j'):
        if game.state == 'fight':
            player.fake_punch('R')
    if key == 'escape':
        tracker.stop()
        application.quit()


app.run()

import sys, cv2, numpy as np
img = cv2.imread(sys.argv[1]); H, Wd = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); S, V = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
box = ((V > 165) | ((S > 90) & (V > 70))).astype(np.uint8); box[:, 1150:] = 0
box = cv2.morphologyEx(box, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(box); box = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
side = (((S > 110) & (V < 200)) & (box > 0)).astype(np.uint8)
side = cv2.morphologyEx(side, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(side); side = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
top = box & (1 - cv2.dilate(side, np.ones((3, 3), np.uint8)))
top = cv2.morphologyEx(top, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(top); top = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)

def fit_quad(mask):
    c = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    hull = cv2.convexHull(c)
    for eps in np.linspace(0.005, 0.06, 60):
        a = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(a) == 4: break
    q = a.reshape(4, 2).astype(float)
    # refine: fit a line to contour points near each side, intersect neighbours
    pts = c.reshape(-1, 2).astype(float); lines = []
    for i in range(4):
        p, r = q[i], q[(i + 1) % 4]; d = (r - p) / np.linalg.norm(r - p); nrm = np.array([-d[1], d[0]])
        t = (pts - p) @ d; L = np.linalg.norm(r - p)
        sel = pts[(np.abs((pts - p) @ nrm) < 6) & (t > 0.1 * L) & (t < 0.9 * L)]
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append(np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0]))
    out = []
    for i in range(4):
        X = np.cross(lines[i - 1], lines[i]); out.append(X[:2] / X[2])
    return np.array(out), lines

T, tl = fit_quad(top)
# order so T[0] is the top-most corner, going clockwise
T = np.roll(T, -int(np.argmin(T[:, 1])), axis=0)
print("top corners", T.round(1).tolist())
c = np.array([Wd / 2, H / 2])
def vp(a, b, c_, d):
    l1 = np.cross([*a, 1], [*b, 1]); l2 = np.cross([*c_, 1], [*d, 1]); X = np.cross(l1, l2); return X[:2] / X[2]
v1 = vp(T[0], T[1], T[3], T[2]); v2 = vp(T[0], T[3], T[1], T[2])
f2 = -np.dot(v1 - c, v2 - c)
print("vanishing pts", v1.round(0), v2.round(0), "f from VPs:", np.sqrt(f2) if f2 > 0 else "unstable")

def recon(f):
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]]); Ki = np.linalg.inv(K)
    d1 = Ki @ [*v1, 1]; d1 /= np.linalg.norm(d1); d2 = Ki @ [*v2, 1]; d2 /= np.linalg.norm(d2)
    nrm = np.cross(d1, d2); nrm /= np.linalg.norm(nrm)
    rays = [Ki @ [*p, 1] for p in T]
    X = [r / (nrm @ r) for r in rays]  # plane n.X = 1
    e01 = np.linalg.norm(X[1] - X[0]); e12 = np.linalg.norm(X[2] - X[1]); e23 = np.linalg.norm(X[3] - X[2]); e30 = np.linalg.norm(X[0] - X[3])
    ang = np.degrees(np.arccos(abs(np.dot((X[1] - X[0]) / e01, (X[3] - X[0]) / e30))))
    return K, nrm, X, (e01 + e23) / 2, (e12 + e30) / 2, ang

# focal-length hypotheses: VP estimate, iPhone wide (~69deg HFOV), ultra-wide crop (~100deg)
fs = {"vp": float(np.sqrt(f2)) if f2 > 0 else None, "wide69": Wd / 2 / np.tan(np.radians(69 / 2)),
      "uw100": Wd / 2 / np.tan(np.radians(100 / 2)), "tele_like50": Wd / 2 / np.tan(np.radians(50 / 2))}
side_pts = cv2.findNonZero(side).reshape(-1, 2).astype(float)
for k, f in fs.items():
    if f is None: continue
    K, nrm, X, a, b, ang = recon(f)
    up = nrm if nrm[2] < 0 else -nrm  # camera z forward; "up" from table points toward camera
    # height: side face is the band between top edge and the parallel bottom edge offset by -h*up.
    # project top-edge points shifted down by h and pick h that best covers the side mask's outer boundary
    best = None
    for h in np.linspace(0.02, 0.8, 400) * min(a, b):
        cover = []
        for P in X:
            Q = K @ (P - h * up); cover.append(Q[:2] / Q[2])
        # far edge of side band: distance from side pixels to shifted top quad edges; maximise fit of extreme
        poly = np.array(cover + list(T)).astype(np.float32)
        hullp = cv2.convexHull(poly).reshape(-1, 2)
        inside = cv2.pointPolygonTest(hullp.reshape(-1, 1, 2), (0, 0), False)
        frac = np.mean([cv2.pointPolygonTest(hullp.reshape(-1, 1, 2), tuple(map(float, p)), False) >= 0 for p in side_pts[::25]])
        if frac > 0.985 and best is None: best = h
    L, Wb = max(a, b), min(a, b)
    print(f"f={f:7.1f} ({k}): L/W={L / Wb:.3f}  corner angle={ang:.1f}deg  H/W={best / Wb if best else float('nan'):.3f}  tilt(deg from straight-down)={np.degrees(np.arccos(abs(nrm[2]))):.1f}")
np.save("top_corners.npy", T)
vis = img.copy(); vis[side > 0] = (0.5 * vis[side > 0] + [0, 0, 120]).astype(np.uint8)
cv2.polylines(vis, [T.astype(np.int32)], True, (0, 255, 0), 2)
for i, p in enumerate(T): cv2.circle(vis, tuple(p.astype(int)), 6, (0, 0, 255), -1); cv2.putText(vis, f"T{i}", tuple((p + 8).astype(int)), 0, 1, (0, 255, 0), 2)
cv2.imwrite("geom_vis.jpg", vis[100:900, 250:1100])

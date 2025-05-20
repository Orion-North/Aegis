import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time

# ——— Configuration ———
CAM_WIDTH      = 640
CAM_HEIGHT     = 480
MIN_DET_CONF   = 0.6
MIN_TRACK_CONF = 0.7
ALPHA          = 0.7    # smoothing for pose angles
ZOOM_FACTOR    = 2.0    # how “zoomed” the turret view is
ARROW_SIZE     = 40
ARROW_MARGIN   = 60
CENTER_THRESH  = 0.05   # fraction of frame size to start arrows

# ——— Pan physics gains ———
P_GAIN = 20.0  # proportional “spring” constant
D_GAIN = 5.0   # damping coefficient

# ——— State variables ———
smoothed_yaw = smoothed_pitch = smoothed_roll = None
# pan position & velocity (in full-frame coords)
cam_x = CAM_WIDTH / 2
cam_y = CAM_HEIGHT / 2
vel_x = vel_y = 0.0
prev_time = time.time()

# last detected head center
last_cx = CAM_WIDTH / 2
last_cy = CAM_HEIGHT / 2

# Start video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# Load models
yolo      = YOLO('yolov8n-face.pt')
mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF
)

# 3D model points for solvePnP
model_pts = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

# FPS timer
fps_start = time.time()
frame_count = 0

while cap.isOpened():
    # ——— Time & dt ———
    now = time.time()
    dt = now - prev_time
    prev_time = now

    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    h, w = frame.shape[:2]

    # ——— 1) Detect face center ———
    det = yolo(frame, conf=0.5)[0]
    if det.boxes:
        x1, y1, x2, y2 = det.boxes[0].xyxy.cpu().numpy().flatten().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        last_cx = (x1 + x2) / 2
        last_cy = (y1 + y2) / 2

        # crop & mesh for tight head box & pose
        roi = frame[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([
                (int(p.x * rw) + x1, int(p.y * rh) + y1)
                for p in lm
            ])
            bx1, by1 = pts.min(axis=0)
            bx2, by2 = pts.max(axis=0)
        else:
            bx1, by1, bx2, by2 = x1, y1, x2, y2

        # pose on 6 keypoints
        image_pts = np.array([
            pts[1], pts[152], pts[33],
            pts[263], pts[61], pts[291]
        ], dtype=np.float64)
        cam_mtx = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
        dist = np.zeros((4,1))
        _, rvec, tvec = cv2.solvePnP(
            model_pts, image_pts, cam_mtx, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        R, _ = cv2.Rodrigues(rvec)
        P = cv2.hconcat((R, tvec))
        _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(P)
        yaw, pitch, roll = angles.flatten()

        # smooth angles
        if smoothed_yaw is None:
            smoothed_yaw, smoothed_pitch, smoothed_roll = yaw, pitch, roll
        else:
            smoothed_yaw   = ALPHA*smoothed_yaw   + (1-ALPHA)*yaw
            smoothed_pitch = ALPHA*smoothed_pitch + (1-ALPHA)*pitch
            smoothed_roll  = ALPHA*smoothed_roll  + (1-ALPHA)*roll

    # ——— 2) Pan acceleration/braking toward last_cx, last_cy ———
    err_x = last_cx - cam_x
    err_y = last_cy - cam_y
    accel_x = P_GAIN*err_x - D_GAIN*vel_x
    accel_y = P_GAIN*err_y - D_GAIN*vel_y
    vel_x  += accel_x * dt
    vel_y  += accel_y * dt
    cam_x  += vel_x  * dt
    cam_y  += vel_y  * dt

    # ——— 3) Zoom‐crop around (cam_x, cam_y) ———
    crop_w = int(w  / ZOOM_FACTOR)
    crop_h = int(h  / ZOOM_FACTOR)
    x0 = int(cam_x - crop_w/2)
    y0 = int(cam_y - crop_h/2)
    x0 = max(0, min(w-crop_w, x0))
    y0 = max(0, min(h-crop_h, y0))
    zoom_roi = frame[y0:y0+crop_h, x0:x0+crop_w]
    view     = cv2.resize(zoom_roi, (w, h))

    # ——— 4) Draw head box into view ———
    s = ZOOM_FACTOR
    vx1 = int((bx1 - x0)*s); vy1 = int((by1 - y0)*s)
    vx2 = int((bx2 - x0)*s); vy2 = int((by2 - y0)*s)
    cv2.rectangle(view, (vx1,vy1), (vx2,vy2), (0,255,0), 2)

    # ——— 5) Arrows if still off‐center ———
    thx = CENTER_THRESH * w
    thy = CENTER_THRESH * h
    if err_x >  thx:
        cv2.arrowedLine(view,
                        (w-ARROW_MARGIN, h//2),
                        (w-ARROW_MARGIN-ARROW_SIZE, h//2),
                        (0,0,255), 2, tipLength=0.3)
    elif err_x < -thx:
        cv2.arrowedLine(view,
                        (ARROW_MARGIN, h//2),
                        (ARROW_MARGIN+ARROW_SIZE, h//2),
                        (0,0,255), 2, tipLength=0.3)
    if err_y >  thy:
        cv2.arrowedLine(view,
                        (w//2, h-ARROW_MARGIN),
                        (w//2, h-ARROW_MARGIN-ARROW_SIZE),
                        (0,0,255), 2, tipLength=0.3)
    elif err_y < -thy:
        cv2.arrowedLine(view,
                        (w//2, ARROW_MARGIN),
                        (w//2, ARROW_MARGIN+ARROW_SIZE),
                        (0,0,255), 2, tipLength=0.3)

    # ——— 6) Overlay smoothed pose ———
    if smoothed_yaw is not None:
        cv2.putText(view,
                    f"Yaw:{smoothed_yaw:.1f} Pitch:{smoothed_pitch:.1f} Roll:{smoothed_roll:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ——— 7) Display & FPS ———
    cv2.imshow("Turret Simulation (Smoothed)", view)
    frame_count += 0  # (we already incremented)
    if frame_count >= 30:
        fps = frame_count / (time.time() - fps_start)
        print(f"≈ {fps:.1f} FPS")
        fps_start = time.time()
        frame_count = 0

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

import math
import time
from typing import List
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Robotic Arm Kinematics & URDF Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

research_benchmarks = []

class LinkConfig(BaseModel):
    length: float
    yaw: float = 0.0
    pitch: float = 0.0

class TrajectoryRequest(BaseModel):
    shape: str
    links: List[LinkConfig]
    motor_speed_dps: float = 90.0
    num_samples: int = 140

def dist3d(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2])

# Multi-Link Inverse Kinematics returning physical joint angles (for URDF revolute joints)
def solve_arm_angles(target_x: float, target_y: float, target_z: float, lengths: List[float], bx=0.0, by=0.3, bz=0.0):
    dx = target_x - bx
    dy = target_y - by
    dz = target_z - bz

    base_yaw = math.atan2(dx, dz)
    horizontal_dist = math.hypot(dx, dz)
    direct_dist = math.hypot(horizontal_dist, dy)
    
    total_reach = sum(lengths)
    is_reachable = True
    
    if direct_dist > (total_reach * 0.98):
        is_reachable = False
        ratio = (total_reach * 0.98) / direct_dist
        horizontal_dist *= ratio
        dy *= ratio
        direct_dist = total_reach * 0.98

    n = len(lengths)
    l1 = lengths[0]
    l_middle = sum(lengths[1:-1]) if n > 2 else (lengths[1] if n > 1 else 0.1)
    l_pen = lengths[-1] if n > 2 else (lengths[1] if n > 1 else 0.1)

    wrist_r = horizontal_dist
    wrist_y = dy + l_pen
    dist_to_wrist = math.hypot(wrist_r, wrist_y)

    reach_arm = l1 + l_middle
    if dist_to_wrist > reach_arm * 0.98:
        wrist_r *= (reach_arm * 0.98) / dist_to_wrist
        wrist_y *= (reach_arm * 0.98) / dist_to_wrist
        dist_to_wrist = reach_arm * 0.98

    cos_elbow = (dist_to_wrist**2 - l1**2 - l_middle**2) / (2 * l1 * l_middle)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    q_elbow = math.acos(cos_elbow)

    cos_shoulder = (l1**2 + dist_to_wrist**2 - l_middle**2) / (2 * l1 * dist_to_wrist)
    cos_shoulder = max(-1.0, min(1.0, cos_shoulder))
    phi = math.atan2(wrist_y, wrist_r)
    q_shoulder = phi + math.acos(cos_shoulder)

    # Build joints in 2D reach plane
    points_2d = [(0.0, 0.0)]
    curr_r = l1 * math.cos(q_shoulder)
    curr_py = l1 * math.sin(q_shoulder)
    points_2d.append((curr_r, curr_py))

    num_mid = max(1, n - 2)
    bend_step = q_elbow / num_mid
    curr_angle = q_shoulder
    for k in range(1, n - 1):
        curr_angle -= bend_step
        curr_r += lengths[k] * math.cos(curr_angle)
        curr_py += lengths[k] * math.sin(curr_angle)
        points_2d.append((curr_r, curr_py))

    points_2d.append((horizontal_dist, dy))

    joints_3d = []
    sin_yaw, cos_yaw = math.sin(base_yaw), math.cos(base_yaw)
    for p in points_2d:
        joints_3d.append({
            "x": round(bx + p[0] * sin_yaw, 3),
            "y": round(by + p[1], 3),
            "z": round(bz + p[0] * cos_yaw, 3)
        })

    angles = {
        "joint_waist": round(base_yaw, 4),
        "joint_shoulder": round(q_shoulder - (math.pi / 2), 4),
        "joint_elbow": round(math.pi - q_elbow, 4),
        "joint_wrist": round(-q_shoulder + q_elbow, 4)
    }

    can_draw = is_reachable and (abs(joints_3d[-1]["y"] - target_y) < 0.25)
    return joints_3d, angles, can_draw, is_reachable

# Dynamic URDF Generator based on current Link parameters
def build_urdf_xml(lengths: List[float]):
    xml = '<?xml version="1.0"?>\n'
    xml += '<robot name="research_arm_3d">\n'
    
    # Base link
    xml += '  <link name="base_link">\n'
    xml += '    <visual>\n'
    xml += '      <geometry><cylinder length="0.3" radius="1.1"/></geometry>\n'
    xml += '      <material name="dark_grey"><color rgba="0.2 0.25 0.3 1.0"/></material>\n'
    xml += '    </visual>\n'
    xml += '  </link>\n'

    # Revolute Joint 1 (Waist Yaw)
    xml += '  <joint name="joint_waist" type="revolute">\n'
    xml += '    <parent link="base_link"/>\n'
    xml += '    <child link="link_1"/>\n'
    xml += '    <origin xyz="0 0 0.15" rpy="0 0 0"/>\n'
    xml += '    <axis xyz="0 1 0"/>\n'
    xml += '    <limit lower="-3.14" upper="3.14" effort="10.0" velocity="3.14"/>\n'
    xml += '  </joint>\n'

    for i, l in enumerate(lengths):
        link_name = f"link_{i+1}"
        xml += f'  <link name="{link_name}">\n'
        xml += '    <visual>\n'
        xml += f'      <origin xyz="0 {l/2:.3f} 0" rpy="0 0 0"/>\n'
        xml += f'      <geometry><cylinder length="{l:.3f}" radius="0.12"/></geometry>\n'
        xml += f'      <material name="blue_{i}"><color rgba="0.1 0.4 {0.5 + i*0.1:.2f} 1.0"/></material>\n'
        xml += '    </visual>\n'
        xml += '  </link>\n'

        if i < len(lengths) - 1:
            j_name = "joint_shoulder" if i == 0 else (f"joint_elbow" if i == 1 else f"joint_{i+1}")
            child_name = f"link_{i+2}"
            xml += f'  <joint name="{j_name}" type="revolute">\n'
            xml += f'    <parent link="{link_name}"/>\n'
            xml += f'    <child link="{child_name}"/>\n'
            xml += f'    <origin xyz="0 {l:.3f} 0" rpy="0 0 0"/>\n'
            xml += '    <axis xyz="1 0 0"/>\n'
            xml += '    <limit lower="-2.5" upper="2.5" effort="10.0" velocity="3.14"/>\n'
            xml += '  </joint>\n'

    xml += '</robot>'
    return xml

@app.post("/api/arm/urdf")
def get_urdf(req: TrajectoryRequest):
    lengths = [l.length for l in req.links]
    urdf_text = build_urdf_xml(lengths)
    return Response(content=urdf_text, media_type="application/xml")

@app.post("/api/arm/trajectory")
def generate_trajectory(req: TrajectoryRequest):
    shape = req.shape
    n_samples = req.num_samples
    lengths = [l.length for l in req.links]
    total_reach = sum(lengths)

    table_y = 0.02
    scale = min(total_reach * 0.45, 5.0)

    path_points = []
    frames = []

    for i in range(n_samples + 1):
        t = (i / n_samples) * 2 * math.pi

        if shape == "circle":
            tx = scale * math.cos(t)
            tz = scale * math.sin(t)
            ty = table_y
        elif shape == "square":
            side = (i / n_samples) * 4.0
            half = scale * 0.75
            if side < 1.0:
                tx, tz = -half + side * (2 * half), -half
            elif side < 2.0:
                tx, tz = half, -half + (side - 1.0) * (2 * half)
            elif side < 3.0:
                tx, tz = half - (side - 2.0) * (2 * half), half
            else:
                tx, tz = -half, half - (side - 3.0) * (2 * half)
            ty = table_y
        elif shape == "star":
            num_points = 10
            angle_step = (2 * math.pi) / num_points
            seg = int(t / angle_step)
            frac = (t % angle_step) / angle_step
            r1 = scale if (seg % 2 == 0) else (scale * 0.45)
            r2 = (scale * 0.45) if (seg % 2 == 0) else scale
            a1 = seg * angle_step - math.pi / 2
            a2 = (seg + 1) * angle_step - math.pi / 2
            tx = r1 * math.cos(a1) + frac * (r2 * math.cos(a2) - r1 * math.cos(a1))
            tz = r1 * math.sin(a1) + frac * (r2 * math.sin(a2) - r1 * math.sin(a1))
            ty = table_y
        elif shape == "helix":
            tx = (scale * 0.7) * math.cos(t * 3)
            tz = (scale * 0.7) * math.sin(t * 3)
            ty = table_y + (i / n_samples) * (total_reach * 0.4)
        elif shape == "lissajous":
            tx = scale * math.sin(3 * t)
            tz = scale * math.cos(2 * t)
            ty = table_y
        else:
            tx, tz, ty = scale, 0.0, table_y

        path_points.append((tx, ty, tz))
        joints, angles, can_draw, is_reachable = solve_arm_angles(tx, ty, tz, lengths, bx=0.0, by=0.3, bz=0.0)

        frames.append({
            "joints": joints,
            "angles": angles,
            "target": {"x": tx, "y": ty, "z": tz},
            "can_draw": can_draw,
            "is_reachable": is_reachable
        })

    arc_length = 0.0
    for k in range(1, len(path_points)):
        if frames[k]["can_draw"]:
            arc_length += dist3d(path_points[k - 1], path_points[k])

    theoretical_time = round((360.0 * len(lengths)) / max(req.motor_speed_dps, 1.0), 3)
    step_delay_ms = max(16, int((theoretical_time * 1000) / n_samples))

    return {
        "frames": frames,
        "arc_length": round(arc_length, 3),
        "step_delay_ms": step_delay_ms,
        "theoretical_time": theoretical_time,
        "urdf_xml": build_urdf_xml(lengths)
    }

class LogEntry(BaseModel):
    test_id: str
    shape: str
    link_count: int
    link_lengths: str
    total_reach: float
    motor_speed_dps: float
    arc_length: float
    elapsed_time_sec: float
    avg_linear_speed: float
    avg_angular_speed: float
    status: str

@app.post("/api/arm/log")
def log_experiment(data: LogEntry):
    research_benchmarks.append(data.dict())
    return {"status": "recorded", "count": len(research_benchmarks)}

@app.get("/api/arm/benchmarks")
def get_benchmarks():
    return research_benchmarks

@app.delete("/api/arm/benchmarks")
def clear_benchmarks():
    research_benchmarks.clear()
    return {"status": "cleared"}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
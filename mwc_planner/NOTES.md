# MWC Planner Notes

## Architecture
- **IK**: Mink differential IK solver, solves from physics state every 10th physics step (200Hz)
- **Control**: Custom PID torque loop at full physics rate (2000Hz), motor actuators
- **Magnets**: Dipole model, per-sphere distance calculation against floor/wall box geoms
- **Passive joints**: ee2 (orthogonal to ee, no Z component) + em_z (world Z at EM mount) — no spring, no damping, adds DOFs for foot compliance

## Key Findings

### IK
- Mink IK is perfect in kinematic mode (0mm error)
- Solving from physics state (ground truth) is essential — solving from kinematic copy causes drift that compounds over time
- Body FrameTask (position + orientation, cost=50) prevents body sag
- Stance foot FrameTasks (cost=50) hold feet in place
- Passive joint DOFs frozen via DofFreezingTask so IK doesn't use them
- Freejoint NOT frozen — body task handles it

### Control
- MuJoCo position actuators (PD) have steady-state error under load → error feeds back into IK → compounds
- Fix: motor actuators + custom PID with integral term eliminates steady-state error
- PID gains for 24kg robot: KP=500, KI=200, KD=30, I_CLAMP=100
- Joint damping=2.0, armature=0.01

### Magnets
- Floor box must be thick (100mm) so mj_geomDistance always picks top face, not bottom
- At close range (<1mm) dipole force explodes (billions of N), clamped by max_force_per_wheel=917N
- All 4 magnets saturate at max on floor — 3669N total on a 288N robot
- Magnets help stance by anchoring feet; swing foot magnet disabled

### Robot Model
- 24kg main frame (density=7800), 5.15kg legs total
- 4 actuated joints per leg: hip_pitch (Z), knee (diagonal), wrist (diagonal), ee (diagonal)
- knee/wrist/ee share same world rotation axis at default pose — only 2 effective DOFs per leg
- Added ee2 + em_z passive joints for extra DOFs
- Legs at neutral pose already have feet on floor (z≈0.02)

### Tracking Performance (3s bounce, 5cm amplitude, 24kg, magnets on)
- Swing dz error: <10mm (PID eliminates steady-state)
- Swing dx/dy error: ~18mm (kinematic coupling from diagonal joint axes, irreducible)
- Stance drift: ~29mm plateau, stable
- Body height: ±3mm, no drift over 30s

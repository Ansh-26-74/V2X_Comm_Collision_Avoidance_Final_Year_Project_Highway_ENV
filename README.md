# Step 2 — No-V2V Blind-Spot Collision Baseline

This deterministic scene has exactly three vehicles:

- **A** — green ego/host vehicle, middle lane
- **B** — long orange truck, directly ahead of A in the middle lane
- **C** — red oncoming vehicle, initially hidden behind B in the middle lane

Rather than using fixed times, C moves into lane 0 (its right side) only when it gets within `C_CHANGE_DISTANCE` of B. A overtakes B into lane 0 (A's left side) only when it gets within `A_CHANGE_DISTANCE` of B.

## Step 3: V2V message display

C broadcasts its position, speed, intended lane, direction, and a short planned trajectory through `v2v_manager.py` at 10 Hz. Green A receives the latest message. The top-left display shows `V2V: CONNECTED` and C's direction/lane; the dashed red line is C's received V2V path.

To test C moving left, set `C_TARGET_LANE = 2` in `main.py`. The overlay then automatically displays `C intent: LEFT -> lane 2` and shares/draws C's lane-2 trajectory. Set it back to `0` for `RIGHT -> lane 0`.

## Step 4: V2V lane-conflict avoidance

A begins its overtake from its local view of truck B. C only broadcasts its path when it begins changing lanes. If the incoming V2V path's `intended_lane` matches `A_TARGET_LANE`, the overlay shows `SAFETY: ALERT - RETURNING + BRAKING`; A returns to the original lane and brakes behind the truck. This is deliberately a simple path-conflict rule. TTC-based SAFE/WARNING/CRITICAL logic comes next.

## Run

```powershell
cd "C:\Users\HP\Documents\Codex\2026-08-19\files-pasted-by-the-user-i\outputs\highway_v2x"
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py main.py
```

The window displays only A, B, and C for up to 10 seconds. With the default conflicting lane intent, A brakes and remains behind B instead of colliding with C.

## Change positions and speeds

Edit the constants at the top of `main.py`:

```python
A_LANE, A_POSITION, A_SPEED = 1, 0.0, 25.0
B_LANE, B_POSITION, B_SPEED = 1, 55.0, 10.0
C_LANE, C_POSITION, C_SPEED = 1, 125.0, 25.0
```

Lane `0` is left, lane `1` is middle, and lane `2` is right. A, B, and C initially share lane `1`, ordered A → B ← C. C faces A, so it is oncoming. `C_CHANGE_DISTANCE` and `A_CHANGE_DISTANCE` control when the lane changes begin.

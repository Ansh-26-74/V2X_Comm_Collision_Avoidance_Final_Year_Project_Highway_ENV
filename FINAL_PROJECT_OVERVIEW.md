# V2X Autonomous Safety System: Vehicle-to-Vehicle (V2V) Collision Avoidance & Vehicle-to-Infrastructure (V2I) Emergency Signal Preemption

## 1. Project Title
**V2X Autonomous Safety System: Cooperative V2V Highway Collision Avoidance & V2I Smart Intersection Emergency Preemption**

---

## 2. Problem Statement

This project addresses two critical transportation safety and efficiency challenges using Vehicle-to-Everything (V2X) cooperative communication:

1. **V2V Highway Collision Avoidance (Non-Line-of-Sight Overtaking Hazard)**:
   * On high-speed highways, an overtaking vehicle attempting to pass a large visual obstruction (e.g., an occluding heavy truck) cannot visually detect an oncoming or erratic vehicle in the opposing lane using line-of-sight sensors (cameras, LiDAR) alone.
   * Without cooperative communication, this blind overtaking maneuver leads to severe head-on collisions.
   * *Solution*: Direct low-latency Vehicle-to-Vehicle (V2V) trajectory broadcasting alerts the ego vehicle to compute Time-To-Collision (TTC) and fuse AI trajectory predictions, safely aborting the maneuver before collision occurs.

2. **V2I Emergency Vehicle Priority at a Smart Intersection (Urban Gridlock & Conflict Management)**:
   * In urban intersections, emergency response vehicles (ambulances) encounter red signals and stopped civilian traffic queues, incurring life-threatening transit delays and risking broadside ("T-bone") collisions if crossing unsafely against signals.
   * Blind preemption without traffic state awareness can cause secondary accidents with conflicting cross-traffic trapped in the intersection.
   * *Solution*: Cooperative Vehicle-to-Infrastructure (V2I) communication enables the ambulance to transmit cryptographically structured emergency priority requests to an intersection Road-Side Unit (RSU). The RSU analyzes conflict zones, orchestrates safe all-red clearance, provides a dedicated green corridor, monitors full-body exit, and gracefully restores coordinated signal cycling.

---

## 3. System Architecture

```
                                  V2X SYSTEM ARCHITECTURE
       =============================================================================

       [ V2V HIGHWAY SCENARIO ]
       +-------------------+       V2V Broadcast (DSRC)       +--------------------+
       | Oncoming Vehicle C| -------------------------------> |  Ego Vehicle A     |
       +-------------------+  (Position, Velocity, Traj)      +--------------------+
                                                                        |
                                                                        v
                                                              +--------------------+
                                                              |Trajectory Predictor|
                                                              +--------------------+
                                                                        |
                                                                        v
                                                              +--------------------+
                                                              |Safety Fusion Engine|
                                                              +--------------------+
                                                                        |
                                                                        v
                                                              +--------------------+
                                                              | Evasive Braking &  |
                                                              | Lane Return Action |
                                                              +--------------------+

       [ V2I SMART INTERSECTION SCENARIO ]
       +-------------------+       V2I Request (DSRC)         +--------------------+
       |  Ambulance AMB-01 | -------------------------------> |  Road-Side Unit    |
       +-------------------+  (Approach, Speed, ETA, Route)   |  (RSU-INT-01)      |
                                                              +--------------------+
                                                                        |
                                                                        v
                                                              +--------------------+
                                                              | Traffic & Conflict |
                                                              | Real-time Analyzer |
                                                              +--------------------+
                                                                        |
                                                                        v
                                                              +--------------------+
                                                              | Smart Traffic      |
                                                              | Signal Controller  |
                                                              +--------------------+
                                                                        |
                                                                        v
                                                              +--------------------+
                                                              | 4-Approach Signals |
                                                              | Protected Corridor |
                                                              +--------------------+
```

### A. V2V Subsystem Architecture
$$\text{Vehicle C} \xrightarrow{\text{V2V Communication}} \text{Vehicle A} \xrightarrow{\text{Trajectory Prediction}} \text{Safety Fusion Engine} \xrightarrow{\text{Action: Abort / Brake}}$$
* **Vehicle Kinematics & Road Network**: Built on Gymnasium `HighwayEnv` straight road network with 3 lanes, modeling ego vehicle A, occluding truck B, and oncoming vehicle C.
* **V2V Communication Manager (`V2VManager`)**: Handles dedicated short-range communication with configurable range (300m), packet loss, and latency simulation.
* **AI Trajectory Predictor (`ai/trajectory_predictor.py`)**: Neural/polynomial trajectory prediction forecasting vehicle C’s future path ahead of local sensor visibility.
* **Safety Fusion Engine (`ai/safety_fusion.py`)**: Evaluates Time-To-Collision (TTC) against critical thresholds ($\text{TTC} < 2.0\text{s}$ critical, $\text{TTC} < 4.0\text{s}$ warning), commanding emergency braking and abortion of the overtake until conflict clears.

### B. V2I Subsystem Architecture
$$\text{Ambulance} \xrightarrow{\text{V2I DSRC}} \text{RSU Station} \xrightarrow{\text{Conflict Analysis}} \text{Signal Controller} \xrightarrow{\text{Protected Corridor}}$$
* **Smart Traffic Signal Controller (`v2i/smart_signal.py`)**: Coordinated 4-phase traffic signal controller (`NS_GREEN_EW_RED`, `NS_YELLOW_EW_RED`, `NS_RED_EW_GREEN`, `NS_RED_EW_YELLOW`) with deterministic preemption safety gates.
* **Civilian Traffic Manager (`v2i/traffic_manager.py`)**: Microscopic kinematic simulation of autonomous civilian traffic and emergency vehicle `AMB-01` across 4 approaches (North, South, East, West).
* **RSU Traffic & Conflict Analyzer (`v2i/traffic_analyzer.py`)**: Real-time geometric analysis of the central $120\text{px} \times 120\text{px}$ conflict zone and critical approach sectors ($480\text{px} \le x, y \le 640\text{px}$).
* **V2I Communication Channel (`v2i/v2i_manager.py`)**: Simulated DSRC channel with 400px broadcast radius, 200ms message latency, and packet loss modeling.
* **Event Logger & Observability Engine (`v2i/event_logger.py`, `v2i/v2i_renderer.py`)**: Deterministic timestamped audit log and glassmorphic real-time telemetry HUD.

---

## 4. V2I Emergency Lifecycle Flow

The complete emergency signal preemption and recovery lifecycle follows a 13-stage deterministic state progression:

```
[NORMAL COUNCYCLE] ---> [EMERGENCY REQUEST] ---> [VALIDATION & CONFLICT CHECK]
                                                            |
                                                            v
[SOUTH GREEN CORRIDOR] <--- [ALL-RED CLEARANCE] <--- [YELLOW CLEARANCE]
          |
          v
[AMBULANCE CROSSING] ---> [FULL-BODY CLEARANCE] ---> [TERMINATING YELLOW]
                                                            |
                                                            v
[NORMAL CYCLE RESUMED] <--- [RECOVERY ALL-RED] <-----------+
```

1. **Stage 1 (Normal Cycle)**: Signal executes coordinated 4-phase cycle (NS 8s green, 3s yellow; EW 8s green, 3s yellow).
2. **Stage 2 (Civilian Obedience)**: Civilian vehicles obey stop lines when facing yellow/red signals and maintain safe deceleration.
3. **Stage 3 (Ambulance Approach)**: AMB-01 approaches South stop line while South signal is RED.
4. **Stage 4 (Safe Stopping)**: AMB-01 decelerates and comes to a safe stop at the South stop line ($y \ge 535.0\text{px}$).
5. **Stage 5 (DSRC Entry)**: AMB-01 enters the 400px communication radius of the RSU.
6. **Stage 6 (Request Delivery)**: AMB-01 transmits `EMERGENCY_REQUEST` (`approach="SOUTH"`, `priority="HIGH"`), reaching RSU after 200ms latency.
7. **Stage 7 (Validation & Conflict Analysis)**: RSU validates request freshness ($\Delta t \le 3.0\text{s}$) and analyzes central intersection conflict zone.
8. **Stage 8 (Safe Clearance)**: Conflicting EW phase transitions to `PREEMPTION_YELLOW` (3.0s) then `ALL_RED_HOLD` (1.5s hold + dynamic conflict holding if vehicles occupy the junction).
9. **Stage 9 (Emergency South Green)**: RSU safety gate confirms intersection is clear; South signal turns `GREEN` while North, East, and West are locked `RED`.
10. **Stage 10 (Protected Corridor Crossing)**: AMB-01 accelerates through South stop line into protected intersection corridor.
11. **Stage 11 (Full-Body Clearance)**: Sensor/RSU verifies authoritative full-body intersection exit (`rear_bumper_y <= 350.0px`).
12. **Stage 12 (Controlled Termination)**: Signal transitions South to `EMERGENCY_TERMINATING` (yellow, 2.0s) followed by `RECOVERY_ALL_RED` (1.5s).
13. **Stage 13 (Normal Cycle Resumption)**: Signal safely resumes normal coordinated cycling from the interrupted/next phase (`NS_RED_EW_GREEN`).

---

## 5. Safety & Failure Containment Mechanisms

The system incorporates defensive fault-tolerant mechanisms validated under Phase 11 testing:

* **Stale Request Rejection**: Requests with timestamps exceeding freshness threshold ($\Delta t > 3.0\text{s}$) are dropped without preemption.
* **Duplicate Request Idempotency**: Repeated requests during any lifecycle state do not re-trigger preemption or reset phase timers.
* **Packet Loss Containment**: Dropped packets fail safely; signals maintain standard cycles and the ambulance safely waits at red.
* **Communication Delay Handling**: Delayed packets are evaluated against current intersection state (`REQUEST != IMMEDIATE GREEN`).
* **Conflict-Zone Clearance Holding**: If an East/West vehicle is inside or unsafely approaching the junction, All-Red clearance is dynamically held until the zone is vacated.
* **Mutual-Green Prevention Invariant**: Conflicting signal phases (NS and EW) can never be simultaneously GREEN under any circumstance.
* **Full-Body Clearance Rule**: Clearance is verified strictly when vehicle rear bumper clears the junction exit line, preventing premature termination.
* **Idempotent Emergency Termination**: Duplicate clearance signals cannot trigger multiple recovery cycles or corrupt state transitions.
* **Bounded Lifetime Guarantee**: Preemption states cannot lock into permanent emergency green or permanent all-red deadlocks.

---

## 6. Measured Quantitative Results (Phase 10 Benchmark)

In controlled, deterministic simulation experiments comparing **`BASELINE_NO_V2I`** against **`V2I_ENABLED`** across 10 paired trials with identical kinematics, geometry, traffic seeds, and initial cycle timings:

```
+-----------------------------------------------------------------------------------+
| METRIC                       BASELINE (NO V2I)     V2I ENABLED       CHANGE / BENEFIT    |
+-----------------------------------------------------------------------------------+
| Mean Ambulance Travel Time        11.35 s             8.65 s        -2.70 s (-23.8%)    |
| Median Ambulance Travel Time      12.22 s             9.64 s        -2.58 s (-21.1%)    |
| Mean Signal Waiting Time           4.97 s             2.27 s        -2.70 s (-54.3%)    |
| Mean Intersection Traversal        1.73 s             1.73 s         0.00 s (  0.0%)    |
| Mean Full Vehicle Stops            0.80 stops          0.80 stops    0.00 stops ( 0.0%) |
| Mean V2I Preemption Delay            N/A              0.21 s        (Request -> Green)  |
+-----------------------------------------------------------------------------------+
```

> **Operational Insight**:
> In the controlled simulation, V2I reduced mean ambulance travel time by **23.8%** across 10 paired trials.
> The primary performance difference originated entirely from **reduced signal waiting time** (54.3% reduction, saving 2.70s) rather than faster physical intersection traversal (1.73s in both cases), demonstrating that V2I delivers efficiency through coordination rather than unsafe speeding.

---

## 7. Verification & Test Suite Summary

The project is backed by a multi-tier automated test suite:

* **Phases 1–10 Functional & Benchmark Tests**: 104 tests
  * Phase 2: Civilian traffic behavior and stop line obedience (12 tests)
  * Phase 3: Ambulance kinematics and braking (10 tests)
  * Phase 4: V2I DSRC message encoding, range, latency (10 tests)
  * Phase 5: RSU conflict zone geometric evaluation (12 tests)
  * Phase 6: Preemption state transitions and mutex enforcement (16 tests)
  * Phase 7: Ambulance crossing and full-body clearance (18 tests)
  * Phase 8: Termination sequence and normal recovery (14 tests)
  * Phase 9: Event logger and observability pipeline (6 tests)
  * Phase 10: Benchmark metrics and paired experiment runner (6 tests)
* **Phase 11 Failure & Safety Validation Tests**: 12 tests
  * Communication failure containment (packet loss, high latency)
  * Request validation (stale rejection, duplicate idempotency)
  * Geometric conflict holding during preemption
  * Authoritative clearance & stuck-state lifetime checks
  * Continuous per-tick signal invariant checks (2,500 ticks)
  * Multi-cycle long-duration stability (3,600 frames)
  * Seeded randomized failure combinations (Seeds 301–305)
  * V2V architectural isolation check
* **Total Project Tests**: **116 / 116 tests passing (100% passing, 0 failures, 0 errors in ~8s)**

---

## 8. Academic Limitations & Scope Boundary

To maintain rigorous academic honesty, the following constraints of this implementation are noted:

1. **Simulated Communication**: V2I and V2V communication channels model range limits, latency, and packet loss using software parameters; no physical RF propagation models (e.g., Nakagami/Rayleigh fading, Doppler spread) are implemented.
2. **Simulated Traffic & Environment**: Vehicle dynamics are 2D kinematic point-mass approximations implemented via Python and Pygame/Gymnasium.
3. **No Real Hardware Integration**: The project does not interface with physical DSRC/C-V2X On-Board Units (OBUs), physical Road-Side Units (RSUs), or NEMA/170/2070 traffic signal controllers.
4. **Environment-Specific Benchmark**: The benchmark metrics reflect the synthetic simulation environment and controlled traffic density profiles defined in Phase 10.
5. **Not a Certified Safety System**: This simulation serves as an academic proof-of-concept demonstration and does not constitute real-world automotive safety certification (e.g., ISO 26262, MISRA, or UL 4600).

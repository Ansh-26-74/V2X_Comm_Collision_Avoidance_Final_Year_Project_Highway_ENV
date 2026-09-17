"""Run the exact v2i_main.py simulation loop for 15 seconds with seed 42,
printing AI HUD state, C-01 trajectory waypoints, and conflict detection.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import random
import pygame

import v2i_main
from v2i.smart_signal import SmartTrafficSignal, CyclePhase
from v2i.traffic_manager import TrafficManager
from v2i.v2i_manager import V2IManager
from v2i.v2v_inter_manager import IntersectionV2VManager
from v2i.event_logger import EventLogger, EventCategory
import v2i.v2i_renderer as renderer

def test_live_simulation():
    # Let's run the actual main with a time limit or monkey-patched loop
    # Or let's inspect the actual main() function by setting a timer to press ESC after 15 seconds!
    print("Testing v2i_main with auto-quit after 14 seconds...")
    import threading
    
    def post_quit():
        time.sleep(14.0)
        print("[TEST] 14 seconds elapsed, posting pygame.QUIT event...")
        event = pygame.event.Event(pygame.QUIT)
        pygame.event.post(event)
        
    t = threading.Thread(target=post_quit, daemon=True)
    t.start()
    
    v2i_main.main()
    print("[TEST] v2i_main exited cleanly!")

if __name__ == "__main__":
    test_live_simulation()

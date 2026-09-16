from collections import namedtuple

# A minimal mock V2V message to feed the TTC engine when AI detects a conflict but V2V hasn't yet.
AIMockMessage = namedtuple('AIMockMessage', ['x', 'y', 'speed'])

class SafetyFusionEngine:
    def __init__(self):
        pass
        
    def evaluate(self, v2v_conflict: bool, ai_conflict: bool, v2v_message, c_x: float, c_y: float, c_speed: float):
        """
        Fuses V2V and AI conflict booleans into a single decision.
        Returns (fused_conflict, threat_object, source_name).
        """
        fused_conflict = v2v_conflict or ai_conflict
        
        if not fused_conflict:
            return False, None, "NONE"
            
        if v2v_conflict and ai_conflict:
            source = "BOTH"
            threat = v2v_message # Prefer the V2V message as it has the actual remote-broadcasted speed
        elif v2v_conflict:
            source = "V2V"
            threat = v2v_message
        else:
            source = "AI"
            # Construct a lightweight mock message using C's current physical state for the TTC calculation
            threat = AIMockMessage(x=c_x, y=c_y, speed=c_speed)
            
        return True, threat, source

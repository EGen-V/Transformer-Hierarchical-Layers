from typing import List
from thl.config import THLConfig

class TierClock:
    """
    Manages multi-timescale updates for tiers.
    c_k(t) = 1 if t % tau_k == 0 else 0
    """
    def __init__(self, config: THLConfig):
        self.timescales = config.tier_timescales
        self.num_tiers = config.num_tiers
        
    def get_active_tiers(self, timestep: int) -> List[bool]:
        """
        Returns a boolean list indicating which tiers should update at this timestep.
        """
        active = []
        for tau in self.timescales:
            # Shifted timestep? Usually t start at 0 or 1.
            # Spec says "c_k(t) = 1 if t mod tau_k == 0". 
            # If t=0, 0 mod anything is 0 -> All tiers active at start?
            # Yes, usually initialize detailed state at step 0.
            active.append((timestep % tau) == 0)
        return active

import torch

class GeometricCommitAgent:
    def __init__(self, trace_threshold=0.5):
        """
        A deterministic agent that commits if the trace signal exceeds a threshold.
        No learning, no memory state (other than the latching which happens in the env).
        """
        self.trace_threshold = trace_threshold

    def act(self, observation):
        """
        observation: [u_mean, trace_fast_mean, brace, health]
        """
        # observation[1] is trace_fast_mean (the Cue)
        trace_val = observation[1]

        # Simple Hysteresis / Threshold Logic
        if trace_val > self.trace_threshold:
            return True # Commit
        return False

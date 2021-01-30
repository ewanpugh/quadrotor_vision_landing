import time

class PID:
    
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_time = None
        self.last_error = None
        return self
    
    def set_setpoint(self, setpoint):
        self.setpoint = setpoint
        
    def output_value(self, feedback):
        if (self.last_time is not None) & (self.last_error is not None):
    else:
        self.last_time = time.time()
        
import time


class PID:

    
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = None
        self.last_time = None
        self.last_error = None
        self.i_term_sum = 0

    
    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

        
    def generate_output(self, feedback):
        
        if self.setpoint is None:
            raise ValueError('No setpoint value set')
        
        error = self.setpoint - feedback
        
        p_term = error * self.Kp
        d_term = 0
        if self.last_time is not None:
            time_delta = time.time() - self.last_time
            self.i_term_sum += error * time_delta

            if self.last_error is not None:
                error_delta = error - self.last_error
                d_term = error_delta / time_delta
                
        # Integral windup protection, if error = 0 or error crosses zero, 
        # reset integral sum
        if self.last_error is not None:
            if (error == 0) | ((error > 0) & (self.last_error < 0)) | \
               ((error < 0) & (self.last_error > 0)):
                self.i_term_sum = 0
        
        output_value = (p_term +
                       (self.Ki * self.i_term_sum) +
                       (self.Kd * d_term))
        
        self.last_time = time.time()
        self.last_error = error
        
        return output_value
from datetime import datetime

def get_current_time_string():
    # Get the current time
    now = datetime.now()
    
    # Format the time components into a string
    time_string = now.strftime("%B %d, %Y, %H:%M:%S")
    
    return time_string
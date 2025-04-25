import socket


def is_center_inside_rectangle(point, rectangle):
    " Check whether a point is inside a rectangle "
    px, py = point
    x1, y1, x2, y2 = rectangle
    return x1 <= px <= x2 and y1 <= py <= y2

def is_internet_available(timeout=3):
    """Check if localhost on port 5000 is reachable."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("127.0.0.1", 5000))
        return True
    except socket.error as ex:
        print(f"Localhost check failed: {ex}")
        return False

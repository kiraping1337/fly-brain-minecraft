"""Углы и направления в координатах Minecraft."""

import math


#геометрия Minecraft

def heading_vector(yaw_deg):
    radians = math.radians(yaw_deg)
    return -math.sin(radians), math.cos(radians)


def bearing_deg(dx, dz):
    return math.degrees(math.atan2(-dx, dz))


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0

import math
def calculate_damaged_area_coordinates(latitude, longitude, fire_radius_km):
    # Earth's radius in kilometers (mean radius)
    earth_radius_km = 6371.0

    # Convert fire radius from kilometers to degrees (approximation)
    fire_radius_deg = fire_radius_km / earth_radius_km * (180.0 / math.pi)

    # Generate coordinates for the circular damaged area
    num_points = 360  # Number of points to generate for the circle
    step = 360.0 / num_points  # Angle step between points

    # Initialize lists to store latitude and longitude coordinates
    damaged_area_latitudes = []
    damaged_area_longitudes = []

    # Generate coordinates for the circular area
    for angle in range(0, 360, int(step)):
        angle_rad = math.radians(angle)
        lat = latitude + (fire_radius_deg * math.sin(angle_rad))
        lon = longitude + (fire_radius_deg * math.cos(angle_rad))
        damaged_area_latitudes.append(lat)
        damaged_area_longitudes.append(lon)

    return damaged_area_latitudes, damaged_area_longitudes

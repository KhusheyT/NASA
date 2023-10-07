from geopy.geocoders import Nominatim

def get_country_name(latitude, longitude):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True)
    
    if location and 'address' in location.raw:
        country = location.raw['address'].get('country')
        if country:
            return country
    
    return "Country not found"

# Example usage:
latitude = x 
longitude = y

country_name = get_country_name(latitude, longitude)
print(f"The coordinates are in {country_name}.")

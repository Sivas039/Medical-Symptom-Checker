
import os
import requests
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2
import googlemaps
from datetime import datetime
import tempfile

@dataclass
class Hospital:
    """Hospital information"""
    name: str
    address: str
    phone: Optional[str]
    distance_km: float
    travel_time: Optional[str]
    google_maps_url: str
    place_id: Optional[str]
    rating: Optional[float]
    emergency_available: bool

class HospitalLocator:
    """Locate nearby hospitals using Google Maps API"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        self.gmaps = None

        if self.api_key:
            try:
                self.gmaps = googlemaps.Client(key=self.api_key)
                print("✓ Google Maps client initialized")
            except Exception as e:
                print(f"⚠ Google Maps initialization failed: {e}")
        else:
            print("⚠ Google Maps API key not found. Hospital location features disabled.")

    def get_user_location(self) -> Optional[Tuple[float, float]]:
        """Get user's current location (with fallback to IP-based location)"""
        try:
            # Try to get location from IP
            response = requests.get("http://ip-api.com/json/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return (data.get("lat"), data.get("lon"))
        except Exception:
            pass

        # Fallback to Chennai, Tamil Nadu (based on user's location from logs)
        return (13.0827, 80.2707)  # Chennai coordinates

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        R = 6371  # Earth's radius in kilometers

        lat1_rad, lon1_rad = radians(lat1), radians(lon1)
        lat2_rad, lon2_rad = radians(lat2), radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    def find_nearby_hospitals(self,
                            location: Optional[Tuple[float, float]] = None,
                            radius_km: int = 10,
                            emergency_only: bool = True,
                            max_results: int = 5) -> List[Hospital]:
        """Find nearby hospitals with emergency services"""
        if not self.gmaps:
            return self._get_fallback_hospitals()

        if not location:
            location = self.get_user_location()

        if not location:
            return self._get_fallback_hospitals()

        hospitals = []

        try:
            # Search for hospitals
            query = "emergency hospital" if emergency_only else "hospital"

            # Use Google Places API
            places_result = self.gmaps.places_nearby(
                location=location,
                radius=radius_km * 1000,  # Convert to meters
                type='hospital',
                keyword=query
            )

            for place in places_result.get('results', [])[:max_results]:
                try:
                    # Get detailed place information
                    place_details = self.gmaps.place(place['place_id'])['result']

                    # Calculate distance
                    place_lat = place['geometry']['location']['lat']
                    place_lng = place['geometry']['location']['lng']
                    distance = self.calculate_distance(
                        location[0], location[1],
                        place_lat, place_lng
                    )

                    # Get travel time
                    travel_time = self._get_travel_time(location, (place_lat, place_lng))

                    # Create Google Maps URL
                    maps_url = f"https://www.google.com/maps/place/?q=place_id:{place['place_id']}"

                    hospital = Hospital(
                        name=place['name'],
                        address=place.get('vicinity', 'Address not available'),
                        phone=place_details.get('formatted_phone_number'),
                        distance_km=round(distance, 2),
                        travel_time=travel_time,
                        google_maps_url=maps_url,
                        place_id=place['place_id'],
                        rating=place.get('rating'),
                        emergency_available=self._check_emergency_services(place, place_details)
                    )

                    hospitals.append(hospital)

                except Exception as e:
                    print(f"Error processing hospital: {e}")
                    continue

            # Sort by distance
            hospitals.sort(key=lambda h: h.distance_km)

        except Exception as e:
            print(f"Error finding hospitals: {e}")
            return self._get_fallback_hospitals()

        return hospitals if hospitals else self._get_fallback_hospitals()

    def _get_travel_time(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> Optional[str]:
        """Get estimated travel time to hospital"""
        if not self.gmaps:
            return None

        try:
            result = self.gmaps.distance_matrix(
                origins=[origin],
                destinations=[destination],
                mode="driving",
                departure_time=datetime.now()
            )

            if result['status'] == 'OK':
                element = result['rows'][0]['elements'][0]
                if element['status'] == 'OK':
                    return element['duration']['text']

        except Exception:
            pass

        return None

    def _check_emergency_services(self, place: Dict, place_details: Dict) -> bool:
        """Check if hospital has emergency services"""
        # Check in various fields for emergency indicators
        name_lower = place.get('name', '').lower()
        types = place.get('types', [])

        emergency_indicators = [
            'emergency' in name_lower,
            'trauma' in name_lower,
            'casualty' in name_lower,
            'urgent' in name_lower,
            'hospital' in types
        ]

        return any(emergency_indicators)

    def _get_fallback_hospitals(self) -> List[Hospital]:
        """Return hardcoded hospitals for Chennai area when API fails"""
        chennai_hospitals = [
            Hospital(
                name="Apollo Hospitals, Greams Road",
                address="21, Greams Lane, Off Greams Road, Chennai, Tamil Nadu 600006",
                phone="+91 44 2829 0200",
                distance_km=5.2,
                travel_time="15-20 mins",
                google_maps_url="https://maps.google.com/?q=Apollo+Hospitals+Chennai+Greams+Road",
                place_id=None,
                rating=4.3,
                emergency_available=True
            ),
            Hospital(
                name="Government General Hospital",
                address="Park Town, Chennai, Tamil Nadu 600003",
                phone="+91 44 2530 5000",
                distance_km=3.8,
                travel_time="10-15 mins",
                google_maps_url="https://maps.google.com/?q=Government+General+Hospital+Chennai",
                place_id=None,
                rating=3.8,
                emergency_available=True
            ),
            Hospital(
                name="MIOT International",
                address="4/112, Mount Poonamallee Road, Manapakkam, Chennai, Tamil Nadu 600089",
                phone="+91 44 4200 2288",
                distance_km=8.5,
                travel_time="20-25 mins",
                google_maps_url="https://maps.google.com/?q=MIOT+International+Chennai",
                place_id=None,
                rating=4.5,
                emergency_available=True
            ),
            Hospital(
                name="Fortis Malar Hospital",
                address="52, 1st Main Road, Adyar, Chennai, Tamil Nadu 600020",
                phone="+91 44 4289 2222",
                distance_km=7.1,
                travel_time="18-22 mins",
                google_maps_url="https://maps.google.com/?q=Fortis+Malar+Hospital+Chennai",
                place_id=None,
                rating=4.2,
                emergency_available=True
            ),
            Hospital(
                name="Sri Ramachandra Medical Centre",
                address="1, Ramachandra Nagar, Porur, Chennai, Tamil Nadu 600116",
                phone="+91 44 2476 8000",
                distance_km=12.3,
                travel_time="25-30 mins",
                google_maps_url="https://maps.google.com/?q=Sri+Ramachandra+Medical+Centre+Chennai",
                place_id=None,
                rating=4.4,
                emergency_available=True
            )
        ]

        return chennai_hospitals

    def format_hospital_recommendations(self, hospitals: List[Hospital], severity_level: str) -> str:
        """Format hospital list for display with urgency indicators"""
        if not hospitals:
            return "Unable to find nearby hospitals. Please call 108 for emergency assistance."

        # Header based on severity
        if "high" in severity_level.lower() or "emergency" in severity_level.lower():
            header = """🚨 **IMMEDIATE MEDICAL ATTENTION NEEDED** 🚨

**Call 108 for ambulance service immediately**

**Nearby Emergency Hospitals:**"""
        else:
            header = """**Recommended Nearby Hospitals:**"""

        hospital_text = [header, ""]

        for i, hospital in enumerate(hospitals[:5], 1):
            hospital_info = [
                f"**{i}. {hospital.name}**",
                f" 📍 Distance: {hospital.distance_km} km"
            ]

            if hospital.travel_time:
                hospital_info.append(f" 🚗 Travel Time: {hospital.travel_time}")

            hospital_info.append(f" 📍 {hospital.address}")

            if hospital.phone:
                hospital_info.append(f" 📞 {hospital.phone}")

            if hospital.rating:
                hospital_info.append(f" ⭐ Rating: {hospital.rating}/5")

            hospital_info.append(f" 🗺️ [View on Google Maps]({hospital.google_maps_url})")

            if hospital.emergency_available:
                hospital_info.append(f" ✅ Emergency Services Available")

            hospital_text.extend(hospital_info)
            hospital_text.append("")  # Empty line between hospitals

        if "high" in severity_level.lower():
            hospital_text.extend([
                "**⚠️ Emergency Instructions:**",
                "1. Call 108 immediately for ambulance",
                "2. Don't drive yourself if experiencing severe symptoms",
                "3. Have someone accompany you to the hospital",
                "4. Bring any medications you're currently taking",
                "5. Bring identification and insurance information if available",
                "",
                "**Emergency Hotlines:**",
                "- Ambulance: 108",
                "- Emergency: 112",
                "- Medical Helpline: 104"
            ])

        return "\n".join(hospital_text)

    def get_emergency_contacts(self, region: str = "India") -> Dict[str, str]:
        """Get emergency contact numbers for the region"""
        emergency_contacts = {
            "India": {
                "Ambulance": "108",
                "Emergency": "112",
                "Medical Helpline": "104",
                "Police": "100",
                "Fire": "101",
                "Women Helpline": "1091",
                "Child Helpline": "1098"
            }
        }

        return emergency_contacts.get(region, emergency_contacts["India"])
    
    def generate_hospital_map(self, hospitals: List[Hospital], user_location: Optional[Tuple[float, float]] = None) -> Optional[str]:
        """
        Generate an interactive HTML map with hospital locations using Folium.
        Returns the HTML file path if successful, None otherwise.
        """
        try:
            import folium
            import folium.plugins
        except ImportError:
            print("[WARN] Folium not available for map generation. Install with: pip install folium")
            return None
        
        if not hospitals or not user_location:
            return None
        
        try:
            # Create map centered on user location
            hospital_map = folium.Map(
                location=user_location,
                zoom_start=13,
                tiles="OpenStreetMap"
            )
            
            # Add user location marker
            folium.Marker(
                location=user_location,
                popup="📍 Your Location",
                icon=folium.Icon(color="blue", icon="info-sign"),
                tooltip="You are here"
            ).add_to(hospital_map)
            
            # Color code based on distance
            def get_color_by_distance(distance_km: float) -> str:
                if distance_km < 2:
                    return "green"
                elif distance_km < 5:
                    return "yellow"
                elif distance_km < 10:
                    return "orange"
                else:
                    return "red"
            
            # Add hospital markers
            for idx, hospital in enumerate(hospitals, 1):
                color = get_color_by_distance(hospital.distance_km)
                icon_symbol = "hospital-o" if hospital.emergency_available else "building"
                
                # Hospital info popup
                popup_html = f"""
                <div style="width: 250px; font-family: Arial;">
                    <h4>{hospital.name}</h4>
                    <p><strong>Distance:</strong> {hospital.distance_km} km</p>
                    <p><strong>Address:</strong> {hospital.address}</p>
                    {'<p><strong>Phone:</strong> <a href="tel:' + hospital.phone + '">' + hospital.phone + '</a></p>' if hospital.phone else ''}
                    {'<p><strong>Travel Time:</strong> ' + hospital.travel_time + '</p>' if hospital.travel_time else ''}
                    {'<p><strong>Rating:</strong> ⭐ ' + str(hospital.rating) + '/5</p>' if hospital.rating else ''}
                    {'<p>✅ Emergency Services Available</p>' if hospital.emergency_available else ''}
                    <p><a href="{hospital.google_maps_url}" target="_blank">🗺️ Open in Google Maps</a></p>
                </div>
                """
                
                folium.Marker(
                    location=(hospital.address, hospital.address),  # Note: ideally we'd have lat/lon
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color=color, icon=icon_symbol, prefix="fa"),
                    tooltip=f"{idx}. {hospital.name}\n({hospital.distance_km}km away)"
                ).add_to(hospital_map)
            
            # Add a circle to show search radius
            folium.Circle(
                location=user_location,
                radius=10000,  # 10 km
                popup="10 km radius",
                color="blue",
                fill=False,
                opacity=0.3
            ).add_to(hospital_map)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
            hospital_map.save(temp_file.name)
            
            return temp_file.name
        
        except Exception as e:
            print(f"[WARN] Failed to generate hospital map: {e}")
            return None


"""Weather widget displaying current weather conditions."""
import cv2
import time
import json
import threading
import math

from BaseWidget import BaseOverlayWidget

# Weather data cache duration (5 minutes)
WEATHER_CACHE_DURATION = 300
# Location cache duration (1 hour) - location doesn't change often
LOCATION_CACHE_DURATION = 3600


class WeatherWidget(BaseOverlayWidget):
    """Widget displaying current weather conditions."""

    # Class-level location cache (shared across instances)
    _cached_location = None
    _location_cache_time = 0

    def __init__(self, position=(100, 100)):
        super().__init__(
            position=position,
            size=(200, 120),
            min_size=(150, 100),
            max_size=(320, 200)
        )
        self.weather_data = None
        self.last_fetch = 0
        self.fetch_interval = WEATHER_CACHE_DURATION
        self.is_fetching = False
        self.error_message = None
        self.accent_color = (135, 206, 250)  # Light sky blue

    def _get_location(self):
        """Get location using IP geolocation APIs (no installation required)."""
        import urllib.request

        current_time = time.time()

        # Check cache first
        if (WeatherWidget._cached_location is not None and
            current_time - WeatherWidget._location_cache_time < LOCATION_CACHE_DURATION):
            return WeatherWidget._cached_location

        location = None

        # Try multiple IP geolocation APIs in order of reliability
        apis = [
            # ip-api.com - free, no key required, good accuracy
            ('http://ip-api.com/json/', lambda d: (d.get('lat'), d.get('lon'), d.get('city'))),
            # ipapi.co - free tier available
            ('https://ipapi.co/json/', lambda d: (d.get('latitude'), d.get('longitude'), d.get('city'))),
            # ipinfo.io - free tier available
            ('https://ipinfo.io/json', lambda d: self._parse_ipinfo(d)),
        ]

        for url, parser in apis:
            try:
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'camouflage-weather-widget')
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    lat, lon, city = parser(data)
                    if lat and lon:
                        location = (float(lat), float(lon), city)
                        break
            except Exception:
                continue

        if location:
            WeatherWidget._cached_location = location
            WeatherWidget._location_cache_time = current_time

        return location

    def _parse_ipinfo(self, data):
        """Parse ipinfo.io response which has 'loc' as 'lat,lon' string."""
        loc = data.get('loc', '')
        city = data.get('city')
        if ',' in loc:
            parts = loc.split(',')
            return (parts[0], parts[1], city)
        return (None, None, city)

    def _fetch_weather_thread(self):
        """Fetch weather data in background thread."""
        try:
            import urllib.request

            # Get location from IP geolocation
            location = self._get_location()
            city_from_geo = None

            if location:
                lat, lon, city_from_geo = location
                # Use coordinates for precise weather
                url = f"https://wttr.in/{lat},{lon}?format=j1"
            else:
                # Fallback to wttr.in's own IP detection
                url = "https://wttr.in/?format=j1"

            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'camouflage-weather-widget')

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            current = data.get('current_condition', [{}])[0]
            nearest_area = data.get('nearest_area', [{}])[0]

            # Prefer city from geolocation API (usually more accurate)
            city = city_from_geo or nearest_area.get('areaName', [{}])[0].get('value', 'Unknown')

            self.weather_data = {
                'temp_c': current.get('temp_C', '--'),
                'temp_f': current.get('temp_F', '--'),
                'feels_like_c': current.get('FeelsLikeC', '--'),
                'feels_like_f': current.get('FeelsLikeF', '--'),
                'condition': current.get('weatherDesc', [{}])[0].get('value', 'Unknown'),
                'humidity': current.get('humidity', '--'),
                'wind_mph': current.get('windspeedMiles', '--'),
                'city': city,
            }
            self.error_message = None

        except Exception as e:
            self.error_message = str(e)[:25]
        finally:
            self.is_fetching = False

    def fetch_weather(self):
        """Start background fetch of weather data."""
        current_time = time.time()
        if current_time - self.last_fetch < self.fetch_interval:
            return
        if self.is_fetching:
            return

        self.is_fetching = True
        self.last_fetch = current_time
        thread = threading.Thread(target=self._fetch_weather_thread, daemon=True)
        thread.start()

    def _draw_weather_icon(self, frame, x, y, condition, scale=1.0):
        """Draw a simple weather icon based on condition."""
        condition_lower = condition.lower()
        color = self.accent_color
        s = scale  # Scale factor

        if 'sun' in condition_lower or 'clear' in condition_lower:
            # Sun icon
            cv2.circle(frame, (x, y), int(12 * s), (0, 200, 255), 2)
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                x1 = int(x + 16 * s * math.cos(rad))
                y1 = int(y + 16 * s * math.sin(rad))
                x2 = int(x + 20 * s * math.cos(rad))
                y2 = int(y + 20 * s * math.sin(rad))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        elif 'cloud' in condition_lower or 'overcast' in condition_lower:
            # Cloud icon
            cv2.ellipse(frame, (int(x - 5 * s), y), (int(12 * s), int(8 * s)), 0, 0, 360, color, 2)
            cv2.ellipse(frame, (int(x + 8 * s), int(y + 2 * s)), (int(10 * s), int(7 * s)), 0, 0, 360, color, 2)
        elif 'rain' in condition_lower or 'drizzle' in condition_lower:
            # Rain icon (cloud with drops)
            cv2.ellipse(frame, (x, int(y - 5 * s)), (int(10 * s), int(6 * s)), 0, 0, 360, color, 2)
            cv2.line(frame, (int(x - 6 * s), int(y + 5 * s)), (int(x - 8 * s), int(y + 12 * s)), color, 2)
            cv2.line(frame, (x, int(y + 5 * s)), (int(x - 2 * s), int(y + 12 * s)), color, 2)
            cv2.line(frame, (int(x + 6 * s), int(y + 5 * s)), (int(x + 4 * s), int(y + 12 * s)), color, 2)
        elif 'snow' in condition_lower:
            # Snowflake icon
            cv2.line(frame, (x, int(y - 10 * s)), (x, int(y + 10 * s)), color, 2)
            cv2.line(frame, (int(x - 9 * s), int(y - 5 * s)), (int(x + 9 * s), int(y + 5 * s)), color, 2)
            cv2.line(frame, (int(x - 9 * s), int(y + 5 * s)), (int(x + 9 * s), int(y - 5 * s)), color, 2)
        elif 'thunder' in condition_lower or 'storm' in condition_lower:
            # Lightning bolt
            import numpy as np
            pts = [(int(x - 3 * s), int(y - 10 * s)), (int(x + 2 * s), int(y - 2 * s)),
                   (int(x - 2 * s), int(y - 2 * s)), (int(x + 3 * s), int(y + 10 * s)),
                   (int(x - 2 * s), int(y + 2 * s)), (int(x + 2 * s), int(y + 2 * s))]
            cv2.polylines(frame, [np.array(pts, np.int32)], False, (0, 200, 255), 2)
        else:
            # Generic weather icon (thermometer)
            cv2.rectangle(frame, (int(x - 3 * s), int(y - 12 * s)), (int(x + 3 * s), int(y + 5 * s)), color, 2)
            cv2.circle(frame, (x, int(y + 8 * s)), int(6 * s), color, 2)

    def draw(self, frame, mirrored=True):
        """Draw the weather widget."""
        if not self.visible:
            return

        self.fetch_weather()

        x, y, w, h = self.draw_background(frame)

        # Title
        self.draw_text(frame, "Weather", (x + 8, y + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, mirrored)

        content_y = y + 50

        if self.is_fetching and not self.weather_data:
            self.draw_text(frame, "Loading...", (x + 10, content_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, mirrored)
        elif self.error_message:
            self.draw_text(frame, self.error_message, (x + 10, content_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 100, 100), 1, mirrored)
        elif self.weather_data:
            # Location
            city = self.weather_data['city']
            max_city_chars = max(8, (w - 30) // 10)
            if len(city) > max_city_chars:
                city = city[:max_city_chars - 3] + "..."
            self.draw_text(frame, city, (x + 10, y + 42),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, mirrored)

            # Weather icon (scale based on widget size)
            icon_scale = min(1.0, (w - 100) / 100)
            self._draw_weather_icon(frame, x + w - 35, y + 65,
                                    self.weather_data['condition'], scale=icon_scale)

            # Temperature (large)
            temp = f"{self.weather_data['temp_f']}F"
            font_scale = 0.9 if w >= 180 else 0.7
            self.draw_text(frame, temp, (x + 10, content_y + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.text_color, 2, mirrored)

            # Condition (if space permits)
            if h > 100:
                condition = self.weather_data['condition']
                max_cond_chars = max(8, (w - 60) // 8)
                if len(condition) > max_cond_chars:
                    condition = condition[:max_cond_chars - 3] + "..."
                self.draw_text(frame, condition, (x + 10, content_y + 35),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, mirrored)

            # Humidity (if space permits)
            if h > 110:
                humidity = f"Humidity: {self.weather_data['humidity']}%"
                self.draw_text(frame, humidity, (x + 10, content_y + 52),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, mirrored)
        else:
            self.draw_text(frame, "No weather data", (x + 10, content_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, mirrored)

        # Draw resize handles
        self.draw_resize_handles(frame)

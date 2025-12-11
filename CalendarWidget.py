"""Calendar widget displaying upcoming events from Google Calendar."""
import cv2
import time
import json
import subprocess
import threading

from BaseWidget import BaseOverlayWidget


class CalendarWidget(BaseOverlayWidget):
    """Widget displaying upcoming calendar events from Google Calendar."""

    def __init__(self, position=(100, 100)):
        super().__init__(
            position=position,
            size=(280, 180),
            min_size=(200, 120),
            max_size=(400, 300)
        )
        self.events = []
        self.last_fetch = 0
        self.fetch_interval = 60  # Fetch every 60 seconds
        self.is_fetching = False
        self.error_message = None
        self.accent_color = (100, 149, 237)  # Cornflower blue

    def _get_google_token(self):
        """Get Google access token from system keychain."""
        try:
            # Try macOS keychain
            result = subprocess.run(
                ['security', 'find-generic-password', '-s', 'camouflage',
                 '-a', 'google_access_token', '-w'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _fetch_events_thread(self):
        """Fetch calendar events in background thread."""
        try:
            token = self._get_google_token()
            if not token:
                self.error_message = "Not connected to Google"
                self.is_fetching = False
                return

            import urllib.request
            import urllib.error

            # Get events for next 24 hours
            now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            tomorrow = time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                    time.gmtime(time.time() + 86400))

            url = (f"https://www.googleapis.com/calendar/v3/calendars/primary/events"
                   f"?timeMin={now}&timeMax={tomorrow}&maxResults=5"
                   f"&singleEvents=true&orderBy=startTime")

            req = urllib.request.Request(url)
            req.add_header('Authorization', f'Bearer {token}')

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            self.events = []
            for item in data.get('items', []):
                start = item.get('start', {})
                start_time = start.get('dateTime', start.get('date', ''))
                self.events.append({
                    'summary': item.get('summary', 'No title'),
                    'start': start_time,
                })

            self.error_message = None

        except urllib.error.HTTPError as e:
            if e.code == 401:
                self.error_message = "Token expired"
            else:
                self.error_message = f"API error: {e.code}"
        except Exception as e:
            self.error_message = str(e)[:30]
        finally:
            self.is_fetching = False

    def fetch_events(self):
        """Start background fetch of calendar events."""
        current_time = time.time()
        if current_time - self.last_fetch < self.fetch_interval:
            return
        if self.is_fetching:
            return

        self.is_fetching = True
        self.last_fetch = current_time
        thread = threading.Thread(target=self._fetch_events_thread, daemon=True)
        thread.start()

    def _format_time(self, iso_time):
        """Format ISO time string to readable format."""
        try:
            if 'T' in iso_time:
                # DateTime format
                time_part = iso_time.split('T')[1][:5]
                # Convert 24h to 12h format
                hour, minute = int(time_part[:2]), time_part[3:5]
                ampm = 'AM' if hour < 12 else 'PM'
                hour = hour % 12 or 12
                return f"{hour}:{minute} {ampm}"
            else:
                # All-day event
                return "All day"
        except Exception:
            return iso_time[:10]

    def draw(self, frame, mirrored=True):
        """Draw the calendar widget."""
        if not self.visible:
            return

        self.fetch_events()

        x, y, w, h = self.draw_background(frame)

        # Title
        self.draw_text(frame, "Calendar", (x + 8, y + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, mirrored)

        # Calendar icon in title bar
        icon_x = x + w - 24
        cv2.rectangle(frame, (icon_x, y + 6), (icon_x + 16, y + 22), self.accent_color, 2)
        cv2.line(frame, (icon_x + 4, y + 6), (icon_x + 4, y + 3), self.accent_color, 2)
        cv2.line(frame, (icon_x + 12, y + 6), (icon_x + 12, y + 3), self.accent_color, 2)

        content_y = y + 40

        if self.is_fetching and not self.events:
            self.draw_text(frame, "Loading...", (x + 10, content_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, mirrored)
        elif self.error_message:
            self.draw_text(frame, self.error_message, (x + 10, content_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 100, 100), 1, mirrored)
        elif not self.events:
            self.draw_text(frame, "No upcoming events", (x + 10, content_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, mirrored)
        else:
            # Calculate how many events fit based on widget height
            available_height = h - 50  # Title bar + margins
            event_height = 32
            max_events = max(1, available_height // event_height)

            for i, event in enumerate(self.events[:max_events]):
                event_y = content_y + i * event_height
                if event_y + 30 > y + h - 10:
                    break

                # Time
                time_str = self._format_time(event['start'])
                self.draw_text(frame, time_str, (x + 10, event_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.accent_color, 1, mirrored)

                # Event title (truncate based on widget width)
                title = event['summary']
                max_chars = max(10, (w - 30) // 8)
                if len(title) > max_chars:
                    title = title[:max_chars - 3] + "..."
                self.draw_text(frame, title, (x + 10, event_y + 16),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1, mirrored)

        # Draw resize handles
        self.draw_resize_handles(frame)

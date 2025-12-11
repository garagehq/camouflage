"""Notifications widget displaying recent system notifications."""
import cv2
import time
import json
import subprocess
import threading
import platform
from pathlib import Path

from BaseWidget import BaseOverlayWidget


class NotificationsWidget(BaseOverlayWidget):
    """Widget displaying recent system notifications."""

    def __init__(self, position=(100, 100)):
        super().__init__(
            position=position,
            size=(280, 160),
            min_size=(200, 100),
            max_size=(400, 280)
        )
        self.notifications = []
        self.last_fetch = 0
        self.fetch_interval = 5  # Check every 5 seconds
        self.is_fetching = False
        self.max_notifications = 4
        self.accent_color = (255, 165, 0)  # Orange

    def _fetch_notifications_thread(self):
        """Fetch recent notifications in background thread."""
        try:
            system = platform.system()

            if system == 'Darwin':  # macOS
                self._fetch_macos_notifications()
            elif system == 'Linux':
                self._fetch_linux_notifications()
            else:
                self.notifications = [{'app': 'System', 'title': 'Not supported',
                                       'body': 'Notifications not available on this platform'}]

        except Exception as e:
            self.notifications = [{'app': 'Error', 'title': 'Fetch failed',
                                   'body': str(e)[:40]}]
        finally:
            self.is_fetching = False

    def _fetch_macos_notifications(self):
        """Fetch notifications from macOS Notification Center database."""
        try:
            import sqlite3
            import shutil
            import tempfile

            # macOS stores notifications in a SQLite database
            # Note: This requires Full Disk Access permission
            db_path = Path.home() / 'Library/Group Containers/group.com.apple.usernoted/db2/db'

            if not db_path.exists():
                # Try alternative path
                db_path = Path.home() / 'Library/Notifications/db2/db'

            if not db_path.exists():
                self.notifications = [{'app': 'Info', 'title': 'Notifications',
                                       'body': 'Grant Full Disk Access to view'}]
                return

            # Copy database to temp location (it may be locked)
            temp_db = Path(tempfile.gettempdir()) / 'notif_temp.db'
            shutil.copy2(db_path, temp_db)

            conn = sqlite3.connect(str(temp_db))
            cursor = conn.cursor()

            # Query recent notifications
            cursor.execute("""
                SELECT app_id, title, body, delivered_date
                FROM record
                ORDER BY delivered_date DESC
                LIMIT ?
            """, (self.max_notifications,))

            rows = cursor.fetchall()
            conn.close()
            temp_db.unlink(missing_ok=True)

            self.notifications = []
            for row in rows:
                app_id = row[0] or 'Unknown'
                # Extract app name from bundle ID
                app_name = app_id.split('.')[-1] if '.' in app_id else app_id
                self.notifications.append({
                    'app': app_name.capitalize(),
                    'title': row[1] or 'No title',
                    'body': row[2] or ''
                })

        except PermissionError:
            self.notifications = [{'app': 'Info', 'title': 'Permission needed',
                                   'body': 'Grant Full Disk Access in System Settings'}]
        except Exception:
            # Fall back to showing that notifications aren't accessible
            self.notifications = [{'app': 'Demo', 'title': 'Notifications Widget',
                                   'body': 'Simulated notification preview'}]

    def _fetch_linux_notifications(self):
        """Fetch notifications from Linux notification daemon."""
        try:
            # Try using notify-history if available
            result = subprocess.run(
                ['notify-history', '--json', '--limit', str(self.max_notifications)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                self.notifications = []
                for item in data:
                    self.notifications.append({
                        'app': item.get('app_name', 'Unknown'),
                        'title': item.get('summary', 'No title'),
                        'body': item.get('body', '')
                    })
                return
        except Exception:
            pass

        # Fallback to demo notifications
        self.notifications = [{'app': 'Demo', 'title': 'Notifications Widget',
                               'body': 'Connect notification service'}]

    def fetch_notifications(self):
        """Start background fetch of notifications."""
        current_time = time.time()
        if current_time - self.last_fetch < self.fetch_interval:
            return
        if self.is_fetching:
            return

        self.is_fetching = True
        self.last_fetch = current_time
        thread = threading.Thread(target=self._fetch_notifications_thread, daemon=True)
        thread.start()

    def draw(self, frame, mirrored=True):
        """Draw the notifications widget."""
        if not self.visible:
            return

        self.fetch_notifications()

        x, y, w, h = self.draw_background(frame)

        # Title
        self.draw_text(frame, "Notifications", (x + 8, y + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, mirrored)

        # Bell icon in title bar
        bell_x = x + w - 16
        cv2.ellipse(frame, (bell_x, y + 12), (8, 8), 0, 180, 360, self.accent_color, 2)
        cv2.line(frame, (bell_x - 8, y + 12), (bell_x + 8, y + 12), self.accent_color, 2)
        cv2.circle(frame, (bell_x, y + 20), 3, self.accent_color, -1)

        content_y = y + 40

        if self.is_fetching and not self.notifications:
            self.draw_text(frame, "Loading...", (x + 10, content_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, mirrored)
        elif not self.notifications:
            self.draw_text(frame, "No notifications", (x + 10, content_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, mirrored)
        else:
            # Calculate how many notifications fit based on widget height
            available_height = h - 50  # Title bar + margins
            notif_height = 28
            max_notifs = max(1, available_height // notif_height)

            for i, notif in enumerate(self.notifications[:max_notifs]):
                notif_y = content_y + i * notif_height
                if notif_y + 25 > y + h - 10:
                    break

                # App name
                app = notif['app']
                max_app_chars = min(12, (w - 30) // 10)
                if len(app) > max_app_chars:
                    app = app[:max_app_chars - 2] + ".."
                self.draw_text(frame, app, (x + 10, notif_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.accent_color, 1, mirrored)

                # Title (truncate based on widget width)
                title = notif['title']
                max_title_chars = max(10, (w - 30) // 8)
                if len(title) > max_title_chars:
                    title = title[:max_title_chars - 3] + "..."
                self.draw_text(frame, title, (x + 10, notif_y + 14),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1, mirrored)

        # Draw resize handles
        self.draw_resize_handles(frame)

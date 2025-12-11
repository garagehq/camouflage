"""Modern UI rendering utilities using Pillow for high-quality graphics.

Provides anti-aliased rounded rectangles, soft shadows, and better typography
for a polished, contemporary look similar to iOS/macOS widgets.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import os

# Try to find a good system font
FONT_PATHS = [
    "/System/Library/Fonts/SFNSText.ttf",  # macOS San Francisco
    "/System/Library/Fonts/Helvetica.ttc",  # macOS Helvetica
    "/System/Library/Fonts/SFNS.ttf",  # macOS newer SF
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch Linux
    "C:\\Windows\\Fonts\\segoeui.ttf",  # Windows Segoe UI
    "C:\\Windows\\Fonts\\arial.ttf",  # Windows Arial
]

_font_cache = {}


def get_font(size, bold=False):
    """Get a TrueType font at the specified size."""
    cache_key = (size, bold)
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    # Try each font path
    for path in FONT_PATHS:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _font_cache[cache_key] = font
                return font
            except Exception:
                continue

    # Fallback to default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    _font_cache[cache_key] = font
    return font


def create_rounded_rect_mask(size, radius):
    """Create an alpha mask for a rounded rectangle."""
    w, h = size
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, w-1, h-1], radius=radius, fill=255)
    return mask


def draw_shadow(size, radius, blur=12, offset=(0, 4), opacity=60):
    """Create a soft shadow image.

    Args:
        size: (width, height) of the element casting the shadow
        radius: Corner radius
        blur: Blur amount for softness
        offset: (x, y) shadow offset
        opacity: Shadow opacity (0-255)

    Returns:
        RGBA PIL Image with the shadow
    """
    # Create larger canvas for blur spread
    padding = blur * 2
    shadow_w = size[0] + padding * 2
    shadow_h = size[1] + padding * 2

    shadow = Image.new('RGBA', (shadow_w, shadow_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)

    # Draw shadow shape
    x1 = padding + offset[0]
    y1 = padding + offset[1]
    x2 = x1 + size[0]
    y2 = y1 + size[1]
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=(0, 0, 0, opacity))

    # Apply blur
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))

    return shadow, padding


class ModernCard:
    """A modern card/panel with rounded corners and shadow."""

    def __init__(self, width, height, corner_radius=16,
                 bg_color=(255, 255, 255, 240),
                 shadow_blur=12, shadow_offset=(0, 4), shadow_opacity=40,
                 scale=2.0):
        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.bg_color = bg_color
        self.shadow_blur = shadow_blur
        self.shadow_offset = shadow_offset
        self.shadow_opacity = shadow_opacity
        self.scale = scale # Supersampling factor

        # Pre-render the card background
        self._render_card()

    def _render_card(self):
        """Pre-render the card with shadow."""
        scaled_width = int(self.width * self.scale)
        scaled_height = int(self.height * self.scale)
        scaled_corner_radius = int(self.corner_radius * self.scale)
        scaled_shadow_blur = int(self.shadow_blur * self.scale)
        scaled_shadow_offset = (int(self.shadow_offset[0] * self.scale), int(self.shadow_offset[1] * self.scale))

        # Create shadow
        shadow, padding_val = draw_shadow(
            (scaled_width, scaled_height),
            scaled_corner_radius,
            blur=scaled_shadow_blur,
            offset=scaled_shadow_offset,
            opacity=self.shadow_opacity
        )
        self.padding = int(padding_val)

        # Create card on top of shadow
        card = Image.new('RGBA', shadow.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(card)

        # Draw card background
        x1 = self.padding
        y1 = self.padding
        x2 = x1 + scaled_width
        y2 = y1 + scaled_height
        draw.rounded_rectangle([x1, y1, x2, y2], radius=scaled_corner_radius, fill=self.bg_color)

        # Composite shadow and card
        self.image = Image.alpha_composite(shadow, card)
        self.draw = ImageDraw.Draw(self.image)

    def add_text(self, text, position, font_size=14, color=(0, 0, 0, 255), bold=False, anchor='lt', mirrored=False):
        """Add text to the card.

        Args:
            text: Text string
            position: (x, y) relative to card content area
            font_size: Font size in pixels
            color: RGBA color tuple
            bold: Use bold font
            anchor: PIL text anchor ('lt' = left-top, 'mm' = middle-middle, etc.)
            mirrored: If True, flip text horizontally
        """
        scaled_font_size = int(font_size * self.scale)
        font = get_font(scaled_font_size, bold)
        # Adjust position for padding and scaling
        x = self.padding + int(position[0] * self.scale)
        y = self.padding + int(position[1] * self.scale)

        if mirrored:
            # Get bounding box relative to 0,0 for scaled font
            bbox = self.draw.textbbox((0, 0), text, font=font, anchor=anchor)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            # Create temp image
            # Ensure strictly positive dimensions
            if w <= 0 or h <= 0: return
            
            txt_img = Image.new('RGBA', (w + int(4*self.scale), h + int(4*self.scale)), (0, 0, 0, 0)) # Add some padding
            d = ImageDraw.Draw(txt_img)
            
            # Draw text shifted to origin
            # Use bbox coordinates relative to the original text image's top-left corner
            ox = int(2*self.scale) - bbox[0]
            oy = int(2*self.scale) - bbox[1]

            # Draw text shadow/outline on temp image
            shadow_color = (255, 255, 255, 180)
            shadow_offset = max(1, int(1 * self.scale))

            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                d.text((ox + dx * shadow_offset, oy + dy * shadow_offset), text, font=font, fill=shadow_color, anchor=anchor)
            d.text((ox, oy), text, font=font, fill=color, anchor=anchor)
            
            # Flip
            txt_img = txt_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Paste back at calculated position
            # Paste at (x + bbox[0] - pad, y + bbox[1] - pad) relative to the card's self.image
            paste_x = x + bbox[0] - int(2*self.scale)
            paste_y = y + bbox[1] - int(2*self.scale)
            
            # Ensure paste area is within bounds
            if paste_x < 0: paste_x = 0
            if paste_y < 0: paste_y = 0
            if paste_x + txt_img.width > self.image.width: txt_img = txt_img.crop((0,0, self.image.width - paste_x, txt_img.height))
            if paste_y + txt_img.height > self.image.height: txt_img = txt_img.crop((0,0, txt_img.width, self.image.height - paste_y))

            self.image.alpha_composite(txt_img, (paste_x, paste_y))
        else:
            self.draw.text((x, y), text, font=font, fill=color, anchor=anchor)

    def add_button(self, bounds, text, font_size=12,
                   bg_color=(66, 133, 244, 255),  # Google blue
                   text_color=(255, 255, 255, 255),
                   corner_radius=8,
                   progress=0.0,
                   mirrored=False):
        """Add a button to the card.

        Args:
            bounds: (x, y, width, height) relative to card content area
            text: Button label
            font_size: Font size
            bg_color: Button background color (RGBA)
            text_color: Button text color (RGBA)
            corner_radius: Button corner radius
            progress: Optional progress indicator (0.0-1.0)
            mirrored: If True, flip text horizontally
        """
        x, y, w, h = [int(val * self.scale) for val in bounds]
        x1 = self.padding + x
        y1 = self.padding + y
        x2 = x1 + w
        y2 = y1 + h

        # Draw button background
        self.draw.rounded_rectangle([x1, y1, x2, y2], radius=int(corner_radius * self.scale), fill=bg_color)

        # Draw progress overlay if needed
        if progress > 0:
            progress_color = (255, 255, 255, 60)
            progress_width = int(w * progress)
            if progress_width > 0:
                self.draw.rounded_rectangle(
                    [x1, y1, x1 + progress_width, y2],
                    radius=int(corner_radius * self.scale),
                    fill=progress_color
                )

        # Draw text centered
        # No need to use get_font here as add_text handles it
        # Pass unscaled cx, cy to add_text as it will scale internally
        cx_unscaled = bounds[0] + bounds[2] // 2
        cy_unscaled = bounds[1] + bounds[3] // 2
        self.add_text(text, (cx_unscaled, cy_unscaled), font_size, text_color, anchor='mm', mirrored=mirrored)

    def add_icon_circle(self, center, radius, bg_color=(100, 100, 100, 255),
                        icon_color=(255, 255, 255, 255), icon_type='pencil',
                        is_active=False):
        """Add a circular icon.

        Args:
            center: (x, y) center relative to card content area
            radius: Circle radius
            bg_color: Background color
            icon_color: Icon color
            icon_type: Type of icon to draw
            is_active: Whether the icon represents an active state
        """
        scaled_cx = self.padding + int(center[0] * self.scale)
        scaled_cy = self.padding + int(center[1] * self.scale)
        scaled_radius = int(radius * self.scale)

        # Draw circle background
        self.draw.ellipse(
            [scaled_cx - scaled_radius, scaled_cy - scaled_radius, scaled_cx + scaled_radius, scaled_cy + scaled_radius],
            fill=bg_color
        )

        # Draw icon (simplified versions)
        self._draw_icon(scaled_cx, scaled_cy, scaled_radius, icon_type, icon_color, is_active)

    def _draw_icon(self, cx, cy, radius, icon_type, color, is_active):
        """Draw an icon symbol at the specified position."""
        # Scale icon based on radius
        s = radius / 20  # Base scale
        lw = max(1, int(2 * s))  # Line width, ensure minimum 1

        if icon_type == 'pencil':
            # Diagonal pencil line
            self.draw.line(
                [(cx - int(8*s), cy + int(8*s)), (cx + int(7*s), cy - int(7*s))],
                fill=color, width=lw
            )
            # Pencil tip
            self.draw.ellipse(
                [cx - int(10*s), cy + int(6*s), cx - int(6*s), cy + int(10*s)],
                fill=color
            )
        elif icon_type == 'eye':
            # Eye shape
            self.draw.ellipse(
                [cx - int(10*s), cy - int(5*s), cx + int(10*s), cy + int(5*s)],
                outline=color, width=lw
            )
            self.draw.ellipse(
                [cx - int(4*s), cy - int(4*s), cx + int(4*s), cy + int(4*s)],
                fill=color
            )
            if not is_active:
                # Strike-through
                self.draw.line(
                    [(cx - int(12*s), cy + int(10*s)), (cx + int(12*s), cy - int(10*s))],
                    fill=(255, 80, 80, 255), width=lw
                )
        elif icon_type == 'clock':
            # Clock circle
            self.draw.ellipse(
                [cx - int(10*s), cy - int(10*s), cx + int(10*s), cy + int(10*s)],
                outline=color, width=lw
            )
            # Clock hands
            self.draw.line([(cx, cy), (cx, cy - int(6*s))], fill=color, width=lw)
            self.draw.line([(cx, cy), (cx + int(4*s), cy)], fill=color, width=lw)
        elif icon_type == 'pause':
            # Two vertical bars
            bar_color = (255, 180, 0, 255) if is_active else color
            bar_w = int(4*s)
            bar_h = int(12*s)
            gap = int(3*s)
            self.draw.rectangle(
                [cx - gap - bar_w, cy - bar_h//2, cx - gap, cy + bar_h//2],
                fill=bar_color
            )
            self.draw.rectangle(
                [cx + gap, cy - bar_h//2, cx + gap + bar_w, cy + bar_h//2],
                fill=bar_color
            )
        elif icon_type == 'calendar':
            # Calendar body
            self.draw.rectangle(
                [cx - int(8*s), cy - int(6*s), cx + int(8*s), cy + int(8*s)],
                outline=color, width=lw
            )
            # Calendar top bar
            self.draw.line(
                [(cx - int(8*s), cy - int(2*s)), (cx + int(8*s), cy - int(2*s))],
                fill=color, width=lw
            )
            # Calendar rings
            self.draw.line([(cx - int(4*s), cy - int(6*s)), (cx - int(4*s), cy - int(9*s))], fill=color, width=lw)
            self.draw.line([(cx + int(4*s), cy - int(6*s)), (cx + int(4*s), cy - int(9*s))], fill=color, width=lw)
        elif icon_type == 'trash':
            # Trash can body
            self.draw.rectangle(
                [cx - int(6*s), cy - int(4*s), cx + int(6*s), cy + int(8*s)],
                outline=color, width=lw
            )
            # Trash can lid
            self.draw.line(
                [(cx - int(8*s), cy - int(4*s)), (cx + int(8*s), cy - int(4*s))],
                fill=color, width=lw
            )
            # Trash can handle
            self.draw.rectangle(
                [cx - int(3*s), cy - int(7*s), cx + int(3*s), cy - int(4*s)],
                outline=color, width=lw
            )
        elif icon_type == 'undo':
            # Curved arrow
            self.draw.arc(
                [cx - int(8*s), cy - int(4*s), cx + int(8*s), cy + int(8*s)],
                start=180, end=360, fill=color, width=lw
            )
            # Arrow head
            self.draw.polygon(
                [(cx - int(8*s), cy + int(2*s)),
                 (cx - int(12*s), cy - int(2*s)),
                 (cx - int(4*s), cy - int(2*s))],
                fill=color
            )

    def to_numpy(self):
        """Convert to numpy array (BGRA format for OpenCV)."""
        # Resize down to original dimensions before converting
        img_resized = self.image.resize((self.width, self.height), resample=Image.LANCZOS)
        arr = np.array(img_resized)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

    def composite_onto(self, frame, position):
        """Composite this card onto an OpenCV frame (BGR).

        Args:
            frame: OpenCV BGR frame
            position: (x, y) position for the card (top-left of content, not shadow)
        """
        card_bgra = self.to_numpy()

        # Calculate actual position accounting for shadow padding
        x = position[0] - self.padding
        y = position[1] - self.padding

        h, w = card_bgra.shape[:2]

        # Clamp to frame bounds
        frame_h, frame_w = frame.shape[:2]

        # Source region (card)
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        src_x2 = min(w, frame_w - x)
        src_y2 = min(h, frame_h - y)

        # Destination region (frame)
        dst_x1 = max(0, x)
        dst_y1 = max(0, y)
        dst_x2 = min(frame_w, x + w)
        dst_y2 = min(frame_h, y + h)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return

        # Extract regions
        src_region = card_bgra[src_y1:src_y2, src_x1:src_x2]
        dst_region = frame[dst_y1:dst_y2, dst_x1:dst_x2]

        # Alpha blend
        alpha = src_region[:, :, 3:4].astype(float) / 255.0
        blended = (src_region[:, :, :3].astype(float) * alpha +
                   dst_region.astype(float) * (1 - alpha))

        frame[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)


class ModernPieMenu:
    """A modern-styled pie menu with smooth rendering."""

    def __init__(self, items, inner_radius=40, outer_radius=90, corner_radius=12, mirrored=False):
        self.items = items
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.corner_radius = corner_radius

        # Colors
        self.bg_color = (255, 255, 255, 230)  # Semi-transparent white
        self.section_inactive = (255, 255, 255, 30)  # Very transparent white
        self.section_active = (66, 133, 244, 230)  # Blue with high opacity
        self.section_hover = (210, 230, 255, 150)  # Light blue with medium opacity
        self.border_color = (200, 200, 200, 255)
        self.text_color = (40, 40, 40, 255)  # Darker text for visibility
        self.icon_color = (60, 60, 60, 255)  # Darker icons
        self.accent_color = (66, 133, 244, 255)  # Google blue
        
        # Rendering scale for supersampling (antialiasing)
        self.scale = 2.0
        self.mirrored = mirrored

    def render(self, center, selected_index=-1, item_states=None, progress=0.0, hold_progress=0.0, mirrored=False):
        """Render the pie menu.

        Args:
            center: (x, y) center position (will be in center of returned image)
            selected_index: Currently selected section (-1 for none)
            item_states: List of booleans indicating active state per item
            progress: Selection progress for selected item (0.0-1.0)
            hold_progress: Initial hold progress before menu opens (0.0-1.0)

        Returns:
            (image, offset) where image is BGRA numpy array and offset is (x, y) to subtract from center
        """
        import math

        if item_states is None:
            item_states = [False] * len(self.items)

        # Calculate canvas size (menu + labels + shadow padding)
        # We work at scaled resolution
        padding = 70
        base_size = (self.outer_radius + padding) * 2
        size = int(base_size * self.scale)
        
        # Scaled dimensions
        s_inner_r = self.inner_radius * self.scale
        s_outer_r = self.outer_radius * self.scale

        # Create canvas
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx, cy = size // 2, size // 2
        n = len(self.items)

        # If still in hold phase, just draw progress circle
        if hold_progress < 1.0:
            # Draw background circle
            r_prog = 30 * self.scale
            draw.ellipse(
                [cx - r_prog, cy - r_prog, cx + r_prog, cy + r_prog],
                outline=(200, 200, 200, 200), width=int(3 * self.scale)
            )
            # Draw progress arc
            end_angle = -90 + (360 * hold_progress)
            draw.arc(
                [cx - r_prog, cy - r_prog, cx + r_prog, cy + r_prog],
                start=-90, end=end_angle,
                fill=self.accent_color, width=int(4 * self.scale)
            )
            # Resize down
            img = img.resize((int(base_size), int(base_size)), resample=Image.LANCZOS)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA), (int(base_size) // 2, int(base_size) // 2)

        # Draw shadow first
        shadow = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        offset_val = 4 * self.scale
        shadow_draw.ellipse(
            [cx - s_outer_r + offset_val, cy - s_outer_r + offset_val * 1.5,
             cx + s_outer_r + offset_val, cy + s_outer_r + offset_val * 1.5],
            fill=(0, 0, 0, 30)
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(8 * self.scale))
        img = Image.alpha_composite(shadow, img)
        draw = ImageDraw.Draw(img)

        # Draw each section
        section_angle = 360 / n

        for i in range(n):
            start_angle = -90 + (i * section_angle) - (section_angle / 2)
            end_angle = start_angle + section_angle

            # Determine section color
            if i == selected_index:
                color = self.section_hover
            elif item_states[i]:
                color = self.section_active
            else:
                color = self.section_inactive

            # Draw section as pie slice with hole
            self._draw_donut_section(draw, cx, cy, self.inner_radius, self.outer_radius,
                                    start_angle, end_angle, color)

        # Draw outer ring border
        draw.ellipse(
            [cx - s_outer_r, cy - s_outer_r,
             cx + s_outer_r, cy + s_outer_r],
            outline=self.border_color, width=int(2 * self.scale)
        )

        # Draw inner ring border
        draw.ellipse(
            [cx - s_inner_r, cy - s_inner_r,
             cx + s_inner_r, cy + s_inner_r],
            outline=self.border_color, width=int(2 * self.scale)
        )

        # Draw section dividers, icons, and labels
        mid_radius = (s_inner_r + s_outer_r) / 2
        font_size = int(13 * self.scale)
        font = get_font(font_size)
        border_width = max(1, int(1 * self.scale))

        for i, item in enumerate(self.items):
            angle_deg = -90 + (i * section_angle)
            angle_rad = math.radians(angle_deg)

            # Section divider
            div_angle = math.radians(-90 + (i * section_angle) - (section_angle / 2))
            x1 = cx + s_inner_r * math.cos(div_angle)
            y1 = cy + s_inner_r * math.sin(div_angle)
            x2 = cx + s_outer_r * math.cos(div_angle)
            y2 = cy + s_outer_r * math.sin(div_angle)
            draw.line([(x1, y1), (x2, y2)], fill=self.border_color, width=border_width)

            # Icon position
            ix = int(cx + mid_radius * math.cos(angle_rad))
            iy = int(cy + mid_radius * math.sin(angle_rad))

            # Draw icon
            icon_type = item.get('icon', 'pencil') if isinstance(item, dict) else 'pencil'
            is_active = item_states[i] if i < len(item_states) else False
            self._draw_icon_pil(draw, ix, iy, icon_type, self.icon_color, is_active, self.scale)

            # Label position (outside the ring)
            label_radius = s_outer_r + (20 * self.scale)
            lx = int(cx + label_radius * math.cos(angle_rad))
            ly = int(cy + label_radius * math.sin(angle_rad))

            label = item.get('label', '') if isinstance(item, dict) else str(item)
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                text_x = lx - tw//2
                text_y = ly - th//2
                
                shadow_color = (255, 255, 255, 180)
                shadow_offset = max(1, int(1 * self.scale))
                
                # If we are going to flip the whole image later (self.mirrored),
                # we don't need to pre-flip the text here.
                # Only flip text if requested (mirrored arg) AND we aren't flipping the whole image.
                should_flip_text = mirrored and not self.mirrored

                if should_flip_text and tw > 0 and th > 0:
                    # Render text to temp image and flip
                    # Add padding for shadow/outline (2px on each side)
                    pad = int(2 * self.scale)
                    txt_img = Image.new('RGBA', (tw + pad*2, th + pad*2), (0, 0, 0, 0))
                    d = ImageDraw.Draw(txt_img)
                    
                    # Draw shadow/outline on temp image
                    # Shift origin by (pad - bbox[0], pad - bbox[1])
                    ox = pad - bbox[0]
                    oy = pad - bbox[1]
                    
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                        d.text((ox + dx * shadow_offset, oy + dy * shadow_offset), label, font=font, fill=shadow_color)
                    d.text((ox, oy), label, font=font, fill=self.text_color)
                    
                    # Flip and composite
                    txt_img = txt_img.transpose(Image.FLIP_LEFT_RIGHT)
                    # Paste at (text_x - pad, text_y - pad)
                    img.alpha_composite(txt_img, (text_x - pad, text_y - pad))
                else:
                    # Draw text shadow/outline directly
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                        draw.text((text_x + dx * shadow_offset, text_y + dy * shadow_offset), label, font=font, fill=shadow_color)
                    draw.text((text_x, text_y), label, font=font, fill=self.text_color)

        # Draw selection progress arc
        if selected_index >= 0 and progress > 0:
            start_angle = -90 + (selected_index * section_angle) - (section_angle / 2)
            progress_end = start_angle + (section_angle * progress)
            prog_offset = 4 * self.scale
            draw.arc(
                [cx - s_outer_r - prog_offset, cy - s_outer_r - prog_offset,
                 cx + s_outer_r + prog_offset, cy + s_outer_r + prog_offset],
                start=start_angle, end=progress_end,
                fill=self.accent_color, width=int(4 * self.scale)
            )

        # Resize down for high quality anti-aliasing
        img = img.resize((int(base_size), int(base_size)), resample=Image.LANCZOS)

        # Flip the image horizontally if self.mirrored is True (from init)
        if self.mirrored:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA), (int(base_size) // 2, int(base_size) // 2)

    def _draw_donut_section(self, draw, cx, cy, inner_r, outer_r, start_angle, end_angle, color):
        """Draw a donut section (pie slice with inner cutout)."""
        import math

        # Create polygon for the section
        points = []
        num_points = 20

        # Outer arc
        for i in range(num_points + 1):
            angle = math.radians(start_angle + (end_angle - start_angle) * i / num_points)
            x = cx + outer_r * math.cos(angle)
            y = cy + outer_r * math.sin(angle)
            points.append((x, y))

        # Inner arc (reversed)
        for i in range(num_points + 1):
            angle = math.radians(end_angle - (end_angle - start_angle) * i / num_points)
            x = cx + inner_r * math.cos(angle)
            y = cy + inner_r * math.sin(angle)
            points.append((x, y))

        draw.polygon(points, fill=color)

    def _draw_icon_pil(self, draw, cx, cy, icon_type, color, is_active, scale=1.0):
        """Draw an icon using PIL."""
        s = scale  # Scale
        lw = max(1, int(2 * scale))

        if icon_type == 'pencil':
            draw.line([(cx - 8*s, cy + 8*s), (cx + 7*s, cy - 7*s)], fill=color, width=lw)
            draw.ellipse([cx - 10*s, cy + 6*s, cx - 6*s, cy + 10*s], fill=color)
        elif icon_type == 'eye':
            draw.ellipse([cx - 10*s, cy - 5*s, cx + 10*s, cy + 5*s], outline=color, width=lw)
            draw.ellipse([cx - 4*s, cy - 4*s, cx + 4*s, cy + 4*s], fill=color)
            if not is_active:
                draw.line([(cx - 12*s, cy + 10*s), (cx + 12*s, cy - 10*s)], fill=(255, 80, 80, 255), width=lw)
        elif icon_type == 'clock':
            draw.ellipse([cx - 10*s, cy - 10*s, cx + 10*s, cy + 10*s], outline=color, width=lw)
            draw.line([(cx, cy), (cx, cy - 6*s)], fill=color, width=lw)
            draw.line([(cx, cy), (cx + 4*s, cy)], fill=color, width=lw)
        elif icon_type == 'pause':
            bar_color = (255, 180, 0, 255) if is_active else color
            draw.rectangle([cx - 7*s, cy - 8*s, cx - 2*s, cy + 8*s], fill=bar_color)
            draw.rectangle([cx + 2*s, cy - 8*s, cx + 7*s, cy + 8*s], fill=bar_color)
        elif icon_type == 'undo':
            draw.arc([cx - 8*s, cy - 4*s, cx + 8*s, cy + 8*s], start=180, end=360, fill=color, width=lw)
            draw.polygon([(cx - 8*s, cy + 2*s), (cx - 12*s, cy - 2*s), (cx - 4*s, cy - 2*s)], fill=color)
        elif icon_type == 'trash':
            draw.rectangle([cx - 6*s, cy - 4*s, cx + 6*s, cy + 8*s], outline=color, width=lw)
            draw.line([(cx - 8*s, cy - 4*s), (cx + 8*s, cy - 4*s)], fill=color, width=lw)
            draw.rectangle([cx - 3*s, cy - 7*s, cx + 3*s, cy - 4*s], outline=color, width=lw)
        elif icon_type == 'calendar':
            draw.rectangle([cx - 8*s, cy - 6*s, cx + 8*s, cy + 8*s], outline=color, width=lw)
            draw.line([(cx - 8*s, cy - 2*s), (cx + 8*s, cy - 2*s)], fill=color, width=lw)
            draw.line([(cx - 4*s, cy - 6*s), (cx - 4*s, cy - 9*s)], fill=color, width=lw)
            draw.line([(cx + 4*s, cy - 6*s), (cx + 4*s, cy - 9*s)], fill=color, width=lw)
        elif icon_type == 'cloud':
            draw.ellipse([cx - 10*s, cy - 5*s, cx + 2*s, cy + 5*s], outline=color, width=lw)
            draw.ellipse([cx - 2*s, cy - 8*s, cx + 10*s, cy + 4*s], outline=color, width=lw)
        elif icon_type == 'bell':
            draw.arc([cx - 8*s, cy - 10*s, cx + 8*s, cy + 6*s], start=180, end=360, fill=color, width=lw)
            draw.line([(cx - 10*s, cy + 6*s), (cx + 10*s, cy + 6*s)], fill=color, width=lw)
            draw.ellipse([cx - 3*s, cy + 6*s, cx + 3*s, cy + 12*s], fill=color)


class ModernTimerWidget:
    """A modern-styled timer widget."""

    def __init__(self):
        self.width = 220
        self.height = 100
        self.corner_radius = 16
        self.button_radius = 8

        # Colors
        self.bg_color = (255, 255, 255, 245)
        self.title_color = (120, 120, 120, 255)
        self.time_color = (30, 30, 30, 255)
        self.time_running_color = (34, 197, 94, 255)  # Green
        self.time_paused_color = (234, 179, 8, 255)  # Yellow/amber
        self.button_color = (66, 133, 244, 255)  # Blue
        self.button_secondary = (229, 231, 235, 255)  # Light gray
        self.button_text = (255, 255, 255, 255)
        self.button_text_secondary = (60, 60, 60, 255)

    def render(self, elapsed_time, is_running, start_pause_progress=0.0, reset_progress=0.0, mirrored=False, scale=1.0):
        """Render the timer widget.

        Returns:
            (image, content_offset) - BGRA numpy array and offset to content area
        """
        # Format time
        total_secs = int(elapsed_time)
        centisecs = int((elapsed_time - total_secs) * 100)
        hours = total_secs // 3600
        mins = (total_secs % 3600) // 60
        secs = total_secs % 60

        if hours > 0:
            time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
        else:
            time_str = f"{mins:02d}:{secs:02d}.{centisecs:02d}"

        # Create card
        card_scale = 2.0 * scale
        card = ModernCard(
            self.width, self.height,
            corner_radius=self.corner_radius,
            bg_color=self.bg_color,
            shadow_blur=15,
            shadow_offset=(0, 6),
            shadow_opacity=35,
            scale=card_scale
        )

        # Title
        card.add_text("Timer", (12, 12), font_size=12, color=self.title_color, mirrored=mirrored)

        # Time display
        time_color = self.time_running_color if is_running else (
            self.time_paused_color if elapsed_time > 0 else self.time_color
        )
        card.add_text(time_str, (self.width // 2, 48), font_size=28, color=time_color, anchor='mm', mirrored=mirrored)

        # Buttons
        btn_width = 85
        btn_height = 28
        btn_y = self.height - btn_height - 12
        btn_gap = 10

        # Start/Pause button (primary)
        sp_text = "Pause" if is_running else "Start"
        sp_color = self.button_color
        if start_pause_progress > 0:
            # Lighten during progress
            sp_color = tuple(min(255, c + int(40 * start_pause_progress)) for c in sp_color[:3]) + (255,)

        card.add_button(
            (12, btn_y, btn_width, btn_height),
            sp_text,
            font_size=12,
            bg_color=sp_color,
            text_color=self.button_text,
            corner_radius=self.button_radius,
            progress=start_pause_progress,
            mirrored=mirrored
        )

        # Reset button (secondary)
        reset_color = self.button_secondary
        if reset_progress > 0:
            reset_color = tuple(min(255, c + int(30 * reset_progress)) for c in reset_color[:3]) + (255,)

        card.add_button(
            (12 + btn_width + btn_gap, btn_y, btn_width, btn_height),
            "Reset",
            font_size=12,
            bg_color=reset_color,
            text_color=self.button_text_secondary,
            corner_radius=self.button_radius,
            progress=reset_progress,
            mirrored=mirrored
        )

        return card.to_numpy(), card.padding


def composite_bgra_onto_bgr(bgra_img, bgr_frame, position, padding=0):
    """Composite a BGRA image onto a BGR frame.

    Args:
        bgra_img: BGRA numpy array (with alpha channel)
        bgr_frame: BGR numpy array (destination)
        position: (x, y) position for content (padding will be subtracted)
        padding: Padding to subtract from position for shadow accommodation
    """
    x = position[0] - padding
    y = position[1] - padding

    h, w = bgra_img.shape[:2]
    frame_h, frame_w = bgr_frame.shape[:2]

    # Calculate visible regions
    src_x1 = max(0, -x)
    src_y1 = max(0, -y)
    src_x2 = min(w, frame_w - x)
    src_y2 = min(h, frame_h - y)

    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = min(frame_w, x + w)
    dst_y2 = min(frame_h, y + h)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return

    # Extract regions
    src_region = bgra_img[src_y1:src_y2, src_x1:src_x2]
    dst_region = bgr_frame[dst_y1:dst_y2, dst_x1:dst_x2]

    # Alpha blend
    alpha = src_region[:, :, 3:4].astype(float) / 255.0
    blended = (src_region[:, :, :3].astype(float) * alpha +
               dst_region.astype(float) * (1 - alpha))

    bgr_frame[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)

import numpy as np
import cv2
import pyautogui
import mediapipe as mp
from openni import openni2
import tkinter as tk
from PIL import ImageGrab, ImageTk, Image
import threading
import time
import os
from contextlib import contextmanager
from collections import defaultdict, deque


class Config:
    """Configuration constants"""
    OPENNI_PATH = r'C:\Users\dell laptop\vscod\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs'
    DEFAULT_GRID_RESOLUTION = 5
    DEFAULT_TOUCH_THRESHOLD = 50
    DEFAULT_CLICK_THRESHOLD = 700
    DEFAULT_GRID_SEARCH_RADIUS = 300
    
    # Enhanced tracking landmarks with full body context
    HAND_LANDMARKS = {
        'RIGHT_WRIST': 16,
        'LEFT_WRIST': 15,
        'RIGHT_INDEX': 20,  # Right index fingertip
        'LEFT_INDEX': 19,   # Left index fingertip
        'RIGHT_PINKY': 18,  # Right pinky
        'LEFT_PINKY': 17    # Left pinky
    }
    
    # Body landmarks for improved tracking stability
    BODY_LANDMARKS = {
        'NOSE': 0,
        'RIGHT_SHOULDER': 12,
        'LEFT_SHOULDER': 11,
        'RIGHT_HIP': 24,
        'LEFT_HIP': 23
    }
    
    SQUARE_COLORS = [(0, 255, 0), (255, 0, 0)]
    MAX_PEOPLE = 4
    
    # Smoothing parameters
    SMOOTHING_WINDOW = 5  # frames
    CONFIDENCE_THRESHOLD = 0.7
    VISIBILITY_THRESHOLD = 0.5


class BodyTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=1,  # Keep at 1 for performance
            min_detection_confidence=Config.CONFIDENCE_THRESHOLD,
            min_tracking_confidence=0.8,  # Higher for stability
            smooth_landmarks=True,
            enable_segmentation=False  # Disable for performance
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_bodies(self, img, draw=True):
        """Process image and extract pose landmarks"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks and draw:
            # Draw only essential connections for performance
            self._draw_essential_skeleton(img)
        
        return img

    def _draw_essential_skeleton(self, img):
        """Draw only essential skeleton parts for performance"""
        if not self.results.pose_landmarks:
            return
            
        landmarks = self.results.pose_landmarks.landmark
        h, w = img.shape[:2]
        
        # Draw main body structure
        essential_connections = [
            # Torso
            (Config.BODY_LANDMARKS['LEFT_SHOULDER'], Config.BODY_LANDMARKS['RIGHT_SHOULDER']),
            (Config.BODY_LANDMARKS['LEFT_SHOULDER'], Config.BODY_LANDMARKS['LEFT_HIP']),
            (Config.BODY_LANDMARKS['RIGHT_SHOULDER'], Config.BODY_LANDMARKS['RIGHT_HIP']),
            (Config.BODY_LANDMARKS['LEFT_HIP'], Config.BODY_LANDMARKS['RIGHT_HIP']),
            
            # Arms to hands
            (Config.BODY_LANDMARKS['LEFT_SHOULDER'], Config.HAND_LANDMARKS['LEFT_WRIST']),
            (Config.BODY_LANDMARKS['RIGHT_SHOULDER'], Config.HAND_LANDMARKS['RIGHT_WRIST']),
        ]
        
        for start_idx, end_idx in essential_connections:
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx].visibility > Config.VISIBILITY_THRESHOLD and
                landmarks[end_idx].visibility > Config.VISIBILITY_THRESHOLD):
                
                start_pos = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end_pos = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(img, start_pos, end_pos, (0, 255, 0), 2)

    def get_hand_positions(self):
        """Extract hand positions with body context for stability"""
        if not self.results.pose_landmarks:
            return {}
            
        landmarks = self.results.pose_landmarks.landmark
        hand_positions = {}
        
        # Get all relevant hand landmarks
        for name, idx in Config.HAND_LANDMARKS.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                if (landmark.visibility > Config.VISIBILITY_THRESHOLD and 
                    landmark.x > 0 and landmark.y > 0):
                    hand_positions[name] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }
        
        # Add body context for stability
        body_context = {}
        for name, idx in Config.BODY_LANDMARKS.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                if landmark.visibility > Config.VISIBILITY_THRESHOLD:
                    body_context[name] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }
        
        return {
            'hands': hand_positions,
            'body': body_context,
            'timestamp': time.time()
        }

    def close(self):
        if hasattr(self, 'pose'):
            self.pose.close()


class PersonState:
    """Enhanced class to track state for each person with smoothing"""
    def __init__(self, person_id):
        self.person_id = person_id
        self.active_squares = {}
        self.active_touches = {}
        self.click_cooldown = {}
        self.last_seen = time.time()
        self.last_centroid = (0, 0)
        
        # Smoothing buffers for each hand landmark
        self.position_history = {}
        self.smoothed_positions = {}
        
        # Initialize tracking data for each hand landmark
        for hand_name in Config.HAND_LANDMARKS.keys():
            self.click_cooldown[hand_name] = 0
            self.position_history[hand_name] = deque(maxlen=Config.SMOOTHING_WINDOW)
            self.smoothed_positions[hand_name] = None
        
        # Body tracking for stability
        self.body_history = {}
        for body_name in Config.BODY_LANDMARKS.keys():
            self.body_history[body_name] = deque(maxlen=Config.SMOOTHING_WINDOW)

    def update_positions(self, pose_data):
        """Update position history with smoothing"""
        current_time = time.time()
        self.last_seen = current_time
        
        # Update hand positions with smoothing
        for hand_name, position in pose_data.get('hands', {}).items():
            self.position_history[hand_name].append({
                'position': position,
                'timestamp': current_time
            })
            
            # Calculate smoothed position
            if len(self.position_history[hand_name]) >= 2:
                self.smoothed_positions[hand_name] = self._calculate_smoothed_position(hand_name)
        
        # Update body positions for stability reference
        for body_name, position in pose_data.get('body', {}).items():
            self.body_history[body_name].append({
                'position': position,
                'timestamp': current_time
            })

    def _calculate_smoothed_position(self, hand_name):
        """Calculate smoothed position using weighted average"""
        history = self.position_history[hand_name]
        if len(history) < 2:
            return history[-1]['position'] if history else None
        
        # Use weighted average with more weight on recent positions
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        weighted_z = 0
        
        for i, entry in enumerate(history):
            weight = (i + 1) / len(history)  # More weight for recent positions
            position = entry['position']
            
            weighted_x += position['x'] * weight
            weighted_y += position['y'] * weight
            weighted_z += position['z'] * weight
            total_weight += weight
        
        return {
            'x': weighted_x / total_weight,
            'y': weighted_y / total_weight,
            'z': weighted_z / total_weight,
            'visibility': history[-1]['position']['visibility']
        }

    def get_stability_score(self):
        """Calculate how stable the person's tracking is"""
        if len(self.position_history) < 2:
            return 0.0
        
        # Check variance in body position for stability
        body_positions = list(self.body_history.get('NOSE', []))
        if len(body_positions) < 2:
            return 0.5
        
        # Calculate position variance
        recent_positions = body_positions[-3:]
        if len(recent_positions) < 2:
            return 0.5
        
        x_variance = np.var([p['position']['x'] for p in recent_positions])
        y_variance = np.var([p['position']['y'] for p in recent_positions])
        
        # Lower variance = higher stability
        stability = 1.0 / (1.0 + (x_variance + y_variance) * 100)
        return min(1.0, max(0.0, stability))


class ScreenFieldControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Multi-Person Skeleton Tracking")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize state variables
        self._initialize_state()
        
        # Create UI
        self._create_ui()
        
        # Initialize enhanced body tracker
        self.body_tracker = BodyTracker()
        
        # Enhanced person tracking
        self.persons = {}  # key: person_id, value: PersonState
        self.next_person_id = 0
        self.tracking_quality_threshold = 0.6

    def _initialize_state(self):
        """Initialize all state variables"""
        # Screen field selection variables
        self.capture_regions = [
            {"start_x": None, "start_y": None, "end_x": None, "end_y": None, "selection_made": False}
            for _ in range(2)
        ]
        self.current_region = 0
        self.capturing = False
        self.capture_thread = None

        # Camera square mapping variables
        self.square_data = [
            {
                "points_2d": [], "points_3d": [], "grid_points_2d": [], "grid_points_3d": [],
                "homography_matrix": None, "nearest_grid_point": None
            }
            for _ in range(2)
        ]
        self.current_square = 0
        self.mapping_active = False
        self.camera_thread = None
        
        # Configuration
        self.grid_resolution = Config.DEFAULT_GRID_RESOLUTION
        self.touch_threshold = Config.DEFAULT_TOUCH_THRESHOLD
        self.grid_search_radius = Config.DEFAULT_GRID_SEARCH_RADIUS
        
        # Tracking state
        self.latest_depth_array = None
        self.is_hd_mode = True
        
        # UI state
        self.screen_windows = [None, None]
        self.screen_labels = [None, None]
        self.selection_window = None
        self.canvas = None
        self.rect = None
        
        # Real-time update state
        self.window_dimensions = [{}, {}]
        self.last_captured_images = [None, None]
        
        # Wall plane variables
        self.wall_plane = None
        self.calibration_points = []
        self.calibration_mode = False

    def _create_ui(self):
        """Create the user interface"""
        # Main control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Enhanced skeleton tracking ready", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Person count display with quality info
        self.person_count_label = tk.Label(control_frame, text="Persons: 0", font=("Arial", 10))
        self.person_count_label.pack(pady=5)
        
        # Tracking quality display
        self.quality_label = tk.Label(control_frame, text="Tracking Quality: N/A", font=("Arial", 9))
        self.quality_label.pack(pady=2)
        
        # Region selection instructions
        region_label = tk.Label(control_frame, text="Select and configure the two regions:")
        region_label.pack(pady=5)
        
        # Region controls
        self._create_region_controls(control_frame, 0)
        self._create_region_controls(control_frame, 1)
        self._create_global_controls(control_frame)

    def _create_region_controls(self, parent, region_idx):
        """Create controls for a specific region"""
        region_frame = tk.LabelFrame(parent, text=f"Region {region_idx + 1}")
        region_frame.pack(fill=tk.X, padx=5, pady=5)
        
        screen_button = tk.Button(
            region_frame, 
            text=f"Choose Screen Field {region_idx + 1}", 
            command=lambda: self.open_selection_window(region_idx)
        )
        screen_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        mapping_button = tk.Button(
            region_frame, 
            text=f"3D Square Mapping {region_idx + 1}", 
            command=lambda: self.start_square_mapping(region_idx),
            state=tk.DISABLED
        )
        mapping_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        if not hasattr(self, 'screen_buttons'):
            self.screen_buttons = []
            self.mapping_buttons = []
        
        self.screen_buttons.append(screen_button)
        self.mapping_buttons.append(mapping_button)

    def _create_global_controls(self, parent):
        """Create global control buttons"""
        self.start_capture_button = tk.Button(
            parent, 
            text="Start Enhanced Tracking", 
            command=self.start_capture, 
            state=tk.DISABLED
        )
        self.start_capture_button.pack(pady=5)
        
        self.stop_button = tk.Button(
            parent, 
            text="Stop All", 
            command=self.stop_all, 
            state=tk.DISABLED
        )
        self.stop_button.pack(pady=10)

    # [Previous UI methods remain the same - open_selection_window, on_button_press, etc.]
    def open_selection_window(self, region_idx):
        """Open window for screen region selection"""
        self.current_region = region_idx
        self.selection_window = tk.Toplevel(self.root)
        self.selection_window.attributes("-fullscreen", True)
        self.selection_window.attributes("-alpha", 0.3)
        self.selection_window.configure(bg='black')
        
        self.canvas = tk.Canvas(self.selection_window, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.selection_window.bind("<Button-1>", self.on_button_press)
        self.selection_window.bind("<B1-Motion>", self.on_mouse_drag)
        self.selection_window.bind("<ButtonRelease-1>", self.on_button_release)
        self.selection_window.bind("<Escape>", lambda e: self.selection_window.destroy())
        
        # Add instruction text
        self.canvas.create_text(
            self.selection_window.winfo_screenwidth() // 2, 50,
            text=f"Drag to select Region {region_idx + 1} (Press Escape to cancel)",
            fill="white", font=("Arial", 16)
        )

    def on_button_press(self, event):
        """Handle mouse button press for region selection"""
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2
        )

    def on_mouse_drag(self, event):
        """Handle mouse drag for region selection"""
        if hasattr(self, 'start_x') and hasattr(self, 'start_y'):
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        """Handle mouse button release for region selection"""
        if not hasattr(self, 'start_x'):
            return
            
        region = self.capture_regions[self.current_region]
        region["start_x"] = min(self.start_x, event.x)
        region["start_y"] = min(self.start_y, event.y)
        region["end_x"] = max(self.start_x, event.x)
        region["end_y"] = max(self.start_y, event.y)
        region["selection_made"] = True
        
        self.selection_window.destroy()
        self.selection_window = None
        
        # Update UI state
        self.mapping_buttons[self.current_region].config(state=tk.NORMAL)
        self._update_capture_button_state()
        self.status_label.config(text=f"Region {self.current_region + 1} selected")

    def _update_capture_button_state(self):
        """Update the state of the capture button based on selections"""
        all_selected = all(region["selection_made"] for region in self.capture_regions)
        if all_selected:
            self.start_capture_button.config(state=tk.NORMAL)

    def start_capture(self):
        """Start screen capture for all regions"""
        if self.capturing:
            return
            
        self.capturing = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Enhanced skeleton tracking started")

    def capture_loop(self):
        """Main capture loop for screen regions with continuous real-time updates"""
        try:
            while self.capturing:
                for i, region in enumerate(self.capture_regions):
                    if not region["selection_made"]:
                        continue
                    
                    # Capture the screen content
                    bbox = (region["start_x"], region["start_y"], region["end_x"], region["end_y"])
                    screen = ImageGrab.grab(bbox)
                    self.last_captured_images[i] = screen
                    
                    # Create window if it doesn't exist
                    if self.screen_windows[i] is None:
                        tk_image = ImageTk.PhotoImage(screen)
                        self._create_screen_window(i, tk_image)
                        continue
                    
                    # Update the display
                    self._update_display_realtime(i, screen)
                
                time.sleep(0.016)  # ~60 FPS for smooth updates
                
        except Exception as e:
            print(f"Error during screen capture: {e}")
            self.status_label.config(text=f"Capture error: {e}")

    def _update_display_realtime(self, region_idx, screen_image):
        """Update display with high-quality scaling"""
        try:
            window = self.screen_windows[region_idx]
            if not window or not window.winfo_exists():
                return
                
            current_width = window.winfo_width()
            current_height = window.winfo_height()
            
            # Calculate scaling while maintaining aspect ratio
            orig_ratio = screen_image.width / screen_image.height
            new_height = int(current_width / orig_ratio)
            
            if new_height > current_height:
                new_width = int(current_height * orig_ratio)
                new_height = current_height
            else:
                new_width = current_width
            
            # Use high-quality resampling based on scale factor
            scale_factor = new_width / screen_image.width
            
            if scale_factor > 2.0:
                # Significant enlargement - use LANCZOS for best quality
                resample = Image.LANCZOS
            elif scale_factor > 1.0:
                # Moderate enlargement - use BICUBIC for smooth scaling
                resample = Image.BICUBIC
            elif scale_factor < 0.5:
                # Significant reduction - use LANCZOS to avoid aliasing
                resample = Image.LANCZOS
            else:
                # Near original size - use BILINEAR for speed
                resample = Image.BILINEAR
            
            # High-quality resize
            resized_img = screen_image.resize((new_width, new_height), resample)
            
            # For very large enlargements, apply additional sharpening
            if scale_factor > 2.0:
                from PIL import ImageFilter, ImageEnhance
                # Slight sharpening to counteract softness from enlargement
                resized_img = resized_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            tk_image = ImageTk.PhotoImage(resized_img)
            
            # Update display
            if self.screen_labels[region_idx]:
                self.screen_labels[region_idx].config(image=tk_image)
                self.screen_labels[region_idx].image = tk_image
                    
        except Exception as e:
            print(f"Display update error: {e}")

    def _create_screen_window(self, region_idx, tk_image):
        """Create borderless window with resize and drag functionality"""
        window = tk.Toplevel(self.root)
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_screen_window(region_idx))
        
        # Remove title bar but keep functionality
        window.overrideredirect(True)
        
        # Store original dimensions
        region = self.capture_regions[region_idx]
        width = region["end_x"] - region["start_x"]
        height = region["end_y"] - region["start_y"]
        
        # Set initial size and position
        window.geometry(f"{width}x{height}+100+100")
        window.attributes('-topmost', True)
        
        # Create main frame
        main_frame = tk.Frame(window, bg='black')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create image label
        self.screen_labels[region_idx] = tk.Label(main_frame, image=tk_image, bg='black')
        self.screen_labels[region_idx].image = tk_image
        self.screen_labels[region_idx].pack(fill=tk.BOTH, expand=True)
        
        self.screen_windows[region_idx] = window
        
        # Add functionality AFTER window is created
        self._make_window_draggable(region_idx)
        self._add_resize_controls(region_idx)
        self._add_window_controls(region_idx)

    def _make_window_draggable(self, region_idx):
        """Make the window draggable by clicking on the image"""
        window = self.screen_windows[region_idx]
        label = self.screen_labels[region_idx]
        
        def start_drag(event):
            window.drag_data = {
                "x": event.x_root - window.winfo_x(),
                "y": event.y_root - window.winfo_y()
            }
        
        def do_drag(event):
            if hasattr(window, 'drag_data'):
                x = event.x_root - window.drag_data["x"]
                y = event.y_root - window.drag_data["y"]
                window.geometry(f"+{x}+{y}")
        
        def end_drag(event):
            if hasattr(window, 'drag_data'):
                delattr(window, 'drag_data')
        
        # Bind to the image label for dragging
        label.bind("<Button-1>", start_drag)
        label.bind("<B1-Motion>", do_drag)
        label.bind("<ButtonRelease-1>", end_drag)

    def _add_resize_controls(self, region_idx):
        """Add invisible resize handles to window edges"""
        window = self.screen_windows[region_idx]
        handle_width = 5
        
        # Create invisible resize areas
        resize_areas = {
            'n': tk.Frame(window, cursor='sb_v_double_arrow', bg=''),
            's': tk.Frame(window, cursor='sb_v_double_arrow', bg=''),
            'w': tk.Frame(window, cursor='sb_h_double_arrow', bg=''),
            'e': tk.Frame(window, cursor='sb_h_double_arrow', bg=''),
            'nw': tk.Frame(window, cursor='size_nw_se', bg=''),
            'ne': tk.Frame(window, cursor='size_ne_sw', bg=''),
            'sw': tk.Frame(window, cursor='size_ne_sw', bg=''),
            'se': tk.Frame(window, cursor='size_nw_se', bg='')
        }
        
        # Position resize areas
        resize_areas['n'].place(x=0, y=0, relwidth=1.0, height=handle_width)
        resize_areas['s'].place(x=0, rely=1.0, y=-handle_width, relwidth=1.0, height=handle_width)
        resize_areas['w'].place(x=0, y=0, width=handle_width, relheight=1.0)
        resize_areas['e'].place(relx=1.0, x=-handle_width, y=0, width=handle_width, relheight=1.0)
        
        # Corners
        corner_size = handle_width * 2
        resize_areas['nw'].place(x=0, y=0, width=corner_size, height=corner_size)
        resize_areas['ne'].place(relx=1.0, x=-corner_size, y=0, width=corner_size, height=corner_size)
        resize_areas['sw'].place(x=0, rely=1.0, y=-corner_size, width=corner_size, height=corner_size)
        resize_areas['se'].place(relx=1.0, x=-corner_size, rely=1.0, y=-corner_size, width=corner_size, height=corner_size)
        
        # Bind events to each area
        for edge, frame in resize_areas.items():
            frame.bind("<Button-1>", lambda e, edge=edge: self._start_resize(region_idx, e, edge))
            frame.bind("<B1-Motion>", lambda e, edge=edge: self._do_resize(region_idx, e, edge))
            frame.bind("<ButtonRelease-1>", lambda e: self._end_resize(region_idx))

    def _start_resize(self, region_idx, event, edge):
        """Start resize operation"""
        window = self.screen_windows[region_idx]
        window.resize_data = {
            "startx": event.x_root,
            "starty": event.y_root,
            "x": window.winfo_x(),
            "y": window.winfo_y(),
            "width": window.winfo_width(),
            "height": window.winfo_height(),
            "edge": edge
        }

    def _do_resize(self, region_idx, event, edge):
        """Perform the resize operation"""
        window = self.screen_windows[region_idx]
        if not hasattr(window, 'resize_data'):
            return
        
        data = window.resize_data
        dx = event.x_root - data["startx"]
        dy = event.y_root - data["starty"]
        
        x, y = data["x"], data["y"]
        width, height = data["width"], data["height"]
        
        # Calculate new dimensions based on edge
        if 'n' in edge:
            y += dy
            height -= dy
        if 's' in edge:
            height += dy
        if 'w' in edge:
            x += dx
            width -= dx
        if 'e' in edge:
            width += dx
        
        # Minimum size constraints
        min_width, min_height = 200, 150
        if width < min_width:
            if 'w' in edge:
                x -= (min_width - width)
            width = min_width
        if height < min_height:
            if 'n' in edge:
                y -= (min_height - height)
            height = min_height
        
        # Apply new geometry
        window.geometry(f"{width}x{height}+{x}+{y}")

    def _end_resize(self, region_idx):
        """End resize operation"""
        window = self.screen_windows[region_idx]
        if hasattr(window, 'resize_data'):
            delattr(window, 'resize_data')

    def _add_window_controls(self, region_idx):
        """Add close button since there's no title bar"""
        window = self.screen_windows[region_idx]

    def _close_screen_window(self, region_idx):
        """Close a specific screen window"""
        if self.screen_windows[region_idx]:
            self.screen_windows[region_idx].destroy()
            self.screen_windows[region_idx] = None
            self.screen_labels[region_idx] = None

    def start_square_mapping(self, square_idx):
        """Start 3D square mapping for a specific square"""
        if self.mapping_active:
            return

        self.current_square = square_idx
        self.mapping_active = True
        
        # Reset square data
        square_data = self.square_data[square_idx]
        square_data["points_2d"] = []
        square_data["points_3d"] = []
        square_data["grid_points_2d"] = []
        square_data["grid_points_3d"] = []
        square_data["homography_matrix"] = None
        square_data["nearest_grid_point"] = None
        
        self.camera_thread = threading.Thread(target=self.camera_mapping_loop, daemon=True)
        self.camera_thread.start()
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text=f"Enhanced 3D mapping started for square {square_idx + 1}")

    def camera_mapping_loop(self):
        """Main camera mapping loop with enhanced skeleton tracking"""
        try:
            with openni_context(Config.OPENNI_PATH) as openni:
                # Open device and streams
                dev = openni.Device.open_any()
                depth_stream = dev.create_depth_stream()
                color_stream = dev.create_color_stream()
                
                depth_stream.start()
                color_stream.start()
                
                # Setup OpenCV window
                cv2.namedWindow("Enhanced Skeleton Tracking", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("Enhanced Skeleton Tracking", self.draw_3d_square)
                
                while self.mapping_active:
                    # Capture frames
                    color_frame = color_stream.read_frame()
                    depth_frame = depth_stream.read_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                    
                    # Process color frame
                    color_data = color_frame.get_buffer_as_uint8()
                    color_image = np.frombuffer(color_data, dtype=np.uint8)
                    color_image = color_image.reshape((color_frame.height, color_frame.width, 3))
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                    
                    # Process depth frame
                    depth_data = depth_frame.get_buffer_as_uint16()
                    depth_array = np.frombuffer(depth_data, dtype=np.uint16)
                    depth_array = depth_array.reshape((depth_frame.height, depth_frame.width))
                    self.latest_depth_array = depth_array
                    
                    # Draw 3D squares and grids
                    overlay = color_image.copy()
                    overlay = self.draw_all_3d_squares_on_image(overlay)
                    color_image = cv2.addWeighted(overlay, 0.4, color_image, 0.6, 0)
                    
                    # Enhanced skeleton tracking processing
                    color_image = self._process_enhanced_skeleton_tracking(color_image, depth_array)
                    
                    # Display enhanced status information
                    self._draw_enhanced_status_info(color_image)
                    
                    # Show frame
                    cv2.imshow("Enhanced Skeleton Tracking", color_image)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self._reset_current_square()
                    elif key == ord('1'):
                        self.current_square = 0
                        print("Switched to Square 1")
                    elif key == ord('2'):
                        self.current_square = 1
                        print("Switched to Square 2")
                
                # Cleanup
                depth_stream.stop()
                color_stream.stop()
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"Error during enhanced camera mapping: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.config(text=f"Camera error: {e}")
        finally:
            self.mapping_active = False

    def _process_enhanced_skeleton_tracking(self, color_image, depth_array):
        """Enhanced skeleton tracking with full body context"""
        # Get pose data using enhanced body tracker
        color_image = self.body_tracker.find_bodies(color_image, draw=True)
        pose_data = self.body_tracker.get_hand_positions()
        
        if not pose_data.get('hands') and not pose_data.get('body'):
            # No pose detected, clean up old tracks
            self._cleanup_old_tracks()
            return color_image
        
        # Enhanced person tracking and assignment
        person_id = self._enhanced_person_matching(pose_data, color_image.shape[:2])
        
        if person_id is not None:
            # Update person state with new pose data
            person_state = self.persons[person_id]
            person_state.update_positions(pose_data)
            
            # Process each hand landmark with enhanced stability
            self._process_enhanced_hand_tracking(color_image, depth_array, person_id, person_state)
            
            # Update UI with tracking quality
            quality = person_state.get_stability_score()
            self.quality_label.config(text=f"Tracking Quality: {quality:.1%}")
        
        # Clean up old tracks
        self._cleanup_old_tracks()
        
        # Update person count
        self.person_count_label.config(text=f"Persons: {len(self.persons)}")
        
        return color_image

    def _enhanced_person_matching(self, pose_data, frame_shape):
        """Enhanced person matching using full body context"""
        frame_height, frame_width = frame_shape
        
        # Calculate person centroid using body landmarks for better stability
        body_positions = pose_data.get('body', {})
        if not body_positions:
            return None
        
        # Use torso center for more stable tracking
        torso_landmarks = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
        valid_torso_points = []
        
        for landmark_name in torso_landmarks:
            if landmark_name in body_positions:
                pos = body_positions[landmark_name]
                valid_torso_points.append((pos['x'] * frame_width, pos['y'] * frame_height))
        
        if not valid_torso_points:
            return None
        
        # Calculate torso centroid
        centroid_x = sum(p[0] for p in valid_torso_points) / len(valid_torso_points)
        centroid_y = sum(p[1] for p in valid_torso_points) / len(valid_torso_points)
        centroid = (int(centroid_x), int(centroid_y))
        
        # Find closest existing person or create new one
        closest_id = None
        min_distance = float('inf')
        max_distance = 150  # Increased threshold for body tracking
        
        for person_id, state in self.persons.items():
            if hasattr(state, 'last_centroid'):
                distance = np.sqrt((centroid[0] - state.last_centroid[0])**2 + 
                                 (centroid[1] - state.last_centroid[1])**2)
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    closest_id = person_id
        
        if closest_id is not None:
            # Update existing person
            self.persons[closest_id].last_centroid = centroid
            return closest_id
        else:
            # Create new person if we haven't reached max
            if len(self.persons) < Config.MAX_PEOPLE:
                new_id = self.next_person_id
                self.persons[new_id] = PersonState(new_id)
                self.persons[new_id].last_centroid = centroid
                self.next_person_id += 1
                return new_id
            else:
                # Reuse oldest track if we've reached max
                oldest_id = min(self.persons.keys(), key=lambda k: self.persons[k].last_seen)
                self.persons[oldest_id].last_centroid = centroid
                return oldest_id

    def _process_enhanced_hand_tracking(self, color_image, depth_array, person_id, person_state):
        """Process enhanced hand tracking with smoothing and stability"""
        frame_height, frame_width = color_image.shape[:2]
        
        # Process each hand landmark
        for hand_name, smoothed_pos in person_state.smoothed_positions.items():
            if smoothed_pos is None or smoothed_pos['visibility'] < Config.VISIBILITY_THRESHOLD:
                continue
            
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(smoothed_pos['x'] * frame_width)
            pixel_y = int(smoothed_pos['y'] * frame_height)
            
            # Enhanced visual feedback with stability indication
            stability = person_state.get_stability_score()
            color = self._get_stability_color(stability)
            
            # Draw hand landmark with stability indication
            cv2.circle(color_image, (pixel_x, pixel_y), 8, color, -1)
            cv2.circle(color_image, (pixel_x, pixel_y), 12, color, 2)
            
            # Label with person ID and hand name
            label = f"P{person_id}_{hand_name.split('_')[1]}"  # e.g., "P0_RIGHT"
            cv2.putText(color_image, label, (pixel_x + 15, pixel_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Only process touch detection if tracking is stable enough
            if stability > self.tracking_quality_threshold:
                self._process_enhanced_touch_detection(color_image, depth_array, person_id, 
                                                     hand_name, pixel_x, pixel_y, smoothed_pos)

    def _get_stability_color(self, stability):
        """Get color based on tracking stability"""
        if stability > 0.8:
            return (0, 255, 0)  # Green - very stable
        elif stability > 0.6:
            return (0, 255, 255)  # Yellow - moderately stable
        else:
            return (0, 0, 255)  # Red - unstable

    def _process_enhanced_touch_detection(self, color_image, depth_array, person_id, hand_name, 
                                        pixel_x, pixel_y, smoothed_pos):
        """Enhanced touch detection with improved accuracy"""
        if not (0 <= pixel_y < depth_array.shape[0] and 0 <= pixel_x < depth_array.shape[1]):
            return
        
        # Sample depth from multiple nearby pixels for better accuracy
        depth_val = self._get_enhanced_depth_sample(depth_array, pixel_x, pixel_y)
        if depth_val <= 0 or not np.isfinite(depth_val):
            return
        
        # Enhanced visual feedback
        cv2.putText(color_image, f"Depth: {depth_val}mm", 
                   (pixel_x + 15, pixel_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        point_3d = (pixel_x, pixel_y, depth_val)
        
        # Check all squares for intersection
        for square_idx in range(2):
            if len(self.square_data[square_idx]["points_3d"]) != 4:
                continue
            
            is_inside, mapped_pos = self.check_point_in_3d_square(point_3d, square_idx)
            if is_inside:
                self._handle_enhanced_square_interaction(
                    color_image, person_id, hand_name, square_idx, 
                    pixel_x, pixel_y, depth_val, mapped_pos
                )
                break

    def _get_enhanced_depth_sample(self, depth_array, x, y, sample_radius=2):
        """Get enhanced depth sample by averaging nearby valid pixels"""
        valid_depths = []
        
        for dy in range(-sample_radius, sample_radius + 1):
            for dx in range(-sample_radius, sample_radius + 1):
                ny, nx = y + dy, x + dx
                if (0 <= ny < depth_array.shape[0] and 0 <= nx < depth_array.shape[1]):
                    depth = depth_array[ny, nx]
                    if depth > 0 and np.isfinite(depth):
                        valid_depths.append(depth)
        
        if valid_depths:
            # Use median to reduce noise
            return np.median(valid_depths)
        else:
            return depth_array[y, x] if depth_array[y, x] > 0 else 0

    def _handle_enhanced_square_interaction(self, color_image, person_id, hand_name, square_idx,
                                          pixel_x, pixel_y, depth_val, mapped_pos):
        """Handle enhanced square interaction with improved accuracy"""
        person_state = self.persons[person_id]
        person_state.active_squares[hand_name] = square_idx
        
        # Enhanced visual feedback
        cv2.putText(color_image, f"P{person_id}_{hand_name} -> Sq{square_idx+1}", 
                   (20, 120 + 25 * len(person_state.active_squares)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        screen_x, screen_y = mapped_pos
        
        # Find nearest grid point with enhanced search
        nearest_grid_point = self.find_nearest_grid_point((pixel_x, pixel_y), square_idx)
        if not nearest_grid_point:
            return
        
        grid_x, grid_y, grid_depth = nearest_grid_point
        
        # Enhanced visual connection
        cv2.line(color_image, (pixel_x, pixel_y), (int(grid_x), int(grid_y)), (255, 0, 255), 2)
        cv2.circle(color_image, (int(grid_x), int(grid_y)), 6, (255, 0, 255), -1)
        
        # Enhanced touch detection with adaptive threshold
        depth_diff = grid_depth - depth_val
        click_threshold = Config.DEFAULT_CLICK_THRESHOLD
        
        # Display enhanced depth information
        cv2.putText(color_image, f"Surface: {int(grid_depth)}mm", 
                   (pixel_x + 15, pixel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        diff_color = (0, 255, 0) if depth_diff > 0 else (0, 0, 255)
        diff_sign = "+" if depth_diff > 0 else ""
        cv2.putText(color_image, f"Diff: {diff_sign}{int(depth_diff)}mm", 
                   (pixel_x + 15, pixel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, diff_color, 1)
        
        # Enhanced cooldown management
        if person_state.click_cooldown[hand_name] > 0:
            person_state.click_cooldown[hand_name] -= 1
        
        should_click = depth_diff <= click_threshold and person_state.click_cooldown[hand_name] <= 0
        
        # Enhanced visual feedback with stability consideration
        stability = person_state.get_stability_score()
        proximity_percentage = min(1.0, max(0.0, abs(depth_diff / click_threshold)))
        
        # Adjust visual feedback based on stability
        base_radius = 15
        stability_bonus = int(5 * stability)
        circle_radius = base_radius + stability_bonus + int(5 * proximity_percentage)
        
        proximity_color = (0, int(255 * (1-proximity_percentage)), int(255 * proximity_percentage))
        cv2.circle(color_image, (pixel_x, pixel_y), circle_radius, proximity_color, 2)
        
        # Enhanced click handling with stability requirement
        if should_click and stability > self.tracking_quality_threshold:
            pyautogui.click(screen_x, screen_y)
            cv2.circle(color_image, (pixel_x, pixel_y), circle_radius, (0, 0, 255), -1)
            person_state.click_cooldown[hand_name] = 10  # Longer cooldown for stability
            
            person_state.active_touches[hand_name] = {
                "pos": (screen_x, screen_y),
                "region": square_idx,
                "time": time.time(),
                "stability": stability
            }
            
            print(f"Enhanced click: Person {person_id} {hand_name} at ({screen_x}, {screen_y}) "
                  f"in region {square_idx+1} (stability: {stability:.2f})")
            print(f"depth_diff: {depth_diff} click threshold: {click_threshold} stability: {stability}")

    def _draw_enhanced_status_info(self, color_image):
        """Draw enhanced status information on the camera feed"""
        # Person and tracking statistics
        person_count = len(self.persons)
        total_touches = sum(len(person.active_touches) for person in self.persons.values())
        avg_stability = np.mean([person.get_stability_score() for person in self.persons.values()]) if self.persons else 0
        
        cv2.putText(color_image, f"Enhanced Tracking - Persons: {person_count}, Touches: {total_touches}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(color_image, f"Avg Stability: {avg_stability:.1%}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Current square status
        square_data = self.square_data[self.current_square]
        if len(square_data["points_3d"]) < 4:
            cv2.putText(color_image, f"Click to define 3D square {self.current_square+1}: {len(square_data['points_3d'])}/4 points", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(color_image, f"3D square {self.current_square+1} mapped! Enhanced tracking active.", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Enhanced help text
        help_texts = [
            "Enhanced Skeleton Tracking Controls:",
            "r=reset square, 1,2=switch square, q=quit"
        ]
        
        for i, text in enumerate(help_texts):
            cv2.putText(color_image, text, 
                       (20, color_image.shape[0] - 40 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Clean up old touches for all persons
        current_time = time.time()
        for person in self.persons.values():
            for hand_name in list(person.active_touches.keys()):
                if current_time - person.active_touches[hand_name]["time"] > 2.0:
                    del person.active_touches[hand_name]

    def _cleanup_old_tracks(self):
        """Remove tracks for people who haven't been seen recently"""
        current_time = time.time()
        timeout = 3.0  # Increased timeout for stability
        
        for person_id in list(self.persons.keys()):
            if current_time - self.persons[person_id].last_seen > timeout:
                del self.persons[person_id]

    def _reset_current_square(self):
        """Reset the current square data"""
        square_data = self.square_data[self.current_square]
        square_data["points_2d"] = []
        square_data["points_3d"] = []
        square_data["grid_points_2d"] = []
        square_data["grid_points_3d"] = []
        square_data["homography_matrix"] = None
        print(f"Reset square {self.current_square + 1}")

    # [Rest of the methods remain largely the same but with enhanced error handling and stability checks]
    def find_nearest_grid_point(self, hand_pos_2d, square_idx):
        """Find the nearest grid point to the hand position for a specific square"""
        square_data = self.square_data[square_idx]
        if not square_data["grid_points_2d"] or not square_data["grid_points_3d"]:
            return None
            
        hand_x, hand_y = hand_pos_2d
        nearest_point = None
        min_distance = float('inf')
        
        for i, (grid_x, grid_y) in enumerate(square_data["grid_points_2d"]):
            distance = np.sqrt((grid_x - hand_x)**2 + (grid_y - hand_y)**2)
            if distance < min_distance and distance < self.grid_search_radius:
                min_distance = distance
                nearest_point = square_data["grid_points_3d"][i]
        
        return nearest_point

    def calculate_grid_points(self, square_idx):
        """Calculate the 3D grid points inside the square (now with depth sampling)"""
        square_data = self.square_data[square_idx]
        if len(square_data["points_2d"]) != 4 or self.latest_depth_array is None:
            return
            
        square_data["grid_points_2d"] = []
        square_data["grid_points_3d"] = []
        square_data["points_3d"] = []  # Will store corner depths
        
        corners_2d = np.array(square_data["points_2d"], dtype=np.float32)
        center = np.mean(corners_2d, axis=0)
        relative_positions = corners_2d - center
        angles = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])
        sorted_indices = np.argsort(angles)
        corners_2d = corners_2d[sorted_indices]
        
        # First calculate grid points in 2D
        for i in range(self.grid_resolution + 1):
            for j in range(self.grid_resolution + 1):
                u = i / self.grid_resolution
                v = j / self.grid_resolution
                
                p1 = (1-u)*(1-v)
                p2 = u*(1-v)
                p3 = u*v
                p4 = (1-u)*v
                
                x = p1*corners_2d[0][0] + p2*corners_2d[1][0] + p3*corners_2d[2][0] + p4*corners_2d[3][0]
                y = p1*corners_2d[0][1] + p2*corners_2d[1][1] + p3*corners_2d[2][1] + p4*corners_2d[3][1]
                
                square_data["grid_points_2d"].append((x, y))
        
        # Now sample depths at all grid points
        for point in square_data["grid_points_2d"]:
            x, y = int(point[0]), int(point[1])
            if 0 <= y < self.latest_depth_array.shape[0] and 0 <= x < self.latest_depth_array.shape[1]:
                depth = self.latest_depth_array[y, x]
                if depth > 0:
                    square_data["grid_points_3d"].append((x, y, depth))
                else:
                    # If invalid depth, use nearest neighbor
                    depth = self._get_nearest_valid_depth(x, y)
                    square_data["grid_points_3d"].append((x, y, depth))
        
        # Calculate corner depths by averaging nearby grid points
        self._calculate_corner_depths(square_idx)
        
        print(f"Generated {len(square_data['grid_points_2d'])} grid points for square {square_idx+1}")

    def _get_nearest_valid_depth(self, x, y, search_radius=5):
        """Find nearest valid depth value within search radius"""
        if self.latest_depth_array is None:
            return 0
            
        min_dist = float('inf')
        best_depth = 0
        
        for dy in range(-search_radius, search_radius+1):
            for dx in range(-search_radius, search_radius+1):
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.latest_depth_array.shape[0] and 
                    0 <= nx < self.latest_depth_array.shape[1]):
                    depth = self.latest_depth_array[ny, nx]
                    if depth > 0:
                        dist = dx*dx + dy*dy
                        if dist < min_dist:
                            min_dist = dist
                            best_depth = depth
        return best_depth

    def _calculate_corner_depths(self, square_idx):
        """Calculate corner depths by averaging nearby grid points"""
        square_data = self.square_data[square_idx]
        if len(square_data["grid_points_3d"]) == 0:
            return
            
        # Get the 4 corner grid points (first, last, and mid points of first/last rows)
        grid_size = self.grid_resolution + 1
        corners = [
            0,  # First point of first row
            grid_size - 1,  # Last point of first row
            (grid_size * grid_size) - grid_size,  # First point of last row
            (grid_size * grid_size) - 1  # Last point of last row
        ]
        
        # Average the depths of these corner grid points to get the square corner depths
        for i, corner_idx in enumerate(corners):
            if corner_idx < len(square_data["grid_points_3d"]):
                x, y = square_data["points_2d"][i]
                # Get the depth from the corresponding grid point
                depth = square_data["grid_points_3d"][corner_idx][2]
                square_data["points_3d"].append((x, y, depth))

    def draw_3d_square(self, event, x, y, flags, param):
        """Handle clicks for defining 3D square corners (now just 2D initially)"""
        if event == cv2.EVENT_LBUTTONDOWN and self.mapping_active:
            square_data = self.square_data[self.current_square]
            if len(square_data["points_2d"]) < 4:
                square_data["points_2d"].append((x, y))
                print(f"Added square {self.current_square+1} point {len(square_data['points_2d'])}: ({x}, {y})")
                
                if len(square_data["points_2d"]) == 4:
                    self.compute_perspective_transform(self.current_square)
                    self.calculate_grid_points(self.current_square)

    def compute_perspective_transform(self, square_idx):
        """Compute homography matrix for perspective transformation"""
        square_data = self.square_data[square_idx]
        region = self.capture_regions[square_idx]
        
        if len(square_data["points_2d"]) != 4 or not region["selection_made"]:
            return
            
        src_pts = np.array(square_data["points_2d"], dtype=np.float32)
        dst_pts = np.array([
            [region["start_x"], region["start_y"]],
            [region["end_x"], region["start_y"]],
            [region["end_x"], region["end_y"]],
            [region["start_x"], region["end_y"]]
        ], dtype=np.float32)
        
        square_data["homography_matrix"] = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print(f"Perspective transform computed for square {square_idx+1}!")

    def check_point_in_3d_square(self, point_3d, square_idx):
        """Check if a 3D point is inside the mapped 3D square"""
        square_data = self.square_data[square_idx]
        region = self.capture_regions[square_idx]
        
        if len(square_data["points_3d"]) != 4 or square_data["homography_matrix"] is None:
            return False, (0, 0)
            
        x, y, z = point_3d
        
        # Depth check
        nearest_grid = self.find_nearest_grid_point((x, y), square_idx)
        if nearest_grid:
            grid_x, grid_y, grid_z = nearest_grid
            if abs(z - grid_z) > 100:  # 100mm threshold
                return False, (0, 0)
        
        # Transform point
        point_2d = np.array([[x, y]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point_2d.reshape(-1, 1, 2), square_data["homography_matrix"])
        screen_x, screen_y = transformed_point[0][0]
        
        # Check bounds
        if (region["start_x"] <= screen_x <= region["end_x"] and 
            region["start_y"] <= screen_y <= region["end_y"]):
            return True, (screen_x, screen_y)
        
        return False, (0, 0)

    def get_depth_color(self, depth_value, min_depth=700, max_depth=1500):
        """Get color based on depth value"""
        normalized = max(0, min(1, (depth_value - min_depth) / (max_depth - min_depth)))
        
        if normalized < 0.5:
            r = int(255 * (2 * normalized))
            g = 255
            b = 0
        else:
            r = 255
            g = int(255 * (2 - 2 * normalized))
            b = 0
            
        return (b, g, r)

    def draw_all_3d_squares_on_image(self, image):
        """Draw all defined squares and grids on the image"""
        for square_idx in range(2):
            square_data = self.square_data[square_idx]
            color = Config.SQUARE_COLORS[square_idx]
            
            # Draw corner points (now only 2D until depths are calculated)
            for i, point in enumerate(square_data["points_2d"]):
                x, y = point
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
                if i < len(square_data["points_3d"]):
                    z = square_data["points_3d"][i][2]
                    cv2.putText(image, f"S{square_idx+1}P{i+1}: {int(z)}mm", 
                            (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    cv2.putText(image, f"S{square_idx+1}P{i+1}", 
                            (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw complete square and grid
            if len(square_data["points_2d"]) == 4:
                pts = np.array(square_data["points_2d"], np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
                
                centroid = np.mean(pts, axis=0)[0].astype(int)
                cv2.putText(image, f"Square {square_idx+1}", 
                        (centroid[0]-30, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw grid (now with sampled depths)
                if len(square_data["grid_points_3d"]) > 0:
                    self._draw_grid(image, square_data)

        return image

    def _draw_grid(self, image, square_data):
        """Draw grid lines on the image"""
        grid_points = np.array(square_data["grid_points_2d"], dtype=np.int32)
        depths = [point[2] for point in square_data["grid_points_3d"]]
        min_depth, max_depth = min(depths), max(depths)
        
        # Draw horizontal lines
        for i in range(self.grid_resolution + 1):
            row_points = grid_points[i*(self.grid_resolution+1):(i+1)*(self.grid_resolution+1)]
            row_depths = [square_data["grid_points_3d"][i*(self.grid_resolution+1)+j][2] 
                         for j in range(self.grid_resolution+1)]
            
            for j in range(len(row_points)-1):
                pt1 = tuple(np.int32(row_points[j]))
                pt2 = tuple(np.int32(row_points[j+1]))
                avg_depth = (row_depths[j] + row_depths[j+1]) / 2
                line_color = self.get_depth_color(avg_depth, min_depth, max_depth)
                cv2.line(image, pt1, pt2, line_color, 1)
        
        # Draw vertical lines
        for j in range(self.grid_resolution + 1):
            col_points = [grid_points[i*(self.grid_resolution+1)+j] for i in range(self.grid_resolution+1)]
            col_depths = [square_data["grid_points_3d"][i*(self.grid_resolution+1)+j][2] 
                         for i in range(self.grid_resolution+1)]
            
            for i in range(len(col_points)-1):
                pt1 = tuple(np.int32(col_points[i]))
                pt2 = tuple(np.int32(col_points[i+1]))
                avg_depth = (col_depths[i] + col_depths[i+1]) / 2
                line_color = self.get_depth_color(avg_depth, min_depth, max_depth)
                cv2.line(image, pt1, pt2, line_color, 1)

    def stop_all(self):
        """Stop all operations and cleanup"""
        self.capturing = False
        self.mapping_active = False
        self.calibration_mode = False
        
        # Close screen windows
        for i in range(2):
            self._close_screen_window(i)
        
        # Close selection window
        if self.selection_window:
            self.selection_window.destroy()
            self.selection_window = None
                
        cv2.destroyAllWindows()
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Enhanced tracking stopped")
        print("Stopped all enhanced tracking operations.")

    def on_closing(self):
        """Handle application closing"""
        self.stop_all()
        if hasattr(self, 'body_tracker'):
            self.body_tracker.close()
        self.root.destroy()


@contextmanager
def openni_context(openni_path):
    """Context manager for OpenNI2 initialization and cleanup"""
    try:
        openni2.initialize(openni_path)
        yield openni2
    except Exception as e:
        print(f"Error initializing OpenNI2: {e}")
        raise
    finally:
        try:
            openni2.unload()
        except:
            pass


def main():
    """Main application entry point"""
    try:
        root = tk.Tk()
        app = ScreenFieldControlApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Enhanced tracking application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

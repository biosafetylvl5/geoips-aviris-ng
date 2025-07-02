import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import os
import pandas as pd
from matplotlib import gridspec
import hashlib
import xarray as xr
import matplotlib.animation as animation
from PIL import Image
from IPython.display import HTML, display
import ipywidgets as widgets
from ipywidgets import interactive, Layout, VBox, HBox, Tab
import io
import base64

def plot_wavelength_image_with_spectrum_xarray(dataset, wavelength, px=None, py=None, site_name="Site",
                                       figsize=(16, 12), img_cmap='gray', arrow_position=None,
                                       arrow_color='red', band_dictionary=None, cache_dir=None,
                                       window_size=200, offset_x=0, offset_y=0, north_arrow_angle=0,
                                       image_title=None, spectrum_title=None,
                                       spectral_point_x=None, spectral_point_y=None,
                                       show_crosshair=False, crosshair_color='white',
                                       crosshair_alpha=0.5, crosshair_linewidth=1,
                                       center_x=None, center_y=None, vmin=None, vmax=None,
                                       spectrum_linestyle="", spectrum_marker='o'):
    """
    Plot imagery at the top and spectral curve at the bottom with connecting markers
    using xarray Dataset instead of GDAL.
    """
    # Start timing
    start_time = time.time()

    # Use spectral_point_x/y if provided, otherwise use px/py
    if spectral_point_x is not None and spectral_point_y is not None:
        px, py = spectral_point_x, spectral_point_y

    # Ensure px and py are provided
    if px is None or py is None:
        raise ValueError("Either (px, py) or (spectral_point_x, spectral_point_y) must be provided")

    # Get the dataset that contains the radiance data
    if isinstance(dataset, dict) and 'AVIRIS-NG-L1-RADIANCE' in dataset:
        data_ds = dataset['AVIRIS-NG-L1-RADIANCE']
    else:
        data_ds = dataset

    cached_data_loaded = False

    # If cache wasn't loaded, compute the data
    if not cached_data_loaded:
        # Get dimensions
        nrows = data_ds.dims['y']
        ncols = data_ds.dims['x']

        # Extract band information from variable names
        band_names = [var for var in data_ds.data_vars if 'nm_rad' in var]

        # Extract wavelengths from band names
        band_centers = []
        for name in band_names:
            # Extract wavelength from band name (e.g., "violet_376nm_rad" -> 376)
            parts = name.split('_')
            for part in parts:
                if 'nm' in part:
                    band_centers.append(float(part.replace('nm', '')))
                    break

        # Find the band closest to the requested wavelength
        closest_band_idx = np.argmin(np.abs(np.array(band_centers) - wavelength))
        closest_band_name = band_names[closest_band_idx]
        actual_wavelength = band_centers[closest_band_idx]

        # Get the full spectrum for the selected point
        spectrum = []
        for band_name in band_names:
            pixel_data = data_ds[band_name].isel(x=px, y=py).values
            spectrum.append(float(pixel_data))

        # Calculate the window coordinates with offset
        # Determine the center point for the window
        if center_x is not None and center_y is not None:
            # Use the specified center coordinates
            center_x_with_offset = center_x + offset_x
            center_y_with_offset = center_y + offset_y
        else:
            # Use the default center (middle of the image) with offset
            center_x_with_offset = (ncols // 2) + offset_x
            center_y_with_offset = (nrows // 2) + offset_y

        # Calculate window boundaries
        half_window = window_size // 2
        x_start = max(0, center_x_with_offset - half_window)
        x_end = min(ncols, center_x_with_offset + half_window)
        y_start = max(0, center_y_with_offset - half_window)
        y_end = min(nrows, center_y_with_offset + half_window)

        # Read only the window of interest for the selected band
        band_data = get_band_data_xarray(data_ds, closest_band_name, x_start, y_start, x_end, y_end)

        # Get coordinate values for the window extent
        x_coords = data_ds.x.values
        y_coords = data_ds.y.values

        # Calculate extent for the window
        window_extent = [
            x_coords[x_start],
            x_coords[x_end-1],
            y_coords[y_end-1],
            y_coords[y_start]
        ]

        # Get the real-world coordinates of the pixel
        x_coord = data_ds.x.values[px]
        y_coord = data_ds.y.values[py]

    # Default band dictionary if none provided
    if band_dictionary is None:
        band_dictionary = {
            "visible-violet": {'lower': 375, 'upper': 450, 'color': 'violet'},
            "visible-blue": {'lower': 450, 'upper': 485, 'color': 'blue'},
            "visible-cyan": {'lower': 485, 'upper': 500, 'color': 'cyan'},
            "visible-green": {'lower': 500, 'upper': 565, 'color': 'green'},
            "visible-yellow": {'lower': 565, 'upper': 590, 'color': 'yellow'},
            "visible-orange": {'lower': 590, 'upper': 625, 'color': 'orange'},
            "visible-red": {'lower': 625, 'upper': 740, 'color': 'red'},
            "near-infrared": {'lower': 740, 'upper': 1100, 'color': 'gray'},
            "shortwave-infrared": {'lower': 1100, 'upper': 2500, 'color': 'white'}
        }

    # Create a DataFrame for the spectrum
    spectrum_df = pd.DataFrame({
        "Band name": band_names,
        "Band center (nm)": band_centers,
        f"{site_name} radiance": spectrum
    })

    # Create the figure - this part is always executed (not cached)
    fig = plt.figure(figsize=figsize)

    # Create GridSpec with different height ratios
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # Title font dictionary
    titlefont = {'fontsize': 16, 'fontweight': 2,
                 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}

    # Top subplot for the image
    ax_image = fig.add_subplot(gs[0])

    # Plot the image with vmin and vmax if provided
    im = ax_image.imshow(band_data, cmap=img_cmap, extent=window_extent, vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_image)
    cbar.set_label('Radiance', fontsize=12)

    # Check if the point is within the window
    point_in_window = (x_start <= px < x_end) and (y_start <= py < y_end)

    # Mark the selected pixel on the image if it's within the window
    if point_in_window:
        ax_image.plot(x_coord, y_coord, 'ro', markersize=10, markeredgecolor='white')

    # Add crosshair if requested - do this regardless of whether point is in window
    if show_crosshair and point_in_window:
        # Draw horizontal line across the full width of the displayed image
        ax_image.axhline(y=y_coord, color=crosshair_color, alpha=crosshair_alpha,
                       linewidth=crosshair_linewidth, zorder=1)

        # Draw vertical line across the full height of the displayed image
        ax_image.axvline(x=x_coord, color=crosshair_color, alpha=crosshair_alpha,
                       linewidth=crosshair_linewidth, zorder=1)
    elif show_crosshair and not point_in_window:
        print("Point is outside the current window. Crosshair not drawn.")

    # Image plot configuration
    # Use custom title if provided, otherwise use default
    if image_title is None:
        image_title = f"Imagery at {actual_wavelength:.2f} nm (Band {closest_band_name})"
    ax_image.set_title(image_title, fontdict=titlefont, pad=15)

    ax_image.set_xlabel('Easting (m)', fontsize=14)
    ax_image.set_ylabel('Northing (m)', fontsize=14)
    ax_image.tick_params(axis='both', which='major', labelsize=12)
    ax_image.grid(alpha=0.3, linestyle='--')

    # Add north arrow
    if arrow_position is None:
        # Default position in the upper left if not specified
        arrow_position = (window_extent[0] + 0.9 * (window_extent[1] - window_extent[0]),
                          window_extent[3] - 0.2 * (window_extent[3] - window_extent[2]))

    arrow_length = 0.1 * (window_extent[3] - window_extent[2])  # 10% of the y-axis length

    # Calculate arrow components based on the angle
    import math
    angle_rad = math.radians(north_arrow_angle)
    dx = arrow_length * math.sin(angle_rad)
    dy = arrow_length * math.cos(angle_rad)

    # Draw the arrow with the specified angle
    ax_image.arrow(arrow_position[0], arrow_position[1], dx, dy,
                  head_width=arrow_length/5, head_length=arrow_length/3,
                  fc=arrow_color, ec=arrow_color, linewidth=3)

    # Add "North" text next to the arrow
    text_x = arrow_position[0] + dx * 1.6
    text_y = arrow_position[1] + dy * 1.6
    ax_image.text(text_x, text_y, 'N', fontsize=14, ha='center', color=arrow_color, fontweight='bold')

    # Bottom subplot for the spectrum
    ax_spectrum = fig.add_subplot(gs[1])

    # Plot the spectrum
    spectrum_df.plot(x='Band center (nm)', y=f"{site_name} radiance",
                    ax=ax_spectrum, c='black', label='_nolegend_', legend=False,
                    marker=spectrum_marker, linestyle=spectrum_linestyle)

    # Add shaders for band regions
    for region, limits in band_dictionary.items():
        ax_spectrum.axvspan(limits['lower'], limits['upper'], alpha=0.2,
                           color=limits['color'], label=region)

    # Add water vapor region shaders
    ax_spectrum.axvspan(1340, 1445, alpha=0.1, color='blue', label='water vapor regions')
    ax_spectrum.axvspan(1790, 1955, alpha=0.1, color='blue')

    # Add vertical line at the selected wavelength
    ax_spectrum.axvline(x=actual_wavelength, color='red', linestyle='--', linewidth=2,
                       label=f'Selected wavelength: {actual_wavelength:.2f} nm')

    # Mark the radiance value at the selected wavelength
    radiance_at_wavelength = spectrum[closest_band_idx]
    ax_spectrum.plot(actual_wavelength, radiance_at_wavelength, 'ro', markersize=8)

    # Spectrum plot configuration
    ax_spectrum.set_xlim(min(band_centers), max(band_centers))
    ax_spectrum.set_ylabel("Radiance", fontsize=16)
    ax_spectrum.set_xlabel("Wavelength (nm)", fontsize=16)
    ax_spectrum.tick_params(axis='both', which='major', labelsize=14)
    ax_spectrum.grid('on', alpha=0.25)

    # Use custom title if provided, otherwise use default
    if spectrum_title is None:
        spectrum_title = f"Spectral Profile at {site_name} (x={px}, y={py})"
    ax_spectrum.set_title(spectrum_title, fontdict=titlefont, pad=10)

    # Add legend to spectrum plot
    legend = ax_spectrum.legend(prop={'size': 12}, loc='center left',
                               bbox_to_anchor=(1.01, 0.5), ncol=1, framealpha=1)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    return fig

def create_html_animation(frames, frame_duration=500, repeat=True, title="Wavelength Animation"):
    """
    Create an interactive HTML animation with speed control, wavelength selection, and frame navigation.

    Parameters:
    -----------
    frames : list
        List of PIL Image objects representing animation frames
    frame_duration : int
        Default duration between frames in milliseconds
    repeat : bool
        Whether to repeat the animation
    title : str
        Title for the animation
    """
    # Convert frames to base64 encoded images
    frame_data = []
    for i, frame in enumerate(frames):
        buffer = io.BytesIO()
        frame.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        frame_data.append(f"data:image/png;base64,{img_str}")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .container {{
                max-width: 1200px;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .image-container {{
                text-align: center;
                margin: 20px 0;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                background-color: #fafafa;
            }}
            #animationImage {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .controls {{
                margin: 20px 0;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }}
            .control-section {{
                margin: 15px 0;
                padding: 10px;
                border-left: 4px solid #007bff;
                background-color: white;
                border-radius: 0 5px 5px 0;
            }}
            .control-section h3 {{
                margin: 0 0 10px 0;
                color: #495057;
                font-size: 16px;
            }}
            .control-group {{
                margin: 8px 0;
                display: flex;
                align-items: center;
                gap: 10px;
                flex-wrap: wrap;
            }}
            .control-group label {{
                font-weight: bold;
                min-width: 120px;
                color: #495057;
            }}
            button {{
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                margin: 2px;
                transition: background-color 0.3s;
            }}
            button:hover {{
                background-color: #0056b3;
            }}
            button:disabled {{
                background-color: #6c757d;
                cursor: not-allowed;
            }}
            button.active {{
                background-color: #28a745;
            }}
            input[type="range"] {{
                flex: 1;
                min-width: 200px;
                margin: 0 10px;
            }}
            input[type="number"] {{
                width: 80px;
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }}
            select {{
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: white;
            }}
            .info-display {{
                display: flex;
                gap: 20px;
                margin: 10px 0;
                flex-wrap: wrap;
            }}
            .info-item {{
                background-color: #e9ecef;
                padding: 8px 12px;
                border-radius: 5px;
                font-weight: bold;
                color: #495057;
            }}
            .frame-info {{
                color: #007bff;
            }}
            .wavelength-info {{
                color: #28a745;
            }}
            .speed-info {{
                color: #fd7e14;
            }}
            .keyboard-shortcuts {{
                margin-top: 15px;
                padding: 10px;
                background-color: #f1f3f4;
                border-radius: 5px;
                font-size: 12px;
                color: #6c757d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center; color: #495057;">{title}</h1>

            <div class="image-container">
                <img id="animationImage" src="{frame_data[0]}" alt="Animation Frame">
            </div>

            <div class="info-display">
                <div class="info-item frame-info">
                    Frame: <span id="frameDisplay">1 / {len(frames)}</span>
                </div>
                <div class="info-item wavelength-info">
                    Wavelength: <span id="wavelengthDisplay">N/A</span>
                </div>
                <div class="info-item speed-info">
                    Speed: <span id="speedDisplay">{frame_duration}ms</span>
                </div>
            </div>

            <div class="controls">
                <div class="control-section">
                    <h3>Controls</h3>
                    <div class="control-group">
                        <button id="playBtn" onclick="togglePlay()">▶ Play</button>
                        <button id="pauseBtn" onclick="pause()" style="display:none;">⏸ Pause</button>
                        <button onclick="reset()">⏮ Reset</button>
                        <button onclick="goToEnd()">⏭ End</button>
                        <label style="margin-left: 20px;">
                            <input type="checkbox" id="loopCheckbox" checked="{str(repeat).lower()}" onchange="toggleLoop()">
                            Loop Animation
                        </label>
                    </div>
                </div>

                <div class="control-section">
                    <h3>Frame Control</h3>
                    <div class="control-group">
                        <button onclick="previousFrame()">⏪ Previous</button>
                        <button onclick="nextFrame()">⏩ Next</button>
                        <button onclick="stepBackward(10)">⏪⏪ -10</button>
                        <button onclick="stepForward(10)">⏩⏩ +10</button>
                    </div>
                    <div class="control-group">
                        <label>Frame:</label>
                        <input type="range" id="frameSlider" min="1" max="{len(frames)}" value="1"
                               oninput="goToFrame(parseInt(this.value))">
                        <input type="number" id="frameInput" min="1" max="{len(frames)}" value="1"
                               onchange="goToFrame(parseInt(this.value))">
                    </div>
                </div>

                <div class="control-section">
                    <h3>Speed Control</h3>
                    <div class="control-group">
                        <label>Speed (ms):</label>
                        <input type="number" id="speedInput" value="{frame_duration}" min="10" max="10000" step="10"
                               onchange="updateSpeed()">
                        <select id="speedPreset" onchange="setSpeedPreset()">
                            <option value="">Quick Presets</option>
                            <option value="50">Very Fast (50ms)</option>
                            <option value="100">Fast (100ms)</option>
                            <option value="200">Normal (200ms)</option>
                            <option value="500">Slow-ish (500ms)</option>
                            <option value="1000">Slow (1s)</option>
                            <option value="5000">Very Slow (5s)</option>
                            <option value="10000">Ultra Slow (10s)</option>
                        </select>
                    </div>
                </div>

                <div class="control-section">
                    <h3>Jump</h3>
                    <div class="control-group">
                        <label>Jump to Frame:</label>
                        <input type="number" id="frameJumpInput" min="1" max="{len(frames)}" placeholder="Frame #">
                        <button onclick="jumpToFrameNumber()">Go to Frame</button>
                    </div>
                    <div class="control-group">
                        <label>Quick Jumps:</label>
                        <button onclick="jumpToPercent(0)">Start (0%)</button>
                        <button onclick="jumpToPercent(25)">25%</button>
                        <button onclick="jumpToPercent(50)">50%</button>
                        <button onclick="jumpToPercent(75)">75%</button>
                        <button onclick="jumpToPercent(100)">End (100%)</button>
                    </div>
                </div>

                <div class="keyboard-shortcuts">
                    <strong>Keyboard Shortcuts:</strong>
                    Spacebar: Play/Pause | ← →: Previous/Next Frame | Home: Reset | End: Go to End |
                    ↑ ↓: Speed Up/Down | 1-5: Jump to 20%, 40%, 60%, 80%, 100%
                </div>
            </div>
        </div>

        <script>
            // Animation data and state
            const frames = {frame_data};
            const totalFrames = frames.length;
            let currentFrame = 0;
            let isPlaying = false;
            let animationInterval = null;
            let animationSpeed = {frame_duration};
            let loopAnimation = {str(repeat).lower()};

            // DOM elements
            const imageElement = document.getElementById('animationImage');
            const playBtn = document.getElementById('playBtn');
            const pauseBtn = document.getElementById('pauseBtn');
            const frameSlider = document.getElementById('frameSlider');
            const frameInput = document.getElementById('frameInput');
            const speedInput = document.getElementById('speedInput');
            const frameDisplay = document.getElementById('frameDisplay');
            const wavelengthDisplay = document.getElementById('wavelengthDisplay');
            const speedDisplay = document.getElementById('speedDisplay');
            const loopCheckbox = document.getElementById('loopCheckbox');

            // Initialize display
            updateDisplay();

            // Core animation functions
            function updateFrame(frameIndex) {{
                if (frameIndex < 0 || frameIndex >= totalFrames) return;

                currentFrame = frameIndex;
                imageElement.src = frames[currentFrame];
                updateDisplay();
            }}

            function updateDisplay() {{
                const frameNumber = currentFrame + 1;

                frameDisplay.textContent = `${{frameNumber}} / ${{totalFrames}}`;
                frameSlider.value = frameNumber;
                frameInput.value = frameNumber;
                speedDisplay.textContent = `${{animationSpeed}}ms`;

                // Update wavelength display if available (you can customize this)
                wavelengthDisplay.textContent = `Frame ${{frameNumber}}`;
            }}

            function togglePlay() {{
                if (isPlaying) {{
                    pause();
                }} else {{
                    play();
                }}
            }}

            function play() {{
                if (isPlaying) return;

                isPlaying = true;
                playBtn.style.display = 'none';
                pauseBtn.style.display = 'inline-block';

                animationInterval = setInterval(() => {{
                    if (currentFrame >= totalFrames - 1) {{
                        if (loopAnimation) {{
                            updateFrame(0);
                        }} else {{
                            pause();
                        }}
                        return;
                    }}
                    updateFrame(currentFrame + 1);
                }}, animationSpeed);
            }}

            function pause() {{
                isPlaying = false;
                playBtn.style.display = 'inline-block';
                pauseBtn.style.display = 'none';

                if (animationInterval) {{
                    clearInterval(animationInterval);
                    animationInterval = null;
                }}
            }}

            function reset() {{
                pause();
                updateFrame(0);
            }}

            function goToEnd() {{
                pause();
                updateFrame(totalFrames - 1);
            }}

            function nextFrame() {{
                pause();
                if (currentFrame < totalFrames - 1) {{
                    updateFrame(currentFrame + 1);
                }} else if (loopAnimation) {{
                    updateFrame(0);
                }}
            }}

            function previousFrame() {{
                pause();
                if (currentFrame > 0) {{
                    updateFrame(currentFrame - 1);
                }} else if (loopAnimation) {{
                    updateFrame(totalFrames - 1);
                }}
            }}

            function stepForward(steps) {{
                pause();
                const newFrame = Math.min(currentFrame + steps, totalFrames - 1);
                updateFrame(newFrame);
            }}

            function stepBackward(steps) {{
                pause();
                const newFrame = Math.max(currentFrame - steps, 0);
                updateFrame(newFrame);
            }}

            function goToFrame(frameNumber) {{
                pause();
                const frameIndex = Math.max(1, Math.min(frameNumber, totalFrames)) - 1;
                updateFrame(frameIndex);
            }}

            function jumpToFrameNumber() {{
                const frameNumber = parseInt(document.getElementById('frameJumpInput').value);
                if (frameNumber >= 1 && frameNumber <= totalFrames) {{
                    goToFrame(frameNumber);
                    document.getElementById('frameJumpInput').value = '';
                }}
            }}

            function jumpToPercent(percent) {{
                pause();
                const frameIndex = Math.round((percent / 100) * (totalFrames - 1));
                updateFrame(frameIndex);
            }}

            // Speed control functions
            function updateSpeed() {{
                const newSpeed = parseInt(speedInput.value);
                if (newSpeed >= 10 && newSpeed <= 10000) {{
                    animationSpeed = newSpeed;
                    updateDisplay();
                    if (isPlaying) {{
                        pause();
                        play();
                    }}
                }}
            }}

            function setSpeedPreset() {{
                const preset = document.getElementById('speedPreset').value;
                if (preset) {{
                    speedInput.value = preset;
                    updateSpeed();
                    document.getElementById('speedPreset').value = '';
                }}
            }}

            function toggleLoop() {{
                loopAnimation = loopCheckbox.checked;
            }}

            // Keyboard shortcuts
            document.addEventListener('keydown', function(event) {{
                // Prevent default behavior for our handled keys
                const handledKeys = ['Space', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5'];
                if (handledKeys.includes(event.code)) {{
                    event.preventDefault();
                }}

                switch(event.code) {{
                    case 'Space':
                        togglePlay();
                        break;
                    case 'ArrowLeft':
                        previousFrame();
                        break;
                    case 'ArrowRight':
                        nextFrame();
                        break;
                    case 'ArrowUp':
                        // Speed up (decrease delay)
                        animationSpeed = Math.max(10, animationSpeed - 50);
                        speedInput.value = animationSpeed;
                        updateDisplay();
                        if (isPlaying) {{ pause(); play(); }}
                        break;
                    case 'ArrowDown':
                        // Slow down (increase delay)
                        animationSpeed = Math.min(10000, animationSpeed + 50);
                        speedInput.value = animationSpeed;
                        updateDisplay();
                        if (isPlaying) {{ pause(); play(); }}
                        break;
                    case 'Home':
                        reset();
                        break;
                    case 'End':
                        goToEnd();
                        break;
                    case 'Digit1':
                        jumpToPercent(20);
                        break;
                    case 'Digit2':
                        jumpToPercent(40);
                        break;
                    case 'Digit3':
                        jumpToPercent(60);
                        break;
                    case 'Digit4':
                        jumpToPercent(80);
                        break;
                    case 'Digit5':
                        jumpToPercent(100);
                        break;
                }}
            }});

            // Handle window focus/blur to pause/resume animation
            window.addEventListener('blur', function() {{
                if (isPlaying) {{
                    pause();
                }}
            }});

            console.log('Interactive animation loaded with', totalFrames, 'frames');
        </script>
    </body>
    </html>
    """

    return HTML(html_content)

def create_wavelength_animation(dataset, wavelengths, px=None, py=None, site_name="Site",
                               figsize=(16, 12), img_cmap='gray', arrow_position=None,
                               arrow_color='red', band_dictionary=None, cache_dir=None,
                               window_size=200, offset_x=0, offset_y=0, north_arrow_angle=0,
                               image_title_template="Imagery at {wavelength:.2f} nm",
                               spectrum_title=None, spectral_point_x=None, spectral_point_y=None,
                               show_crosshair=False, crosshair_color='white',
                               crosshair_alpha=0.5, crosshair_linewidth=1,
                               center_x=None, center_y=None, frame_duration=500,
                               repeat=True, save_path=None, save_frames=False,
                               frames_directory="animation_frames", dpi=100, vmin=None, vmax=None):
    """
    Create an animation that cycles through multiple wavelengths.
    """
    import tempfile
    import shutil

    print("Generating frames for animation...")

    # Create directories
    temp_dir = None
    if save_path:
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")

    if save_frames:
        os.makedirs(frames_directory, exist_ok=True)
        print(f"Individual frames will be saved to: {frames_directory}/")

    # Generate each frame
    frames = []
    for i, wavelength in enumerate(wavelengths):
        print(f"Generating frame {i+1}/{len(wavelengths)} - wavelength: {wavelength}nm")

        wavelength_um = wavelength / 1000

        if image_title_template:
            custom_image_title = image_title_template.format(wavelength=wavelength, wavelength_um=wavelength_um)
        else:
            custom_image_title = None

        # Create a figure for this wavelength
        fig = plot_wavelength_image_with_spectrum_xarray(
            dataset=dataset,
            wavelength=wavelength,
            px=px,
            py=py,
            site_name=site_name,
            figsize=figsize,
            img_cmap=img_cmap,
            arrow_position=arrow_position,
            arrow_color=arrow_color,
            band_dictionary=band_dictionary,
            cache_dir=cache_dir,
            window_size=window_size,
            offset_x=offset_x,
            offset_y=offset_y,
            north_arrow_angle=north_arrow_angle,
            image_title=custom_image_title,
            spectrum_title=spectrum_title,
            spectral_point_x=spectral_point_x,
            spectral_point_y=spectral_point_y,
            show_crosshair=show_crosshair,
            crosshair_color=crosshair_color,
            crosshair_alpha=crosshair_alpha,
            crosshair_linewidth=crosshair_linewidth,
            center_x=center_x,
            center_y=center_y,
            vmin=vmin,
            vmax=vmax,
        )

        # Save the figure to a buffer and load into PIL
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)

        # Load the image and immediately copy it to avoid buffer dependency
        img = Image.open(buf)
        img.load()  # Force loading of image data into memory
        img_copy = img.copy()  # Create an independent copy
        frames.append(img_copy)

        # Now it's safe to close the buffer
        buf.close()

        # Save individual frame if requested
        if save_frames:
            frame_filename = f"frame_{i+1:03d}_{wavelength:.0f}nm.png"
            frame_path = os.path.join(frames_directory, frame_filename)
            fig.savefig(frame_path, dpi=dpi, bbox_inches='tight')
            if i == 0:
                print(f"Saving frames as: {frame_filename} (and similar)")

        # If saving animation, also save to temp directory
        if save_path and temp_dir:
            temp_frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
            fig.savefig(temp_frame_path, dpi=dpi, bbox_inches='tight')

        # Close the figure to free memory
        plt.close(fig)

    # Print summary of saved frames
    if save_frames:
        print(f"✓ Saved {len(wavelengths)} individual frames to: {frames_directory}/")
        print(f"  Frame naming pattern: frame_XXX_YYYnm.png")

    # Save the animation if a path is provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        try:
            # Create GIF - frames are already loaded and copied, so they should work
            gif_path = save_path.replace('.mp4', '.gif')
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=frame_duration,
                loop=0 if repeat else 1
            )
            print(f"✓ Animation saved as GIF: {gif_path}")

            # For MP4 creation
            if temp_dir:
                try:
                    import subprocess

                    cmd = [
                        'ffmpeg',
                        '-y',
                        '-framerate', str(1000/frame_duration),
                        '-i', os.path.join(temp_dir, 'frame_%03d.png'),
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                        save_path
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✓ Animation saved as MP4: {save_path}")
                    else:
                        print(f"ffmpeg error: {result.stderr}")
                        print("GIF version is available instead.")
                except FileNotFoundError:
                    print("ffmpeg not found. Only GIF version created.")
                except Exception as e:
                    print(f"Could not create MP4 (ffmpeg error): {e}")
                    print("GIF version is available instead.")

        except Exception as e:
            print(f"Error saving animation: {e}")

        # Clean up temporary directory
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    # Create an HTML animation for display in the notebook
    print("Creating HTML animation for notebook display...")
    html_animation = create_html_animation(frames, frame_duration, repeat)

    print("✓ Animation complete!")
    return html_animation

# Modified caching function for xarray datasets
def get_band_data_xarray(dataset, band_name, x_start, y_start, x_end, y_end):
    """Function to get band data from an xarray dataset."""
    data = dataset[band_name].isel(x=slice(x_start, x_end), y=slice(y_start, y_end)).values
    data = data.astype(float)
    data[data == -9999.] = np.nan
    return data

def get_cache_key_xarray(dataset, wavelength, px, py, window_size, offset_x, offset_y):
    """Generate a unique cache key based on input parameters."""
    # Use dataset attributes to create a unique identifier
    if hasattr(dataset, 'source_file_names'):
        dataset_id = str(dataset.source_file_names)
    else:
        dataset_id = str(dataset.dims)

    key_str = f"{dataset_id}_{wavelength}_{px}_{py}_{window_size}_{offset_x}_{offset_y}"
    return hashlib.md5(key_str.encode()).hexdigest()

def create_hyperspectral_widgets(dataset, default_values=None):
    """
    Create interactive widgets for the hyperspectral plotting function with animation support.
    """
    # Set up default values (including animation defaults)
    defaults = {
        'wavelength': 500,
        'center_x': 361,
        'center_y': 3600,
        'px': 400,
        'py': 3600,
        'site_name': "Site 1",
        'window_size': 800,
        'north_arrow_angle': 352,
        'image_title': "Hyperspectral Image",
        'spectrum_title': "Spectral Profile at the marked point",
        'show_crosshair': True,
        'crosshair_color': 'red',
        'crosshair_alpha': 0.3,
        'crosshair_linewidth': 1,
        # Animation defaults
        'create_animation': False,
        'wl_start': 400,
        'wl_end': 900,
        'wl_step': 100,
        'frame_duration': 100,
        'animation_title_template': "Location 1 at {wavelength:.0f}nm ({wavelength_um:.1f}um)",
        'save_animation': False,
        'save_path': "location1_wavelength_animation_slow.mp4",
        'save_frames': False,
        'frames_directory': "animation_frames",
        'vmin': 0,
        'vmax': 1.8
    }

    if default_values:
        defaults.update(default_values)

    # Try to get coordinate ranges from dataset
    try:
        if isinstance(dataset, dict) and 'AVIRIS-NG-L1-RADIANCE' in dataset:
            data = dataset['AVIRIS-NG-L1-RADIANCE']
        else:
            data = dataset

        x_max = int(data.sizes.get('x', data.sizes.get('longitude', 1000)))
        y_max = int(data.sizes.get('y', data.sizes.get('latitude', 1000)))

        # Try to get wavelength range
        if hasattr(data, 'wavelength'):
            wl_min = float(data.wavelength.min())
            wl_max = float(data.wavelength.max())
            available_wavelengths = data.wavelength.values
        else:
            wl_min, wl_max = 400, 2500
            available_wavelengths = np.arange(400, 2500, 10)
    except:
        x_max, y_max = 1000, 1000
        wl_min, wl_max = 400, 2500
        available_wavelengths = np.arange(400, 2500, 10)

    # Basic Parameters Tab
    wavelength_widget = widgets.FloatSlider(
        value=defaults['wavelength'],
        min=wl_min,
        max=wl_max,
        step=1,
        description='Wavelength (nm):',
        style={'description_width': 'initial'},
        layout=Layout(width='500px')
    )

    site_name_widget = widgets.Text(
        value=defaults['site_name'],
        description='Site Name:',
        style={'description_width': 'initial'},
        layout=Layout(width='300px')
    )

    # Position Parameters
    center_x_widget = widgets.IntSlider(
        value=defaults['center_x'],
        min=0,
        max=x_max,
        step=1,
        description='Center X:',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    center_y_widget = widgets.IntSlider(
        value=defaults['center_y'],
        min=0,
        max=y_max,
        step=1,
        description='Center Y:',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    px_widget = widgets.IntSlider(
        value=defaults['px'],
        min=0,
        max=x_max,
        step=1,
        description='Spectrum Point X:',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    py_widget = widgets.IntSlider(
        value=defaults['py'],
        min=0,
        max=y_max,
        step=1,
        description='Spectrum Point Y:',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    window_size_widget = widgets.IntSlider(
        value=defaults['window_size'],
        min=100,
        max=2000,
        step=50,
        description='Window Size:',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    # Display Parameters
    figsize_width_widget = widgets.IntSlider(
        value=16,
        min=8,
        max=24,
        step=1,
        description='Figure Width:',
        style={'description_width': 'initial'},
        layout=Layout(width='300px')
    )

    figsize_height_widget = widgets.IntSlider(
        value=12,
        min=6,
        max=18,
        step=1,
        description='Figure Height:',
        style={'description_width': 'initial'},
        layout=Layout(width='300px')
    )

    img_cmap_widget = widgets.Dropdown(
        options=['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet', 'hot', 'cool'],
        value='gray',
        description='Colormap:',
        style={'description_width': 'initial'},
        layout=Layout(width='200px')
    )

    # Titles
    image_title_widget = widgets.Text(
        value=defaults['image_title'],
        description='Image Title:',
        style={'description_width': 'initial'},
        layout=Layout(width='500px')
    )

    spectrum_title_widget = widgets.Text(
        value=defaults['spectrum_title'],
        description='Spectrum Title:',
        style={'description_width': 'initial'},
        layout=Layout(width='500px')
    )

    # North Arrow Parameters
    north_arrow_angle_widget = widgets.FloatSlider(
        value=defaults['north_arrow_angle'],
        min=0,
        max=360,
        step=1,
        description='North Arrow Angle:',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    arrow_color_widget = widgets.Dropdown(
        options=['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple'],
        value='red',
        description='Arrow Color:',
        style={'description_width': 'initial'},
        layout=Layout(width='200px')
    )

    # Crosshair Parameters
    show_crosshair_widget = widgets.Checkbox(
        value=defaults['show_crosshair'],
        description='Show Crosshair',
        style={'description_width': 'initial'}
    )

    crosshair_color_widget = widgets.Dropdown(
        options=['red', 'white', 'blue', 'green', 'yellow', 'black', 'orange', 'purple'],
        value=defaults['crosshair_color'],
        description='Crosshair Color:',
        style={'description_width': 'initial'},
        layout=Layout(width='200px')
    )

    crosshair_alpha_widget = widgets.FloatSlider(
        value=defaults['crosshair_alpha'],
        min=0.1,
        max=1.0,
        step=0.1,
        description='Crosshair Alpha:',
        style={'description_width': 'initial'},
        layout=Layout(width='300px')
    )

    crosshair_linewidth_widget = widgets.FloatSlider(
        value=defaults['crosshair_linewidth'],
        min=0.5,
        max=5.0,
        step=0.5,
        description='Crosshair Width:',
        style={'description_width': 'initial'},
        layout=Layout(width='300px')
    )

    # Color Scale Parameters
    vmin_widget = widgets.FloatText(
        value=defaults['vmin'],
        description='V Min:',
        style={'description_width': 'initial'},
        layout=Layout(width='200px')
    )

    vmax_widget = widgets.FloatText(
        value=defaults['vmax'],
        description='V Max:',
        style={'description_width': 'initial'},
        layout=Layout(width='200px')
    )

    # Offset Parameters
    offset_x_widget = widgets.IntSlider(
        value=0,
        min=-500,
        max=500,
        step=10,
        description='Offset X:',
        style={'description_width': 'initial'},
        layout=Layout(width='300px')
    )

    offset_y_widget = widgets.IntSlider(
        value=0,
        min=-500,
        max=500,
        step=10,
        description='Offset Y:',
        style={'description_width': 'initial'},
        layout=Layout(width='300px')
    )

    # Animation Parameters
    create_animation_widget = widgets.Checkbox(
        value=defaults['create_animation'],
        description='Create Animation',
        style={'description_width': 'initial'}
    )

    # Wavelength range for animation
    wl_start_widget = widgets.FloatSlider(
        value=defaults['wl_start'],
        min=wl_min,
        max=wl_max,
        step=10,
        description='Start Wavelength:',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    wl_end_widget = widgets.FloatSlider(
        value=defaults['wl_end'],
        min=wl_min,
        max=wl_max,
        step=10,
        description='End Wavelength:',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    wl_step_widget = widgets.IntSlider(
        value=defaults['wl_step'],
        min=1,
        max=100,
        step=1,
        description='Wavelength Step:',
        style={'description_width': 'initial'},
        layout=Layout(width='300px')
    )

    frame_duration_widget = widgets.IntSlider(
        value=defaults['frame_duration'],
        min=50,
        max=2000,
        step=50,
        description='Frame Duration (ms):',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    animation_title_template_widget = widgets.Text(
        value=defaults['animation_title_template'],
        description='Animation Title Template:',
        style={'description_width': 'initial'},
        layout=Layout(width='600px')
    )

    save_animation_widget = widgets.Checkbox(
        value=defaults['save_animation'],
        description='Save Animation',
        style={'description_width': 'initial'}
    )

    save_path_widget = widgets.Text(
        value=defaults['save_path'],
        description='Save Path:',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    # Frame saving parameters
    save_frames_widget = widgets.Checkbox(
        value=defaults['save_frames'],
        description='Save Individual Frames',
        style={'description_width': 'initial'}
    )

    frames_directory_widget = widgets.Text(
        value=defaults['frames_directory'],
        description='Frames Directory:',
        style={'description_width': 'initial'},
        layout=Layout(width='400px')
    )

    # Create execute button
    execute_button = widgets.Button(
        description='Generate Plot/Animation',
        button_style='primary',
        layout=Layout(width='200px', height='40px')
    )

    # Create output widget for results
    output_widget = widgets.Output()

    # Create tabs for organization
    basic_tab = VBox([
        HBox([wavelength_widget]),
        HBox([site_name_widget]),
        HBox([image_title_widget]),
        HBox([spectrum_title_widget])
    ])

    position_tab = VBox([
        HBox([center_x_widget, center_y_widget]),
        HBox([px_widget, py_widget]),
        HBox([window_size_widget]),
        HBox([offset_x_widget, offset_y_widget])
    ])

    display_tab = VBox([
        HBox([figsize_width_widget, figsize_height_widget]),
        HBox([img_cmap_widget]),
        HBox([vmin_widget, vmax_widget])
    ])

    arrow_crosshair_tab = VBox([
        HBox([north_arrow_angle_widget, arrow_color_widget]),
        HBox([show_crosshair_widget]),
        HBox([crosshair_color_widget]),
        HBox([crosshair_alpha_widget, crosshair_linewidth_widget])
    ])

    animation_tab = VBox([
        HBox([create_animation_widget]),
        HBox([wl_start_widget, wl_end_widget]),
        HBox([wl_step_widget, frame_duration_widget]),
        HBox([animation_title_template_widget]),
        HBox([save_animation_widget, save_path_widget]),
        HBox([save_frames_widget, frames_directory_widget])
    ])

    # Create tabbed interface
    tab = Tab()
    tab.children = [basic_tab, position_tab, display_tab, arrow_crosshair_tab, animation_tab]
    tab.set_title(0, 'Basic')
    tab.set_title(1, 'Position')
    tab.set_title(2, 'Display')
    tab.set_title(3, 'Arrow & Crosshair')
    tab.set_title(4, 'Animation')

    # Define the plotting function
    def on_button_click(b):
        with output_widget:
            output_widget.clear_output()

            # Get all widget values - using proper variable names
            wavelength = wavelength_widget.value
            site_name = site_name_widget.value
            center_x = center_x_widget.value
            center_y = center_y_widget.value
            px = px_widget.value
            py = py_widget.value
            window_size = window_size_widget.value
            figsize_width = figsize_width_widget.value
            figsize_height = figsize_height_widget.value
            img_cmap = img_cmap_widget.value
            image_title = image_title_widget.value
            spectrum_title = spectrum_title_widget.value
            north_arrow_angle = north_arrow_angle_widget.value
            arrow_color = arrow_color_widget.value
            show_crosshair = show_crosshair_widget.value
            crosshair_color = crosshair_color_widget.value
            crosshair_alpha = crosshair_alpha_widget.value
            crosshair_linewidth = crosshair_linewidth_widget.value
            vmin = vmin_widget.value
            vmax = vmax_widget.value
            offset_x = offset_x_widget.value
            offset_y = offset_y_widget.value
            create_animation = create_animation_widget.value
            wl_start = wl_start_widget.value
            wl_end = wl_end_widget.value
            wl_step = wl_step_widget.value
            frame_duration = frame_duration_widget.value
            animation_title_template = animation_title_template_widget.value
            save_animation = save_animation_widget.value
            save_path = save_path_widget.value
            save_frames = save_frames_widget.value
            frames_directory = frames_directory_widget.value

            # Handle None values for vmin/vmax
            vmin = None if vmin == 0 else vmin
            vmax = None if vmax == 0 else vmax

            if create_animation:
                # Create wavelength array for animation
                wavelengths = np.arange(wl_start, wl_end + wl_step, wl_step)

                print(f"Creating animation with {len(wavelengths)} frames...")
                print(f"Wavelength range: {wl_start} - {wl_end} nm (step: {wl_step} nm)")

                if save_frames:
                    print(f"Individual frames will be saved to: {frames_directory}/")

                # Prepare save path
                animation_save_path = save_path if save_animation else None
                frames_save_dir = frames_directory if save_frames else None

                try:
                    animation = create_wavelength_animation(
                        dataset=dataset,
                        wavelengths=wavelengths,
                        center_x=center_x,
                        center_y=center_y,
                        px=px,
                        py=py,
                        site_name=site_name,
                        window_size=window_size,
                        north_arrow_angle=north_arrow_angle,
                        image_title_template=animation_title_template,
                        spectrum_title=spectrum_title,
                        show_crosshair=show_crosshair,
                        crosshair_color=crosshair_color,
                        crosshair_alpha=crosshair_alpha,
                        crosshair_linewidth=crosshair_linewidth,
                        frame_duration=frame_duration,
                        save_path=animation_save_path,
                        save_frames=save_frames,
                        frames_directory=frames_save_dir,
                        vmin=vmin,
                        vmax=vmax,
                        figsize=(figsize_width, figsize_height),
                        img_cmap=img_cmap,
                        arrow_color=arrow_color,
                        offset_x=offset_x,
                        offset_y=offset_y
                    )

                    if save_animation:
                        print(f"Animation saved to: {save_path}")

                    if save_frames:
                        print(f"Individual frames saved to: {frames_directory}/")

                    display(animation)

                except Exception as e:
                    print(f"Error creating animation: {str(e)}")
                    print("Falling back to single frame plot...")
                    create_animation = False

            if not create_animation:
                # Create single plot
                fig = plot_wavelength_image_with_spectrum_xarray(
                    dataset=dataset,
                    wavelength=wavelength,
                    center_x=center_x,
                    center_y=center_y,
                    px=px,
                    py=py,
                    site_name=site_name,
                    window_size=window_size,
                    north_arrow_angle=north_arrow_angle,
                    image_title=image_title,
                    spectrum_title=spectrum_title,
                    show_crosshair=show_crosshair,
                    crosshair_color=crosshair_color,
                    crosshair_alpha=crosshair_alpha,
                    crosshair_linewidth=crosshair_linewidth,
                    figsize=(figsize_width, figsize_height),
                    img_cmap=img_cmap,
                    arrow_color=arrow_color,
                    vmin=vmin,
                    vmax=vmax,
                    offset_x=offset_x,
                    offset_y=offset_y
                )
                plt.show()

    # Connect button to function
    execute_button.on_click(on_button_click)

    # Return the complete widget interface
    return VBox([
        tab,
        HBox([execute_button]),
        output_widget
    ])
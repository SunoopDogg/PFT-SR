import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw


class PatchSettingsGUI:
    """Patch size settings GUI (slider + input box)"""

    MIN_PATCH_SIZE = 64
    MAX_PATCH_SIZE = 512
    DEFAULT_PATCH_SIZE = 256

    def __init__(self):
        self.patch_width = self.DEFAULT_PATCH_SIZE
        self.patch_height = self.DEFAULT_PATCH_SIZE
        self.result = None
        self.root = None
        self.canvas = None
        self.image = None
        self.image_width = 0
        self.image_height = 0
        self.preview_scale = 1.0

    def show(self, image, image_width, image_height):
        """
        Show patch settings GUI

        Args:
            image: PIL Image (for preview)
            image_width: Image width
            image_height: Image height

        Returns:
            (patch_width, patch_height) or None (if cancelled)
        """
        self.image = image
        self.image_width = image_width
        self.image_height = image_height

        # Calculate preview scale
        max_preview_size = 400
        self.preview_scale = min(
            max_preview_size / image_width,
            max_preview_size / image_height,
            1.0
        )

        self.root = tk.Tk()
        self.root.title("Patch Settings")
        self.root.resizable(False, False)

        self._create_widgets()
        self._update_preview()

        self.root.mainloop()

        return self.result

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Preview canvas
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="5")
        preview_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        preview_width = int(self.image_width * self.preview_scale)
        preview_height = int(self.image_height * self.preview_scale)
        self.canvas = tk.Canvas(
            preview_frame,
            width=preview_width,
            height=preview_height,
            bg='gray'
        )
        self.canvas.pack()

        # Info label
        info_text = f"Image Size: {self.image_width} x {self.image_height}"
        ttk.Label(main_frame, text=info_text).grid(
            row=1, column=0, columnspan=2, pady=(0, 10)
        )

        # Patch Width settings
        width_frame = ttk.LabelFrame(main_frame, text="Patch Width", padding="5")
        width_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)

        self.width_var = tk.IntVar(value=self.patch_width)
        self.width_slider = ttk.Scale(
            width_frame,
            from_=self.MIN_PATCH_SIZE,
            to=self.MAX_PATCH_SIZE,
            orient="horizontal",
            variable=self.width_var,
            command=self._on_width_slider_change
        )
        self.width_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.width_entry = ttk.Entry(width_frame, width=6)
        self.width_entry.pack(side="left")
        self.width_entry.insert(0, str(self.patch_width))
        self.width_entry.bind('<Return>', self._on_width_entry_change)
        self.width_entry.bind('<FocusOut>', self._on_width_entry_change)

        # Patch Height settings
        height_frame = ttk.LabelFrame(main_frame, text="Patch Height", padding="5")
        height_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)

        self.height_var = tk.IntVar(value=self.patch_height)
        self.height_slider = ttk.Scale(
            height_frame,
            from_=self.MIN_PATCH_SIZE,
            to=self.MAX_PATCH_SIZE,
            orient="horizontal",
            variable=self.height_var,
            command=self._on_height_slider_change
        )
        self.height_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.height_entry = ttk.Entry(height_frame, width=6)
        self.height_entry.pack(side="left")
        self.height_entry.insert(0, str(self.patch_height))
        self.height_entry.bind('<Return>', self._on_height_entry_change)
        self.height_entry.bind('<FocusOut>', self._on_height_entry_change)

        # Patch info label
        self.patch_info_var = tk.StringVar()
        self.patch_info_label = ttk.Label(main_frame, textvariable=self.patch_info_var)
        self.patch_info_label.grid(row=4, column=0, columnspan=2, pady=10)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="OK", command=self._on_ok, width=10).pack(
            side="left", padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel, width=10).pack(
            side="left", padx=5
        )

        self._update_patch_info()

    def _on_width_slider_change(self, value):
        self.patch_width = int(float(value))
        self.width_entry.delete(0, tk.END)
        self.width_entry.insert(0, str(self.patch_width))
        self._update_preview()
        self._update_patch_info()

    def _on_height_slider_change(self, value):
        self.patch_height = int(float(value))
        self.height_entry.delete(0, tk.END)
        self.height_entry.insert(0, str(self.patch_height))
        self._update_preview()
        self._update_patch_info()

    def _on_width_entry_change(self, event=None):
        try:
            value = int(self.width_entry.get())
            value = max(self.MIN_PATCH_SIZE, min(self.MAX_PATCH_SIZE, value))
            self.patch_width = value
            self.width_var.set(value)
            self.width_entry.delete(0, tk.END)
            self.width_entry.insert(0, str(value))
            self._update_preview()
            self._update_patch_info()
        except ValueError:
            self.width_entry.delete(0, tk.END)
            self.width_entry.insert(0, str(self.patch_width))

    def _on_height_entry_change(self, event=None):
        try:
            value = int(self.height_entry.get())
            value = max(self.MIN_PATCH_SIZE, min(self.MAX_PATCH_SIZE, value))
            self.patch_height = value
            self.height_var.set(value)
            self.height_entry.delete(0, tk.END)
            self.height_entry.insert(0, str(value))
            self._update_preview()
            self._update_patch_info()
        except ValueError:
            self.height_entry.delete(0, tk.END)
            self.height_entry.insert(0, str(self.patch_height))

    def _update_patch_info(self):
        num_patches_w = self.image_width // self.patch_width + (
            1 if self.image_width % self.patch_width else 0
        )
        num_patches_h = self.image_height // self.patch_height + (
            1 if self.image_height % self.patch_height else 0
        )
        total_patches = num_patches_w * num_patches_h
        self.patch_info_var.set(
            f"Patches: {num_patches_w} x {num_patches_h} = {total_patches}"
        )

    def _update_preview(self):
        if self.image is None:
            return

        # Resize preview image
        preview_width = int(self.image_width * self.preview_scale)
        preview_height = int(self.image_height * self.preview_scale)
        preview_img = self.image.resize(
            (preview_width, preview_height), Image.Resampling.LANCZOS
        )

        # Draw patch grid
        draw = ImageDraw.Draw(preview_img)

        # Scaled patch size
        scaled_patch_w = self.patch_width * self.preview_scale
        scaled_patch_h = self.patch_height * self.preview_scale

        # Draw vertical lines
        x = scaled_patch_w
        while x < preview_width:
            draw.line([(x, 0), (x, preview_height)], fill='lime', width=2)
            x += scaled_patch_w

        # Draw horizontal lines
        y = scaled_patch_h
        while y < preview_height:
            draw.line([(0, y), (preview_width, y)], fill='lime', width=2)
            y += scaled_patch_h

        # Border
        draw.rectangle(
            [(0, 0), (preview_width - 1, preview_height - 1)],
            outline='lime', width=2
        )

        # Display on canvas
        self.photo = ImageTk.PhotoImage(preview_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def _on_ok(self):
        self.result = (self.patch_width, self.patch_height)
        self.root.destroy()

    def _on_cancel(self):
        self.result = None
        self.root.destroy()


def select_patch_settings(image, image_width, image_height):
    """
    Show patch settings GUI and return result

    Args:
        image: PIL Image (for preview)
        image_width: Image width
        image_height: Image height

    Returns:
        (patch_width, patch_height) or None (if cancelled)
    """
    gui = PatchSettingsGUI()
    return gui.show(image, image_width, image_height)

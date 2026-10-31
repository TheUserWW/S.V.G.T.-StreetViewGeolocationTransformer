"""Native Tk map and screenshot canvases; all Tk operations stay on the UI thread."""

import math
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

from PIL import Image, ImageTk

from streetgeo.maps.projection import project, unproject, wrap_delta
from streetgeo.maps.basemap import load_boundaries, load_shapes
from streetgeo.inference.backend import ROOT, TileCache

FONT = "Microsoft YaHei UI"
TEAL = "#087f75"


class WorldMap(tk.Canvas):
    def __init__(self, parent, *, on_click=None, on_move=None, on_view=None, cache_folder=None):
        super().__init__(parent, bg="#e6eff3", highlightthickness=0, cursor="fleur")
        self.on_click, self.on_move, self.on_view = on_click, on_move, on_view
        self.zoom, self.center = 3, project(38, -98)
        self.online = False
        self.countries = load_shapes("un_countries.geojson")
        self.boundaries = load_boundaries()
        self.states = load_shapes("us_states.geojson")
        self.label_font = tkfont.Font(root=self, family=FONT, size=8)
        self.candidates, self.controls, self.pending_point = [], [], None
        self.selected = 0
        self._drag = None
        self._scheduled = None
        self._photos, self._failed, self._needed, self._pending = {}, {}, set(), set()
        self._requests, self._results = queue.Queue(maxsize=24), queue.Queue()
        self._closed = False
        self._cache = TileCache(cache_folder)
        threading.Thread(target=self._tile_worker, daemon=True, name="map-tiles").start()
        self.bind("<Configure>", lambda event: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Motion>", self._hover)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda event: self.change_zoom(1, event.x, event.y))
        self.bind("<Button-5>", lambda event: self.change_zoom(-1, event.x, event.y))
        self.after(90, self._poll)

    @property
    def world_size(self):
        return 256 * 2 ** self.zoom

    def to_screen(self, latitude, longitude):
        x, y = project(latitude, longitude)
        return (self.winfo_width() / 2 + wrap_delta(x, self.center[0]) * self.world_size,
                self.winfo_height() / 2 + (y - self.center[1]) * self.world_size)

    def from_screen(self, x, y):
        return unproject(self.center[0] + (x - self.winfo_width() / 2) / self.world_size,
                         self.center[1] + (y - self.winfo_height() / 2) / self.world_size)

    def focus(self, latitude, longitude, zoom=None):
        self.center = project(latitude, longitude)
        if zoom is not None:
            self.zoom = max(0, min(19, int(zoom)))
        self.redraw()

    def change_zoom(self, delta, x=None, y=None):
        new_zoom = max(0, min(19, self.zoom + delta))
        if x is None:
            self.zoom = new_zoom
        else:
            latitude, longitude = self.from_screen(x, y)
            px, py = project(latitude, longitude)
            self.zoom = new_zoom
            self.center = ((px - (x - self.winfo_width() / 2) / self.world_size) % 1,
                           max(0, min(1, py - (y - self.winfo_height() / 2) / self.world_size)))
        self.redraw()

    def set_online(self, value):
        self.online = bool(value)
        self.redraw()

    def _wheel(self, event):
        if event.delta:
            self.change_zoom(1 if event.delta > 0 else -1, event.x, event.y)
        return "break"

    def _press(self, event):
        self._drag = (event.x, event.y, self.center, False)

    def _motion(self, event):
        if self._drag:
            x, y, center, moved = self._drag
            moved = moved or abs(event.x - x) + abs(event.y - y) > 4
            self._drag = (x, y, center, moved)
            if moved:
                self.center = ((center[0] - (event.x - x) / self.world_size) % 1,
                               max(0, min(1, center[1] - (event.y - y) / self.world_size)))
                self.redraw()

    def _release(self, event):
        if self._drag and not self._drag[3] and self.on_click:
            self.on_click(*self.from_screen(event.x, event.y))
        self._drag = None

    def _hover(self, event):
        if self.on_move:
            self.on_move(*self.from_screen(event.x, event.y))

    def redraw(self):
        if self._scheduled is None and not self._closed:
            self._scheduled = self.after(20, self._draw)

    def _draw_shapes(self, shapes, *, states=False):
        size, width, height = self.world_size, self.winfo_width(), self.winfo_height()
        origin_x = width / 2 - self.center[0] * size
        origin_y = height / 2 - self.center[1] * size
        for ring, (x0, y0, x1, y1), props in shapes:
            if y1 * size + origin_y < 0 or y0 * size + origin_y > height:
                continue
            first_copy = math.floor(self.center[0] - width / (2 * size)) - 1
            last_copy = math.ceil(self.center[0] + width / (2 * size)) + 1
            for offset in range(first_copy, last_copy + 1):
                if (x1 + offset) * size + origin_x < 0 or (x0 + offset) * size + origin_x > width:
                    continue
                positions = [v for px, py in ring for v in
                             (origin_x + (px + offset) * size, origin_y + py * size)]
                self.create_polygon(positions, fill="" if states else "#f8faf7",
                                    outline="#a9bdc2" if states else "", width=1,
                                    tags=("state-boundary" if states else "land",))
        if (states and self.zoom >= 4) or (not states and self.zoom <= 3):
            # Large land areas get priority; suppress overlapping labels at world scale.
            labels = {}
            for _, bounds, props in shapes:
                if not props.get("name"):
                    continue
                area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                if props["name"] not in labels or area > labels[props["name"]][0]:
                    labels[props["name"]] = (area, props)
            occupied = []
            for _, props in sorted(labels.values(), key=lambda item: item[0], reverse=True):
                lon, lat = props["label"]
                x, y = self.to_screen(lat, lon)
                half_width = self.label_font.measure(props["name"]) / 2 + 7
                box = (x - half_width, y - 12, x + half_width, y + 12)
                if not (8 < box[0] and box[2] < width - 8 and 20 < y < height - 38):
                    continue
                if any(box[0] < old[2] and box[2] > old[0] and box[1] < old[3] and box[3] > old[1] for old in occupied):
                    continue
                occupied.append(box)
                self.create_text(x, y, text=props["name"], fill="#6b808b", font=self.label_font)

    def _draw_boundaries(self):
        size, width, height = self.world_size, self.winfo_width(), self.winfo_height()
        origin_x = width / 2 - self.center[0] * size
        origin_y = height / 2 - self.center[1] * size
        first_copy = math.floor(self.center[0] - width / (2 * size)) - 1
        last_copy = math.ceil(self.center[0] + width / (2 * size)) + 1
        for line, (x0, y0, x1, y1), props in self.boundaries:
            if y1 * size + origin_y < 0 or y0 * size + origin_y > height:
                continue
            for offset in range(first_copy, last_copy + 1):
                if (x1 + offset) * size + origin_x < 0 or (x0 + offset) * size + origin_x > width:
                    continue
                positions = [v for px, py in line for v in
                             (origin_x + (px + offset) * size, origin_y + py * size)]
                self.create_line(positions, **props["style"], tags=("un-boundary", f"boundary-{props['bdytyp']}"))

    def _draw(self):
        self._scheduled = None
        self.delete("all")
        width, height, size = self.winfo_width(), self.winfo_height(), self.world_size
        if width < 2 or height < 2:
            return
        self._draw_shapes(self.countries)
        self._draw_boundaries()
        if self.zoom >= 3:
            self._draw_shapes(self.states, states=True)
        self._needed = set()
        if self.online:
            left, top = self.center[0] * size - width / 2, self.center[1] * size - height / 2
            for tx in range(math.floor(left / 256), math.floor((left + width - 1) / 256) + 1):
                for ty in range(max(0, math.floor(top / 256)), min(2 ** self.zoom, math.floor((top + height - 1) / 256) + 1)):
                    key = (self.zoom, tx % 2 ** self.zoom, ty)
                    self._needed.add(key)
                    if key in self._photos:
                        self.create_image(tx * 256 - left, ty * 256 - top, anchor="nw", image=self._photos[key])
                    elif key not in self._pending and self._failed.get(key, 0) < time.monotonic():
                        try:
                            self._requests.put_nowait(key)
                            self._pending.add(key)
                        except queue.Full:
                            pass
        for index, candidate in reversed(list(enumerate(self.candidates))):
            x, y = self.to_screen(candidate["latitude"], candidate["longitude"])
            self._pin(x, y, str(index + 1), TEAL if index == self.selected else "#748b9a", 12)
        for index, point in enumerate(self.controls):
            self._pin(*self.to_screen(point.latitude, point.longitude), chr(65 + index), "#d68826", 9)
        if self.pending_point:
            self._pin(*self.to_screen(*self.pending_point), "+", "#d68826", 10)
        footer_height = 26 if self.online else 44
        attribution = ("© OpenStreetMap contributors · 在线道路底图使用 OSM 的边界表达" if self.online
                       else "边界：UN Geospatial / OCHA · 离线概览")
        self.create_rectangle(0, height - footer_height, width, height, fill="#f8fafc", outline="")
        self.create_text(10, height - footer_height + 13, text=attribution, anchor="w", fill="#526a79", font=(FONT, 8))
        if not self.online:
            self.create_text(10, height - 12, text="来源地图边界不代表联合国对领土归属的认可。",
                             anchor="w", fill="#6b808b", font=(FONT, 8))
            if width >= 660:
                y = height - 31
                self.create_line(width - 230, y, width - 205, y, fill="#686868")
                self.create_text(width - 199, y, text="国际边界", anchor="w", fill="#526a79", font=(FONT, 8))
                self.create_line(width - 125, y, width - 100, y, fill="#999999", dash=(5, 3))
                self.create_text(width - 94, y, text="虚线边界", anchor="w", fill="#526a79", font=(FONT, 8))
        if self.online and any(key in self._failed for key in self._needed):
            self.create_text(10, 16, text="部分道路底图暂不可用，已保留离线轮廓", anchor="w", fill="#98651b", font=(FONT, 9))
        if not self.online and self.zoom >= 7:
            self.create_text(10, 16, text="离线轮廓不含道路；可启用在线道路底图", anchor="w", fill="#526a79", font=(FONT, 9))
        if self.on_view:
            self.on_view(self.zoom)

    def _pin(self, x, y, text, color, radius):
        if -radius <= x <= self.winfo_width() + radius and -radius <= y <= self.winfo_height() + radius:
            self.create_line(x - radius - 5, y, x + radius + 5, y, fill=color, width=1)
            self.create_line(x, y - radius - 5, x, y + radius + 5, fill=color, width=1)
            self.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="white", width=2)
            self.create_text(x, y, text=text, fill="white", font=(FONT, 9, "bold"))

    def _tile_worker(self):
        while not self._closed:
            key = self._requests.get()
            if key is None:
                return
            try:
                picture = self._cache.get(*key) if self.online and key in self._needed else None
                self._results.put((key, picture, None))
            except Exception as error:
                self._results.put((key, None, str(error)))

    def _poll(self):
        changed = False
        for _ in range(24):
            try:
                key, picture, error = self._results.get_nowait()
            except queue.Empty:
                break
            self._pending.discard(key)
            if error:
                self._failed[key] = time.monotonic() + 60
            elif picture is not None:
                self._photos[key] = ImageTk.PhotoImage(picture, master=self)
                self._failed.pop(key, None)
            changed = True
        if len(self._photos) > 96:
            self._photos = {key: photo for key, photo in self._photos.items() if key in self._needed}
        if changed:
            self.redraw()
        if not self._closed:
            self.after(90, self._poll)

    def close(self):
        self._closed = True
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass


class ScreenshotCanvas(tk.Canvas):
    def __init__(self, parent, *, on_click=None):
        super().__init__(parent, bg="#e9eef3", highlightthickness=0, cursor="crosshair")
        self.picture = None
        self.photo = None
        self.controls, self.target = [], None
        self.scale, self.offset = 1, (0, 0)
        self.on_click = on_click
        self.bind("<Configure>", lambda event: self.draw())
        self.bind("<Button-1>", self._click)

    def image_to_canvas(self, x, y):
        return self.offset[0] + x * self.scale, self.offset[1] + y * self.scale

    def canvas_to_image(self, x, y):
        return (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale

    def _click(self, event):
        if self.picture is not None and self.on_click:
            x, y = self.canvas_to_image(event.x, event.y)
            if 0 <= x < self.picture.width and 0 <= y < self.picture.height:
                self.on_click(x, y)

    def draw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if self.picture is None:
            self.create_text(width / 2, height / 2, text="导入游戏地图截图\n\n可用 Win + Shift + S 截取地图并保存", justify="center", fill="#728594", font=(FONT, 11))
            return
        self.scale = min(max(1, width - 20) / self.picture.width, max(1, height - 20) / self.picture.height)
        size = (max(1, round(self.picture.width * self.scale)), max(1, round(self.picture.height * self.scale)))
        self.offset = ((width - size[0]) / 2, (height - size[1]) / 2)
        self.photo = ImageTk.PhotoImage(self.picture.resize(size, Image.Resampling.LANCZOS), master=self)
        self.create_image(*self.offset, anchor="nw", image=self.photo)
        for index, point in enumerate(self.controls):
            x, y = self.image_to_canvas(point.x, point.y)
            self.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#d68826", outline="white", width=2)
            self.create_text(x + 12, y - 10, text=chr(65 + index), fill="#a85900", font=(FONT, 11, "bold"))
        if self.target is not None:
            px, py = self.target
            if 0 <= px < self.picture.width and 0 <= py < self.picture.height:
                x, y = self.image_to_canvas(px, py)
                self.create_oval(x - 12, y - 12, x + 12, y + 12, outline="#e33955", width=3)
                self.create_line(x - 22, y, x + 22, y, fill="#e33955", width=2)
                self.create_line(x, y - 22, x, y + 22, fill="#e33955", width=2)
                self.create_text(x, y + 32, text="建议落点", fill="#d21d3c", font=(FONT, 10, "bold"))

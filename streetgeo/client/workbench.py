"""Daily image → prediction → map correction workspace, shared with the client."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from streetgeo import PROJECT_SHORT_NAME
from streetgeo.paths import ROOT
from streetgeo.maps.projection import parse_coordinates, coordinates
from streetgeo.maps.widgets import WorldMap
from streetgeo.inference.sessions import SessionStore
from streetgeo.ui.common import (ACCENT, BG, BORDER, FONT, INPUT, MUTED, NAV, PANEL,
                                 SURFACE, TEXT, WARNING, ScrollFrame, apply_dark_theme, button, label)
from streetgeo.ui.desktop_input import FileDropTarget, clipboard_content

if __package__:
    from .prediction import ClientApp
    from .i18n import LANGUAGES, map_text
else:
    from client_ui import ClientApp
    from client_i18n import LANGUAGES, map_text


def _preferences(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _distance(a, b):
    p, q = math.radians(a[0]), math.radians(b[0])
    h = math.sin((q-p)/2)**2 + math.cos(p)*math.cos(q)*math.sin(math.radians(b[1]-a[1])/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(max(0, min(1, h))))


class DailyApp(ClientApp):
    def __init__(self, root, *, settings_path=None, checkpoint=None, engine=None, language=None, device="cpu"):
        settings_path = Path(settings_path) if settings_path else ROOT / ".runtime/settings.json"
        saved = _preferences(settings_path)
        self.store = SessionStore(settings_path.parent / "workspace")
        self.auto_predict = tk.BooleanVar(root, value=saved.get("auto_predict", True) is True)
        self.auto_learn = tk.BooleanVar(root, value=saved.get("auto_learn") is True)
        self.online_roads = tk.BooleanVar(root, value=saved.get("online_roads") is True)
        choice = saved.get("device", device)
        self.device_choice = tk.StringVar(root, value=choice if choice in ("auto", "cpu", "cuda") else device)
        self.profile_choice = tk.StringVar(root)
        self.coordinate_text = tk.StringVar(root)
        self.pin_text = tk.StringVar(root)
        self.slot_choice = tk.StringVar(root, value="1")
        self.base_checkpoint = str(saved.get("base_checkpoint") or checkpoint or saved.get("checkpoint", ""))
        self.personal_checkpoint = str(saved.get("personal_checkpoint", ""))
        self.learning_updates = int(saved.get("learning_updates", 0)) if isinstance(saved.get("learning_updates", 0), int) else 0
        self.learning_memory = int(saved.get("learning_memory", 0)) if isinstance(saved.get("learning_memory", 0), int) else 0
        self.manual_point = None
        self.learning = False
        self._internal_model_change = False
        self.history_window = self.settings_window = self.history_table = None
        self._preview_windows, self._aux_inputs = [], []
        self.drop_target = None
        super().__init__(root, settings_path=settings_path, checkpoint=checkpoint, engine=engine,
                         language=language, device=self.device_choice.get())
        if not self.base_checkpoint:
            self.base_checkpoint = self.checkpoint.get()
        self.root.bind("<Control-v>", self.paste_images, add="+")
        self.root.bind("<Control-V>", self.paste_images, add="+")
        self.root.bind("<Control-Shift-V>", lambda e: self.paste_images(e, append=True), add="+")
        self.root.bind("<Control-Return>", lambda e: self.start_prediction(), add="+")
        self.root.bind("<Escape>", lambda e: self.clear_pin() if not self.busy else None, add="+")
        try:
            self.drop_target = FileDropTarget(root, self.import_images)
        except (OSError, AttributeError, tk.TclError):
            pass  # Paste and file selection remain usable on unsupported systems.
        self.map.set_online(self.online_roads.get())
        self._refresh_text()

    def _build(self):
        root = self.root
        apply_dark_theme(root)
        root.geometry(f"{min(1480, int(root.winfo_screenwidth()*.94))}x{min(990, int(root.winfo_screenheight()*.91))}")
        root.minsize(1120, 760)
        header = tk.Frame(root, bg=NAV)
        header.pack(fill="x")
        identity = tk.Frame(header, bg=NAV)
        identity.pack(side="left", padx=22, pady=13)
        label(identity, PROJECT_SHORT_NAME, size=19, bold=True, color="white").pack(anchor="w")
        self._text(label(identity, size=9, color=MUTED), "daily_subtitle").pack(anchor="w", pady=(3,0))
        actions = tk.Frame(header, bg=NAV)
        actions.pack(side="right", padx=18)
        self._text(button(actions, "", self.open_history, padx=11, pady=7), "history").pack(side="left", padx=4)
        self._text(button(actions, "", self.open_settings, padx=11, pady=7), "settings").pack(side="left", padx=4)
        self.language_box = ttk.Combobox(actions, textvariable=self.language_choice,
                                         values=tuple(LANGUAGES.values()), state="readonly", width=9)
        self.language_box.pack(side="left", padx=(10,0))
        self.language_box.bind("<<ComboboxSelected>>", self.change_language)

        models = tk.Frame(root, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        models.pack(fill="x", padx=18, pady=(12,0))
        self.profile_box = ttk.Combobox(models, textvariable=self.profile_choice, state="readonly", width=18)
        self.profile_box.pack(side="left", padx=10, pady=8)
        self.profile_box.bind("<<ComboboxSelected>>", self.change_profile)
        self.model_entry = ttk.Entry(models, textvariable=self.checkpoint, font=(FONT,9))
        self.model_entry.pack(side="left", fill="x", expand=True, pady=8)
        self.model_entry.bind("<FocusOut>", lambda e: self._save_settings())
        self.model_button = self._text(button(models,"",self.choose_checkpoint,padx=12,pady=7),"choose_base")
        self.model_button.pack(side="left", padx=10, pady=8)

        body = tk.Frame(root,bg=BG)
        body.pack(fill="both",expand=True,padx=18,pady=12)
        left = tk.Frame(body,bg=PANEL,width=350,highlightbackground=BORDER,highlightthickness=1)
        left.pack(side="left",fill="y",padx=(0,14));left.pack_propagate(False)
        sidebar = ScrollFrame(left);sidebar.pack(fill="both",expand=True)
        fields=sidebar.content;fields.configure(padx=14,pady=14)
        self._text(label(fields,size=11,bold=True),"step_images").pack(anchor="w")
        self.paste_button=self._text(button(fields,"",self.paste_images,primary=True,pady=12),"paste")
        self.paste_button.pack(fill="x",pady=(11,7))
        tools=tk.Frame(fields,bg=PANEL);tools.pack(fill="x")
        self.choose_button=self._text(button(tools,"",self.choose_images,padx=10,pady=6),"choose")
        self.choose_button.pack(side="left",fill="x",expand=True)
        self.add_view_button=self._text(button(tools,"",lambda:self.choose_images(append=True),padx=9,pady=6),"add_view")
        self.add_view_button.pack(side="left",padx=(7,0))
        self._text(label(fields,size=8,color=MUTED,wraplength=294,justify="left"),"import_hint").pack(fill="x",pady=(7,9))
        self.preview=tk.Label(fields,bg=INPUT,fg=MUTED,height=7,font=(FONT,10),cursor="hand2")
        self.preview.pack(fill="x")
        self.preview.bind("<Button-1>",lambda e:self.open_preview())
        self.image_names=label(fields,size=8,color=MUTED,wraplength=290,justify="left")
        self.image_names.pack(fill="x",pady=(5,4))
        views=tk.Frame(fields,bg=PANEL);views.pack(fill="x",pady=(0,8))
        self.view_box=ttk.Combobox(views,textvariable=self.slot_choice,values=("1",),state="readonly",width=3)
        self.view_box.pack(side="left")
        self.remove_view_button=button(views,"×",self.remove_view,padx=7,pady=4)
        self.remove_view_button.pack(side="left",padx=(3,0))
        self._text(button(views,"",self.open_preview,padx=8,pady=4),"preview").pack(side="left",padx=5)
        self.clear_button=self._text(button(views,"",self.clear_images,padx=8,pady=4),"clear_images")
        self.clear_button.pack(side="right")
        self.auto_predict_check=self._text(ttk.Checkbutton(fields,variable=self.auto_predict,command=self._save_settings),"auto_predict")
        self.auto_predict_check.pack(anchor="w",pady=(0,10))
        self._text(label(fields,size=11,bold=True),"step_predict").pack(anchor="w")
        self.predict_button=self._text(button(fields,"",self.start_prediction,primary=True,pady=10),"predict")
        self.predict_button.pack(fill="x",pady=(9,5))
        self.progress=ttk.Progressbar(fields,mode="indeterminate");self.progress.pack(fill="x",pady=(0,12))
        self.coordinate_entry=ttk.Entry(fields,textvariable=self.coordinate_text,state="readonly",font=("Consolas",12))
        self.coordinate_entry.pack(fill="x")
        result_tools=tk.Frame(fields,bg=PANEL);result_tools.pack(fill="x",pady=(6,8))
        self.copy_button=self._text(button(result_tools,"",self.copy_coordinates,padx=10,pady=5),"copy")
        self.copy_button.pack(side="left")
        self._text(button(result_tools,"",self.export_round,padx=10,pady=5),"export").pack(side="left",padx=7)
        self.probability_label=label(fields,size=8,color=MUTED,wraplength=290,justify="left")
        self.probability_label.pack(fill="x",pady=(0,7))
        self.candidate_list=tk.Listbox(fields,height=5,bg=INPUT,fg=TEXT,relief="flat",selectbackground="#245b61",
            selectforeground="white",highlightthickness=0,font=("Consolas",10),exportselection=False)
        self.candidate_list.pack(fill="x");self.candidate_list.bind("<<ListboxSelect>>",self.select_candidate)
        self.explain_button=self._text(button(fields,"",self.open_explanation,pady=6),"explain")
        self.explain_button.pack(fill="x",pady=(10,0))
        self.coverage_check=ttk.Checkbutton(fields,variable=self.google_coverage_only,command=self._coverage_changed)

        right=tk.Frame(body,bg=BG);right.pack(side="left",fill="both",expand=True)
        map_tools=tk.Frame(right,bg=BG);map_tools.pack(fill="x",pady=(0,7))
        self._text(label(map_tools,size=12,bold=True),"results").pack(side="left")
        self._text(button(map_tools,"",self.map_world,padx=9,pady=5),"world").pack(side="right",padx=(5,0))
        self._text(button(map_tools,"",self.focus_selected,padx=9,pady=5),"focus").pack(side="right")
        self._text(ttk.Checkbutton(map_tools,variable=self.online_roads,command=self.change_roads),"roads").pack(side="right",padx=12)
        self._text(label(right,size=8,color=MUTED),"map_help").pack(anchor="w",pady=(0,6))
        self.map=WorldMap(right,on_click=self.map_click,translate=lambda source:map_text(source,self.language),
                           cache_folder=self.settings_path.parent/"map_tiles")
        self.map.pack(fill="both",expand=True)

        feedback=tk.Frame(right,bg=PANEL,highlightbackground=BORDER,highlightthickness=1)
        feedback.pack(fill="x",pady=(10,0))
        title=tk.Frame(feedback,bg=PANEL);title.pack(fill="x",padx=12,pady=(9,3))
        self._text(label(title,size=11,bold=True),"step_truth").pack(side="left")
        self.error_label=label(title,size=9,color=WARNING);self.error_label.pack(side="right")
        self._text(label(feedback,size=8,color=MUTED),"truth_hint").pack(anchor="w",padx=12,pady=(0,5))
        pin_tools=tk.Frame(feedback,bg=PANEL);pin_tools.pack(fill="x",padx=12,pady=(0,6))
        self.pin_entry=ttk.Entry(pin_tools,textvariable=self.pin_text,font=("Consolas",10))
        self.pin_entry.pack(side="left",fill="x",expand=True)
        self.pin_entry.bind("<Return>",lambda e:self.apply_pin())
        self.pin_button=self._text(button(pin_tools,"",self.apply_pin,padx=10,pady=5),"apply_pin")
        self.pin_button.pack(side="left",padx=6)
        self._text(button(pin_tools,"",self.clear_pin,padx=9,pady=5),"clear_pin").pack(side="left")
        learning_tools=tk.Frame(feedback,bg=PANEL);learning_tools.pack(fill="x",padx=12,pady=(0,6))
        self.learn_button=self._text(button(learning_tools,"",self.start_learning,primary=True,padx=12,pady=8),"learn")
        self.learn_button.pack(side="left")
        self.undo_button=self._text(button(learning_tools,"",self.undo_learning,padx=10,pady=8),"undo")
        self.undo_button.pack(side="left",padx=7)
        self.auto_learn_check=self._text(ttk.Checkbutton(learning_tools,variable=self.auto_learn,command=self._save_settings),"auto_learn")
        self.auto_learn_check.pack(side="right")
        self.learning_label=label(feedback,size=8,color=MUTED,wraplength=780,justify="left")
        self.learning_label.pack(anchor="w",padx=12,pady=(0,8))
        footer=tk.Frame(root,bg=BG);footer.pack(fill="x",padx=20,pady=(0,10))
        tk.Label(footer,textvariable=self.status,bg=BG,fg=ACCENT,font=(FONT,9),anchor="w").pack(fill="x")
        self._set_busy(False)

    def _refresh_text(self):
        super()._refresh_text()
        self.root.title(f"{PROJECT_SHORT_NAME} · {self.t('daily_title')}")
        self.profile_box.configure(values=(self.t("base_model"),self.t("personal_model")))
        personal=self.personal_checkpoint and str(self.checkpoint_path)==str(Path(self.personal_checkpoint))
        self.profile_choice.set(self.t("personal_model" if personal else "base_model"))
        self.learning_label.configure(text=self.t("learning_stats",updates=self.learning_updates,memory=self.learning_memory)
            if self.learning_updates else self.t("learning_hint"))
        self._update_metrics()
        self._refresh_history()
        if self.history_window is not None and self.history_window.winfo_exists():self.history_window.title(self.t("history"))
        if self.settings_window is not None and self.settings_window.winfo_exists():self.settings_window.title(self.t("settings"))

    def _save_settings(self):
        super()._save_settings()
        try:
            saved=_preferences(self.settings_path)
            saved.update(base_checkpoint=self.base_checkpoint,personal_checkpoint=self.personal_checkpoint,
                         auto_predict=self.auto_predict.get(),auto_learn=self.auto_learn.get(),
                         online_roads=self.online_roads.get(),device=self.device_choice.get(),
                         learning_updates=self.learning_updates,learning_memory=self.learning_memory)
            temp=self.settings_path.with_suffix(".tmp")
            temp.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding="utf-8")
            os.replace(temp,self.settings_path)
        except OSError:
            pass

    def _set_busy(self,busy):
        super()._set_busy(busy)
        if not hasattr(self,"learn_button"):
            return
        for widget in (self.paste_button,self.add_view_button,self.clear_button,self.pin_entry,self.pin_button,self.remove_view_button,
                       self.auto_learn_check,self.auto_predict_check):
            widget.configure(state="disabled" if busy or self.closing else "normal")
        self.profile_box.configure(state="disabled" if busy or self.closing else "readonly")
        self.view_box.configure(state="disabled" if busy or self.closing else "readonly")
        self.learn_button.configure(state="normal" if not busy and self.manual_point and self._can_learn() else "disabled")
        self.undo_button.configure(state="normal" if not busy and self.personal_checkpoint else "disabled")
        if busy:
            self.progress.start(15)
        else:
            self.progress.stop()
        self.predict_button.configure(text=self.t("learning" if busy and self.learning else "preparing" if busy else "predict"))
        for widget in self._aux_inputs:
            if widget.winfo_exists():widget.configure(state="disabled" if busy else "readonly" if isinstance(widget,ttk.Combobox) else "normal")

    def _checkpoint_changed(self,*args):
        if self._internal_model_change:
            return
        super()._checkpoint_changed(*args)
        self.coordinate_text.set("")
        self.map.ground_truth=self.manual_point
        self.map.redraw()
        if not self.base_checkpoint:
            self.base_checkpoint=self.checkpoint.get()

    def choose_checkpoint(self):
        if self.busy:
            return
        selected=self.checkpoint_path
        path=filedialog.askopenfilename(parent=self.root,title=self.t("choose_base"),
            initialdir=str(selected.parent) if selected and selected.parent.exists() else str(ROOT),
            filetypes=[(self.t("model_files"),"*.pt *.pth *.ckpt"),(self.t("all_files"),"*.*")])
        if path:
            self.base_checkpoint=str(Path(path).resolve())
            self.personal_checkpoint=""
            self.learning_updates=self.learning_memory=0
            self.checkpoint.set(self.base_checkpoint)
            self._save_settings();self._refresh_text()

    def change_profile(self,event=None):
        if self.busy:
            return
        personal=self.profile_choice.get()==self.t("personal_model")
        path=self.personal_checkpoint if personal else self.base_checkpoint
        if not path or not Path(path).is_file():
            self.error("no_personal" if personal else "missing_model")
            self._refresh_text();return
        self.checkpoint.set(path);self._save_settings();self._refresh_text()

    def choose_images(self,append=False):
        if self.busy:
            return
        paths=filedialog.askopenfilenames(parent=self.root,title=self.t("choose"),
            filetypes=[(self.t("image_files"),"*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),(self.t("all_files"),"*.*")])
        if paths:self.import_images(paths,append=append)

    def import_images(self,paths,append=False):
        if self.busy or self.closing:
            self._set_status("working_wait");return False
        combined=[*self.images,*paths] if append else list(paths)
        if not self.set_images(combined):return False
        self.store.prune_captures(self.images)
        if self.auto_predict.get() and self.checkpoint_path and self.checkpoint_path.is_file():
            self.start_prediction()
        return True

    def paste_images(self,event=None,append=False):
        if event is not None:
            focused=self.root.focus_get()
            if focused is not None and focused.winfo_class() in ("Entry","TEntry","TCombobox","Text","Spinbox","TSpinbox"):
                return None
        if self.busy or self.closing:
            self._set_status("working_wait");return "break"
        try:
            content=clipboard_content()
            if isinstance(content,Image.Image):
                paths=[self.store.capture(content)]
            elif content:
                paths=content
            else:
                self._set_status("no_clipboard");return "break"
            self.import_images(paths,append=append)
        except (OSError,ValueError):
            self._set_status("clipboard_failed")
        return "break"

    def set_images(self,paths):
        if not super().set_images(paths):return False
        self.manual_point=None;self.pin_text.set("");self.coordinate_text.set("")
        self.map.ground_truth=None
        self.view_box.configure(values=tuple(str(i+1) for i in range(len(self.images))))
        self.slot_choice.set("1")
        self._update_metrics();self._set_busy(False)
        return True

    def clear_images(self):
        if self.busy:return
        self.images=[];self.result=None;self.candidates=[];self.preview_photo=None
        self.preview.configure(image="",height=7)
        self.map.candidates=[];self.manual_point=None;self.map.ground_truth=None
        self.pin_text.set("");self.coordinate_text.set("")
        self.map.show_world();self._refresh_text();self._set_busy(False)
        self._set_status("ready" if self.checkpoint.get() else "model_required")
        self.store.prune_captures()

    def remove_view(self):
        if self.busy or not self.images:return
        index=min(len(self.images)-1,max(0,int(self.slot_choice.get() or "1")-1))
        remaining=[p for i,p in enumerate(self.images) if i!=index]
        if remaining:self.import_images(remaining)
        else:self.clear_images()

    def open_preview(self):
        if not self.images:return
        index=min(len(self.images)-1,max(0,int(self.slot_choice.get() or "1")-1))
        try:
            from streetgeo.inference.backend import read_picture
            picture=read_picture(self.images[index])
        except (OSError,ValueError):self.error("invalid_image");return
        window=tk.Toplevel(self.root);window.title(self.t("preview"));window.configure(bg=BG)
        window.geometry("960x680");window.minsize(480,320)
        panel=tk.Label(window,bg=BG);panel.pack(fill="both",expand=True)
        def render(event):
            im=picture.copy();im.thumbnail((max(1,event.width-20),max(1,event.height-20)))
            panel.photo=ImageTk.PhotoImage(im,master=window);panel.configure(image=panel.photo)
        panel.bind("<Configure>",render);window.bind("<Escape>",lambda e:window.destroy())
        self._preview_windows.append(window)

    def show_result(self,value,record=True):
        if "manual_point" in value:self.manual_point=value["manual_point"]
        super().show_result(dict(value))
        self.result.setdefault("images",list(self.images))
        if self.manual_point:
            self.result["manual_point"]=self.manual_point
            self.pin_text.set(f"{self.manual_point['latitude']:.6f}, {self.manual_point['longitude']:.6f}")
        self.map.ground_truth=self.manual_point;self.map.redraw()
        if record:self._save_round()
        self._update_metrics();self._set_busy(False)

    def _save_round(self):
        if self.result is None:return
        try:self.result=self.store.save(self.result);self._refresh_history()
        except (OSError,ValueError):self._set_status("history_save_failed")

    def select_candidate(self,event=None):
        selected=self.candidate_list.curselection()
        if selected and selected[0]<len(self.candidates):
            self.selected=selected[0];self.map.selected=self.selected;self.map.redraw()
        self._update_metrics()

    def _update_metrics(self):
        if not hasattr(self,"error_label"):return
        self.error_label.configure(text="")
        if self.candidates:
            candidate=self.candidates[min(self.selected,len(self.candidates)-1)]
            self.coordinate_text.set(f"{candidate['latitude']:.6f}, {candidate['longitude']:.6f}")
            prob=candidate.get("cell_probability")
            self.probability_label.configure(text=self.t("grid_probability",probability=prob) if prob is not None else "")
            if self.manual_point:
                km=_distance((candidate['latitude'],candidate['longitude']),
                             (self.manual_point['latitude'],self.manual_point['longitude']))
                self.error_label.configure(text=self.t("error_distance",rank=self.selected+1,distance=km))
        else:self.coordinate_text.set("");self.probability_label.configure(text="")

    def map_click(self,latitude,longitude):
        if self.busy or self.closing:return
        self.set_manual_point(latitude,longitude)

    def set_manual_point(self,latitude,longitude):
        if self.busy:return
        lat,lon=coordinates(latitude,longitude)
        self.manual_point={"latitude":lat,"longitude":lon,"label":"手动标点","source":"user_annotation"}
        self.pin_text.set(f"{lat:.6f}, {lon:.6f}")
        self.map.ground_truth=self.manual_point;self.map.redraw()
        if self.result:
            self.result["manual_point"]=self.manual_point;self._save_round()
        self._update_metrics();self._set_busy(False)
        self._set_status("pin_ready",coordinates=self.pin_text.get())
        if self.auto_learn.get() and self._can_learn():self.start_learning()

    def apply_pin(self):
        if self.busy:return
        try:lat,lon=parse_coordinates(self.pin_text.get())
        except (TypeError,ValueError):self.error("invalid_coordinates");return
        self.set_manual_point(lat,lon)
        self.map.focus(lat,lon,max(5,self.map.zoom))

    def clear_pin(self):
        if self.busy:return
        self.manual_point=None;self.pin_text.set("");self.map.ground_truth=None
        if self.result:self.result.pop("manual_point",None);self._save_round()
        self.map.redraw();self._update_metrics();self._set_busy(False)

    def map_world(self):self.map.show_world()

    def focus_selected(self):
        if self.candidates:
            p=self.candidates[self.selected];self.map.focus(p["latitude"],p["longitude"],max(5,self.map.zoom))

    def change_roads(self):
        self.map.set_online(self.online_roads.get());self._save_settings()

    def copy_coordinates(self):
        value=self.coordinate_text.get()
        if value:
            self.root.clipboard_clear();self.root.clipboard_append(value)
            self._set_status("copied",coordinates=value)

    def export_round(self):
        if not self.result:return
        path=filedialog.asksaveasfilename(parent=self.root,title=self.t("export"),defaultextension=".json",initialfile="svgt_round.json",filetypes=[("JSON","*.json")])
        if path:
            try:Path(path).write_text(json.dumps(self.result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");self._set_status("export_done")
            except (OSError,ValueError):self.error("export_failed")

    def _can_learn(self):
        if not self.result or self.result.get("learned") or not self.images:return False
        path=self.checkpoint_path
        if not path or not path.is_file():return False
        return (self.result.get("source")=="model" and self.result.get("images")==self.images
                and self.result.get("checkpoint")==str(path.resolve())
                and self.result.get("checkpoint_mtime_ns")==path.stat().st_mtime_ns)

    def start_learning(self):
        if self.busy or self.closing:return
        if not self.manual_point:self.error("need_pin");return
        if not self._can_learn():self.error("already_learned" if self.result and self.result.get("learned") else "need_prediction");return
        from streetgeo.training.online import _reject_official_test
        try:_reject_official_test([Path(p) for p in self.images])
        except ValueError:self.error("official_test");return
        path=str(self.checkpoint_path.resolve());images=list(self.images);point=dict(self.manual_point)
        device=self.prediction_device
        self.learning=True;self._set_busy(True);self._set_status("learning")
        def worker():
            try:
                from streetgeo.training.online import learn_from_feedback
                self.engine.model,self.engine.key=None,None
                result=learn_from_feedback(path,images,point['latitude'],point['longitude'],device=device,
                    output_root=self.settings_path.parent/"online_learning")
                self.events.put(("learned",result))
            except Exception:
                self._log_failure();self.events.put(("learning_error",None))
        threading.Thread(target=worker,daemon=True,name="svgt-daily-learning").start()

    def undo_learning(self):
        if self.busy or self.closing:return
        if not self.personal_checkpoint:self.error("no_update");return
        path=self.personal_checkpoint
        self.learning=True;self._set_busy(True);self._set_status("learning")
        def worker():
            try:
                from streetgeo.training.online import undo_feedback
                self.engine.model,self.engine.key=None,None
                self.events.put(("learning_undone",undo_feedback(path)))
            except Exception:
                self._log_failure();self.events.put(("learning_error",None))
        threading.Thread(target=worker,daemon=True,name="svgt-daily-undo").start()

    def handle_event(self,kind,value):
        if kind not in ("learned","learning_undone","learning_error"):return False
        self.learning=False;self._set_busy(False)
        if kind=="learning_error":
            if not self.closing:self._set_status("learning_failed");self.error("learning_failed")
            return True
        self._internal_model_change=True
        try:self.checkpoint.set(value["checkpoint"])
        finally:self._internal_model_change=False
        self.learning_updates=value["updates"]
        if kind=="learned":
            self.base_checkpoint=value["base_checkpoint"];self.personal_checkpoint=value["checkpoint"]
            self.learning_memory=value["memory_size"]
            self.result["learned"]=True;self.result["learning_checkpoint"]=value["checkpoint"]
            self._save_round();self._set_status("learning_done",updates=value["updates"])
        else:
            if self.learning_updates==0:self.personal_checkpoint="";self.learning_memory=0
            if self.result:self.result["learned"]=False;self.result["learning_undone"]=True;self._save_round()
            self.result=None;self.candidates=[];self.map.candidates=[];self.candidate_list.delete(0,"end");self.map.redraw()
            self._set_status("undone")
        if self.explanation_window is not None and not self.explanation_window.closed:self.explanation_window.close()
        self._save_settings();self._refresh_text();self._set_busy(False)
        return True

    def open_history(self):
        if self.history_window is not None and self.history_window.winfo_exists():self._refresh_history();self.history_window.lift();return
        window=tk.Toplevel(self.root);self.history_window=window
        window.title(self.t("history"));window.geometry("820x460");window.configure(bg=BG)
        self._text(label(window,size=9,color=MUTED,wraplength=760),"history_hint").pack(fill="x",padx=16,pady=12)
        table=ttk.Treeview(window,columns=("time","image","status"),show="headings",selectmode="browse")
        self.history_table=table
        for key,w in (("time",175),("image",410),("status",100)):
            table.heading(key,text=self.t("history_"+key));table.column(key,width=w)
        table.pack(fill="both",expand=True,padx=16,pady=(0,16))
        self._refresh_history()
        def select(event=None):
            ids=table.selection()
            if ids:self.open_round(ids[0])
        table.bind("<Double-1>",select);table.bind("<Return>",select)
        if not self.store.records:label(window,self.t("history_empty"),color=MUTED).pack(pady=8)

    def _refresh_history(self):
        table=self.history_table
        if table is None or not table.winfo_exists():return
        selected=table.selection();position=table.yview()
        table.delete(*table.get_children())
        for key in ("time","image","status"):table.heading(key,text=self.t("history_"+key))
        for record in reversed(self.store.records):
            try:timestamp=datetime.fromisoformat(record.get("created_at","")).astimezone().strftime("%m-%d %H:%M")
            except ValueError:timestamp="—"
            state="history_learned" if record.get("learned") else "history_marked" if record.get("manual_point") else "history_predicted"
            table.insert("","end",iid=record['round_id'],values=(timestamp,Path(record['images'][0]).name if record['images'] else "—",self.t(state)))
        if selected and table.exists(selected[0]):table.selection_set(selected[0])
        if position:table.yview_moveto(position[0])

    def open_round(self,identifier):
        if self.busy:return
        record=next((r for r in self.store.records if r['round_id']==identifier),None)
        if not record:return
        if not all(Path(p).is_file() for p in record['images']):self.error("history_missing");return
        if not self.set_images(record['images']):return
        self.show_result(record,record=False);self._set_status("history_loaded")

    def open_settings(self):
        if self.busy:return
        if self.settings_window is not None and self.settings_window.winfo_exists():self.settings_window.lift();return
        window=tk.Toplevel(self.root);self.settings_window=window
        window.title(self.t("settings"));window.geometry("540x390");window.configure(bg=PANEL)
        self._text(label(window,size=10,bold=True),"device").pack(anchor="w",padx=18,pady=(18,8))
        device_box=ttk.Combobox(window,textvariable=self.device_choice,values=("auto","cuda","cpu"),state="readonly")
        device_box.pack(fill="x",padx=18)
        self._aux_inputs=[device_box]
        def change(event=None):
            self.prediction_device=self.device_choice.get();self._save_settings()
        device_box.bind("<<ComboboxSelected>>",change)
        coverage=self._text(ttk.Checkbutton(window,variable=self.google_coverage_only,command=self._coverage_changed),"coverage_only")
        coverage.pack(anchor="w",padx=18,pady=(18,6));self._aux_inputs.append(coverage)
        self._text(label(window,size=9,color=MUTED,wraplength=490,justify="left"),"coverage_hint").pack(fill="x",padx=18)
        self._text(label(window,size=9,color=WARNING,wraplength=490,justify="left"),"auto_learn_hint").pack(fill="x",padx=18,pady=(18,10))
        self._text(label(window,size=9,color=MUTED,wraplength=490,justify="left"),"storage_hint").pack(fill="x",padx=18)

    def _destroy(self):
        if self.drop_target is not None:self.drop_target.close()
        super()._destroy()


def main():
    parser=argparse.ArgumentParser(description="S.V.G.T. daily prediction workspace")
    parser.add_argument("--settings");args=parser.parse_args()
    if os.name=="nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError,OSError):pass
    root=tk.Tk()
    settings=Path(args.settings) if args.settings else ROOT/".gui/workspace_settings.json"
    saved=_preferences(settings)
    checkpoint=None
    if not saved.get("checkpoint") and (ROOT/"checkpoints/SVGT135V4.pt").is_file():
        checkpoint=ROOT/"checkpoints/SVGT135V4.pt"
    DailyApp(root,settings_path=settings,checkpoint=checkpoint,language="zh" if not saved.get("language") else None,device="auto")
    root.mainloop()


if __name__=="__main__":main()

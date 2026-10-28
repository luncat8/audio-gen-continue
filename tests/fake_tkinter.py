"""A very small stand-in for tkinter, enough to construct the GUI in tests.

Widgets record their options, Text widgets keep their content, and every other
Tk method is a no-op. That catches construction errors (bad option names, typos,
wrong widget calls) on machines without a display or without tkinter installed.
"""

import sys
import types


class TclError(Exception):
    pass


class Variable:
    def __init__(self, value=None, **kwargs):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class StringVar(Variable):
    def __init__(self, value="", **kwargs):
        super().__init__(value)


class BooleanVar(Variable):
    def __init__(self, value=False, **kwargs):
        super().__init__(value)


class _Widget:
    def __init__(self, *args, **kwargs):
        self._options = dict(kwargs)
        self.children = []
        self.state = kwargs.get("state")
        if args and isinstance(args[0], _Widget):
            args[0].children.append(self)

    # every Tk call we do not model explicitly is a no-op
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def _noop(*args, **kwargs):
            return None

        return _noop

    def configure(self, **kwargs):
        self._options.update(kwargs)
        if "state" in kwargs:
            self.state = kwargs["state"]

    config = configure

    def grid(self, **kwargs):
        self._options.update(kwargs)

    def pack(self, **kwargs):
        self._options.update(kwargs)

    def winfo_geometry(self):
        return "900x900"


class Text(_Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content = ""

    def delete(self, start, end=None):
        self.content = ""

    def insert(self, index, text):
        self.content += text

    def get(self, start=None, end=None):
        return self.content

    def see(self, index):
        pass


class Root(_Widget):
    def __init__(self):
        super().__init__()
        self.callbacks = []
        self.destroyed = False
        self.geometry_value = ""

    def title(self, value):
        self.title_value = value

    def minsize(self, width, height):
        self.minsize_value = (width, height)

    def geometry(self, value=None):
        if value is None:
            return self.geometry_value
        self.geometry_value = value

    def protocol(self, name, callback):
        self.protocol_callback = callback

    def after(self, delay, callback):
        self.callbacks.append((delay, callback))

    def destroy(self):
        self.destroyed = True

    def mainloop(self):
        pass


class _Ttk:
    Frame = _Widget
    LabelFrame = _Widget
    Label = _Widget
    Entry = _Widget
    Button = _Widget
    Checkbutton = _Widget
    Combobox = _Widget
    Spinbox = _Widget
    Progressbar = _Widget
    Scrollbar = _Widget

    class Style:
        def theme_use(self, name):
            pass


def install():
    """Insert the fake modules into sys.modules and return the tkinter module."""
    tk = types.ModuleType("tkinter")
    tk.Tk = Root
    tk.Text = Text
    tk.StringVar = StringVar
    tk.BooleanVar = BooleanVar
    tk.TclError = TclError

    ttk = types.ModuleType("tkinter.ttk")
    for name in ("Frame", "LabelFrame", "Label", "Entry", "Button", "Checkbutton",
                 "Combobox", "Spinbox", "Progressbar", "Scrollbar", "Style"):
        setattr(ttk, name, getattr(_Ttk, name))

    filedialog = types.ModuleType("tkinter.filedialog")
    filedialog.answer = ""
    filedialog.askopenfilename = lambda **kwargs: filedialog.answer
    filedialog.askdirectory = lambda **kwargs: filedialog.answer

    class _MessageBox:
        def showerror(self, *args, **kwargs):
            self.last = ("error", args)

        def showinfo(self, *args, **kwargs):
            self.last = ("info", args)

        def showwarning(self, *args, **kwargs):
            self.last = ("warning", args)

    messagebox = _MessageBox()

    tk.ttk = ttk
    tk.filedialog = filedialog
    tk.messagebox = messagebox

    sys.modules["tkinter"] = tk
    sys.modules["tkinter.ttk"] = ttk
    sys.modules["tkinter.filedialog"] = filedialog
    sys.modules["tkinter.messagebox"] = messagebox
    tk.ttk = ttk
    return tk

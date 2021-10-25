import tkinter as tk
import tkinter.ttk as ttk
from algorithms.ArtisticStyleTransformation import *
from algorithms.Slideshow import *
from algorithms.ColorMix import *
from algorithms.Morphing import *
from algorithms.Morphing2 import *


class ButtonsFrame(tk.Frame):
    def __init__(self, parent, frames):
        tk.Frame.__init__(self, parent)
        self.buttons = {}

        # Generate images for buttons
        self.button_imgs = {}
        for idx, name in enumerate(frames.keys()):
            path = "images/buttons/" + name + ".png"
            self.button_imgs[name] = tk.PhotoImage(file=path)

        # Template images for fake buttons
        self.img_even = tk.PhotoImage(file="images/buttons/button_even.png")
        self.img_odd = tk.PhotoImage(file="images/buttons/button_odd.png")

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        canvas = tk.Canvas(self, height=700, yscrollcommand=vscrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.container = container = tk.Frame(canvas)
        container_id = canvas.create_window(0, 0, window=container, anchor=tk.NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (container.winfo_reqwidth(), container.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if container.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=container.winfo_reqwidth())
        container.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if container.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(container_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)

        # Generate buttons
        self.add_buttons(container, parent, frames)

    # Creating and adding buttons
    def add_buttons(self, container, parent, frames):
        # Generate dictionary of buttons
        for name in frames.keys():
            self.buttons[name] = ttk.Button(container,
                                            text=name,
                                            image=self.button_imgs[name],
                                            command=lambda frame_name=name: parent.show_frame(frame_name))
            # Pack generated button to view it in the scrollbar frame
            self.buttons[name].pack()


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        # Set title and size of the window
        self.title("Computational Creativity")
        self.geometry("1200x700")
        self.resizable(False, False)
        self.frames = {}
        self.button_frame = None

    def create(self, frames):
        # Generate all created frames.
        self.frames = self.generate_frames(frames)

        # Show first frame and buttons
        self.show_frame("Artistic Style Transfer")
        self.button_frame = ButtonsFrame(self, self.frames)
        self.button_frame.grid(row=0, column=0, rowspan=1, columnspan=1, sticky='nsew')

    # Configure algorithm frames
    @staticmethod
    def generate_frames(frames):
        # Set grid so that frames do not overlap buttons
        for frame in frames.keys():
            frames[frame].place(x=281, y=0, anchor="nw", width=919, height=700)

        return frames

    # Change view content frame
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    app = MainApplication()
    # Add frames so that u can call them later
    implemented_alg = {
        "Artistic Style Transfer": AST(app),
        "Slideshow": Slideshow(app),
        "Color Mix": ColorMix(app),
        "Morphing": Morphing(app),
        "Morphing2": Morphing2(app)
    }
    # Create frames to show
    app.create(implemented_alg)
    # Run the app
    app.mainloop()

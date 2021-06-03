import tkinter as tk


class TestFrame2(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        list_of_options = ["first option", "second option", "third option", "forth option"]
        first_option = tk.StringVar(self)
        first_option.set(list_of_options[0])

        tk.Label(self, text="TEST 2").grid(row=0, column=0)
        tk.Entry(self, bd=5).grid(row=0, column=1)
        tk.Label(self, text="TEST 2").grid(row=0, column=2)
        tk.Entry(self, bd=5).grid(row=0, column=3)
        tk.Label(self, text="TEST 2").grid(row=2, column=0)
        tk.Checkbutton(self, bd=5).grid(row=2, column=1)
        tk.Label(self, text="TEST 2").grid(row=2, column=2)
        tk.Checkbutton(self, bd=5).grid(row=2, column=3)
        tk.Label(self, text="TEST 2").grid(row=3, column=0)
        tk.OptionMenu(self, first_option, *list_of_options).grid(row=3, column=1)
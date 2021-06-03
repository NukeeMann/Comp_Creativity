import tkinter as tk


class nowyAlg(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        tk.Label(self, text="TO JEST OPIS").grid(row=0, column=0)
        tk.Entry(self, bd=5).grid(row=0, column=1)
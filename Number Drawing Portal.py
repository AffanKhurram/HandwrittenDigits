import tkinter as tk
from tkinter import Canvas
i = 0
class DrawingWindow:
    def __init__(self, master):
        def coords(event):
            global i
            i = i+1
            print(i)
            print(event.x, event.y)
        #def draw(x, y):

        self.master = master
        self.master.title('Write a Number')
        c = Canvas(master, width=500, height=500)
        c.bind('<Button-1>', coords)
        c.pack()
        b = tk.Button(master, text='Clear', width=70, command=c.delete("all"))
        #b.configure(bg = "blue")
        b.pack()
        b2 = tk.Button(master, text='Stop', width=70, command=master.destroy)
        b2.pack()
def main():
    root = tk.Tk()
    app = DrawingWindow(root)
    root.mainloop()
if __name__ == '__main__':
    main()
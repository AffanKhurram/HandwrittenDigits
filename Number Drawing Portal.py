import tkinter as tk
from tkinter import Canvas
i = 0
drawing = []
stroke = []
x = 0
y = 0
xprev = 0
yprev = 0
firstClick = False
erase = False
imageArray = [[255 for i in range(500)] for j in range(500)]
class DrawingWindow:
    def __init__(self, master):
        
        def draw(event):
            global i, x, y, xprev, yprev, firstClick, stroke, erase
            if erase:
                color = "#ffffff"
                size = 8
            else:
                color = "#000000"
                size = 1
            i = i+1
            if not firstClick:
                xprev = x
                yprev = y
                x = event.x
                y = event.y
                c.create_line(xprev, yprev, x, y, fill=color, width=2*size+1)
                
                coord = [x,y]
                stroke.append(coord)
            else:
                firstClick = False
                x = event.x
                y = event.y
                stroke = []
                coordInfo = [x,y, color, size]   
                stroke.append(coordInfo)
            c.create_oval(x-size, y-size, x+size, y+size, fill = color, outline = color)
            
            
        def released(event):
            drawing.append(stroke)
        def toggleFirstClick(event):
            global firstClick
            firstClick = True
        def eraseStroke():
            print("erase")
            #error protection for if no strokes left
            if len(drawing) != 0:
                clear()
                drawing.pop(len(drawing)-1)
                for stroke in drawing:                #strokeToErase = drawing[len(drawing)-1]
                    firstX = stroke[0][0]
                    firstY = stroke[0][1]
                    color = stroke[0][2]
                    size = stroke[0][3]
                    c.create_oval(firstX-size, firstY-size, firstX+size, firstY+size, fill = color, outline = color)
                    global xprev
                    global yprev
                    xprev = firstX
                    yprev = firstY
                    for coord in stroke[1:]:
                        x = coord[0]
                        y = coord[1]
                        c.create_line(xprev, yprev, x, y, fill=color, width=2*size+1)
                        c.create_oval(x-size, y-size, x+size, y+size, fill = color, outline = color)
                        xprev = x
                        yprev = y
        #def toggleErase():
        #    global erase
        #    erase = not erase
        #    if erase == True:
        #        b2.configure(text="Draw")
        #    else:
        #        b2.configure(text="Erase")
        def clear():
            c.delete("all")
        def save():
            clear()
            global imageArray
            for stroke in drawing:
                firstX = stroke[0][0]
                firstY = stroke[0][1]
                if stroke[0][2] == "#000000":
                    color = 0
                else:
                    color = 255                     ###########not dealing with this, larger size, different color
                size = stroke[0][3]                 ###########not dealing with this, larger size, different color
                imageArray[firstX][y-size] = color
                imageArray[firstX-size][y] = color
                imageArray[firstX][y] = color
                imageArray[x+size][y] = color
                imageArray[x][y+size] = color
                global xprev
                global yprev
                xprev = firstX
                yprev = firstY
                for coord in stroke[1:]:
                    x = coord[0]
                    y = coord[1]
                    #c.create_line(xprev, yprev, x, y, fill=color, width=2*size+1)  deal with this later
                    imageArray[x][y-size] = color
                    imageArray[x-size][y] = color
                    imageArray[x][y] = color
                    imageArray[x+size][y] = color
                    imageArray[x][y+size] = color
                    xprev = x
                    yprev = y


        self.master = master
        self.master.title('Write a Number')
        c = Canvas(master, width=500, height=500)
        c.configure(bg = "white")
        c.bind('<B1-Motion>', draw)
        c.bind('<Button-1>', toggleFirstClick)
        c.bind('<ButtonRelease-1>', released)
        c.bind('<Motion>', )
        c.pack()
        b1 = tk.Button(master, text='Undo', width=70, command=eraseStroke)
        b1.pack()
        #b2 = tk.Button(master, text='Erase', width=70, command=toggleErase)
        #b2.pack()
        b3 = tk.Button(master, text='Clear', width=70, command=clear)
        b3.pack()
        b4 = tk.Button(master, text='Stop', width=70, command=master.destroy)
        b4.pack()
        b5 = tk.Button(master, text='Done', width=70, command=save)
        b5.pack()
def main():
    root = tk.Tk()
    app = DrawingWindow(root)
    root.mainloop()
if __name__ == '__main__':
    main()
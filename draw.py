import tkinter as tk
from tkinter import Canvas
import numpy as np
i = 0
drawing = []
stroke = []
x = 0
y = 0
xprev = 0
yprev = 0
firstClick = False
erase = False
imgs = []
class DrawingWindow:
    def __init__(self, master):
        root = master
        def draw(event):
            global i, x, y, xprev, yprev, firstClick, stroke, erase
            if erase:
                color = "#ffffff"
                size = 10
            else:
                color = "#000000"
                size = 10
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

        def clear():
            c.delete("all")
        def save():
            from PIL import Image, ImageOps, ImageFilter
            
            img = getter(c)
            img = ImageOps.grayscale(img)
            arr = np.array(img)

            mnist_img = resize(arr)

            imgs.append(mnist_img)
            import multiclassAnn as ann
            ann.forward_pass(mnist_img)
            from tensorflow import keras
            model = keras.models.load_model('model.h5')
            print(np.argmax(model(mnist_img[None, :, :, None], training=False)))         


        def getter(widget):
            from PIL import ImageGrab
            x=root.winfo_rootx()+widget.winfo_x()
            y=root.winfo_rooty()+widget.winfo_y()
            x1=x+widget.winfo_width()
            y1=y+widget.winfo_height()
            img = ImageGrab.grab().crop((x,y,x1,y1))
            img.save('test.jpg')
            return img

        def resize(arr):
            from PIL import Image, ImageOps, ImageFilter
            startr, endr, startc, endc = 999, 999, 999, -1
            for i, row in enumerate(arr):
                for j, val in enumerate(row):
                    imgval = -1
                    if val > 200:
                        imgval = 255
                    elif val < 100:
                        imgval = 0
                    else:
                        print('ther was an error ', val)
                    if imgval == 0:
                        endr = i
                        if startr == 999:
                            startr = i
                        if j < startc:
                            startc = j
                        if j > endc:
                            endc = j

            newarr = arr[startr:endr+1, startc:endc+1]
            res_img = Image.fromarray(newarr)
            res_img.thumbnail((20, 20), Image.ANTIALIAS)
            res_img = res_img.filter(ImageFilter.SHARPEN)
            res_img = ImageOps.invert(res_img)
            res_arr = np.array(res_img) / 255
            h, w = res_arr.shape
            sum = np.sum(res_arr)
            avg_h = np.sum(res_arr * np.arange(h)[:, None]) / sum
            avg_w = np.sum(res_arr * np.arange(w)) / sum
            dh = int(14 - avg_h)
            dw = int(14 - avg_w)
            mnist_img = np.zeros((28, 28))
            for i, row in enumerate(res_arr):
                for j, val in enumerate(row):
                    mnist_img[i + dh][j + dw] = val

            return mnist_img

        def save_array():
            import pickle 
            pickle.dump(np.array(imgs), open('imgs.pkl', 'wb'))

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
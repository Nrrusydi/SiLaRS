import tkinter as tk
import cv2
from PIL import ImageTk, Image
from RealTime import execute_realtime

def main():
    root = tk.Tk()
    root.title("Realtime Gesture Recognition")

    label = tk.Label(root)
    label.pack()

    def show_frame():
        frame = execute_realtime()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image)
        label.config(image=image)
        label.image = image
        label.after(10, show_frame)

    show_frame()
    root.mainloop()

if __name__ == '__main__':
    main()

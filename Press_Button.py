from tkinter import *
from subprocess import call

root=Tk()
root.geometry('200x100')
frame = Frame(root)
frame.pack(pady=20,padx=20)

def Open():
    call(["python", "Automated_Text_Summarization.py", "3"])

btn=Button(frame,text='Open File',command=Open)
btn.pack()

root.mainloop()

from tkinter import *
fields = 'Name', 'Item'

def fetch(entries):
	#print(entries)

   for entry in entries:
      field = entry[0]
      text  = entry[1].get()
      print(entry[1].get()) 

	
def makeform(root, fields):
   entries = []
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=10, text=field, anchor='w',font=('Helvetica', '25'))
      ent = Entry(row,font=('Helvetica', '25'))
      row.pack(side=TOP, fill=X, padx=5, pady=15)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)
      entries.append((field, ent))
   return entries

if __name__ == '__main__':
   root = Tk()
   ents = makeform(root, fields)
   root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
   b1 = Button(root, text='Show',command=(lambda e=ents: fetch(e)), bd=1,bg = "green",height = 2, width=8,font=('Helvetica', '25') ).place(x=100,y=300)
  
   b2=Button(root, text="Back", bd=1,bg = "green", command = root.destroy,height = 2, width=8,font=('Helvetica', '25') ).place(x=300,y=300)

  
   root.mainloop()


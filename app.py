from cell import Cell
import numpy as np

input=np.array([0])
long=np.array([0])
short=np.array([0])
params=[np.array([2.7,2,1.41,4.38]),np.array([1.63,1.65,0.94,-0.19]),np.array([1.62,.62,-.32,.59])]

cell1=Cell(gates=(short,long),input_array=input,params=params)
cell2=Cell(gates=cell1.forward(),input_array=np.array([0.5]),params=params)
cell3=Cell(gates=cell2.forward(),input_array=np.array([0.25]),params=params)
cell4=Cell(gates=cell3.forward(),input_array=np.array([1]),params=params)

print(cell4.forward())

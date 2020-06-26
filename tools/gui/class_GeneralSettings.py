'''
*Usage*                                                                                                                                   #####
*Creates and Displays/Raise the GeneralSettings Frame and Initiate the GeneralSettings key of the main dictionary -> results.             #####                                                                                 #####
                                                                                                                                          #####
*Functions and Classes*                                                                                                                   #####
*1 class: GeneralSettings                                                                                                                 #####
*Main Function: display_generalSettings - Is in charge of managing the creation of the General Settings Frame in the SecondFrame          #####
                                                                                                                                          #####
*MAIN DICTIONARY KEY = 'Random Seed'                                                                                                      #####
'''
try:
    import tkinter as tk
    from tkinter import *
    from tkinter import ttk
except ImportError:
    import Tkinter as tk
    from Tkinter import *
    from Tkinter import ttk
import json

#FILES import
import classes_FRAMES
import class_KORALI
import functions

# Frame Variables:
selectorColor = 'aliceblue'
forbidden = ['Variables','Problem','Solver']


class GeneralSettings():
    def __init__(self,master,selectedtab,experiments):
        # master is the frame from the previous class where we want to insert data.      
        
        if experiments[selectedtab]['general_first_time'] == False:
            self.Show_frame(experiments,selectedtab)
            return
        
        self.gs = tk.Frame(master,bg=selectorColor,width=495,height=925)
        self.gs.grid(column=0,row=0)
        self.gs.grid_propagate(0)

        # STORE THIS FRAME IN THE experiments dictionary.
        experiments[selectedtab]['generalSettings'] = self.gs

        file = ''
        file = open('../../source/modules/experiment/experiment.config','r')
        if file != '':
            mainConf = json.load(file)
            cont = 'Random Seed'
            results = experiments[selectedtab]['results']
            experiments[selectedtab]['mainConf'] = {}
            experiments[selectedtab]['mainConf'] = mainConf
            results[cont] = {}
            
            self.efake =tk.Label(self.gs, text='', fg=selectorColor, bg=selectorColor)
            self.efake.grid(row=0, column=0,columnspan=4,pady = 20 ,padx=10, sticky='nw')
            self.e =tk.Button(self.gs, text='General Settings',activebackground='aliceblue', font="Arial 20", fg='black', bg=selectorColor,borderwidth=2,
                              relief='solid')
            self.e.grid(row=1, column=0,columnspan=4,pady = 4 ,padx=10, sticky='n')
            self.edescr =tk.Label(self.gs, text='Main configuration variables...', font="Arial 10", fg='navy', bg=selectorColor)
            self.edescr.grid(row=2, column=0,columnspan=4,rowspan=1,pady = 4 ,padx=10, sticky='n')
            self.efake =tk.Label(self.gs, text='', fg=selectorColor, bg=selectorColor)
            self.efake.grid(row=3, column=0,columnspan=4,pady = 10 ,padx=10, sticky='nw')
            c = 0
            r = 4
            self.display_generalSettings(self.gs,r,c,cont,selectedtab,experiments)
            file.close()
        else:
            functions.popupmsgwarning("There is not an 'experiment.config' file !")
            
        experiments[selectedtab]['general_first_time'] = False

    def Show_frame(self,experiments,selectedtab):
        general_settings = experiments[selectedtab]['generalSettings']
        general_settings.tkraise()

    def display_generalSettings(self,gs,r,c,cont,selectedtab,experiments):
        c=0
        dict_defaults = {"Verbosity": "Normal"}
        results = experiments[selectedtab]['results']
        mainConf = experiments[selectedtab]['mainConf']
        titulos = []
        for key in mainConf.keys():
            if key == "Configuration Settings":
                lista = mainConf[key]
                for dicc in lista:
                    if type(dicc) == dict:
                        if 'Type' not in dicc.keys():
                            functions.popupmsgwarning("No 'Type' field found")
                            break
                        else:
                            for key2 in dicc.keys():
                                texto = dicc['Name']
                                if texto == 'Random Seed':
                                    cont == 'Random Seed'
                                if len(texto)>1: 
                                    titulo = texto[0]
                                    texto = texto[1]
                                    if titulo not in titulos:
                                        e2 =tk.Label(self.gs, text=titulo,justify='left', anchor="w", font="Arial 18 bold", fg='navy', bg=selectorColor)
                                        e2.grid(row=r, column=c,columnspan= 1,pady = 25 ,padx=4, sticky='w')
                                        titulos.append(titulo)
                                        cont=titulo
                                        results[cont] = {}
                                else:
                                    texto = texto[0] # Get the element inside list.
                                if texto in forbidden: # IFF Problem, Solver or Variables... skip
                                    break
                                else:
                                    if key2 == 'Type':
                                        # key2 are the dictionary keys, example: Type, Function, Description, Produced By...
                                        description = dicc['Description']
                                        try:
                                            options = dicc['Options'] # options = list of diccionaries.
                                        except:
                                            options = 'None'
                                        fakedescription = description
                                        functions.printdata(self.gs,experiments,selectedtab,dicc[key2], texto, r, c,cont,options,dict_defaults)                                    
                                    r+=1


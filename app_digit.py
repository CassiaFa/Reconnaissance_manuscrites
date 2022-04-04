##----- Importation des Modules -----##
from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

def main():
    Draw_digit()

class Draw_digit:

    def __init__(self):

        ##----- Chargement du Modèle -----##
        self.model = tf.keras.models.load_model('model_CNN')

        ##----- Définition des Variables globales -----##
        self.x1 = 0				# Mémorisation de la position antérieure de la souris
        self.y1 = 0
        self.l, self.h = 280, 280		# Largeur et hauteur du canevas

        ##----- Création de la fenêtre -----##
        self.fen = Tk()
        self.fen.title('Dessin')

        ##----- Création de l'image -----##
        self.img = Image.new("RGB", (self.l, self.h), "black")
        self.dessin = ImageDraw.Draw(self.img)

        ##----- Création des boutons -----##
        self.bouton_valider = Button(self.fen, text='Valider', command=self.valider)
        self.bouton_valider.grid(row=1, column = 1, padx = 5, pady = 5)
        self.bouton_effacer = Button(self.fen, text='Effacer', command=self.effacer)
        self.bouton_effacer.grid(row=1, column = 0, padx = 5, pady = 5)

        ##----- Création du canevas -----##
        self.zone_dessin = Canvas(self.fen, width = self.l, height = self.h, bg ='black')
        self.zone_dessin.grid(row=0, column = 0, columnspan = 2, padx =5, pady =5)

        ##----- Création de la zone de prediction -----##
        self.zone_prediction = Label(self.fen, text='Prediction:')
        self.zone_prediction.grid(row=2, column = 0, padx = 5, pady = 5)
        self.zone_prediction_valeur = Label(self.fen, text='')
        self.zone_prediction_valeur.grid(row=2, column = 1, padx = 5, pady = 5)

        ##----- Programme principal -----##
        self.zone_dessin.bind('<Button-1>', self.clic)            # pour le premier clic
        self.zone_dessin.bind('<B1-Motion>', self.deplace_clic)   # pour le déplacement bouton enfoncé

        self.fen.mainloop()                  # Boucle d'attente des événements


    ##----- Définition des Fonctions -----##
    def clic(self, event):
        """ Prise en compte de l'événement clic gauche sur la zone graphique """
        global x1, y1
        # position du clic
        self.x1 = event.x
        self.y1 = event.y


    def deplace_clic(self,event):
        """ Prise en compte de l'événement déplacement de la souris avec clic gauche enfoncé
            jusqu'au relâchement du clic gauche."""
        global x1, y1
        # nouvelle position après déplacement de la souris - bouton enfoncé
        self.x2 = event.x
        self.y2 = event.y
        # on dessine une ligne
        self.zone_dessin.create_line(self.x1, self.y1, self.x2, self.y2, fill='white', width=10, smooth=1, joinstyle=ROUND, capstyle=ROUND, splinesteps=36)
        self.dessin.line([self.x1, self.y1, self.x2, self.y2], fill='white', width=30, joint="curve")
        # on mémorise la nouvelle position du clic
        self.x1 = self.x2
        self.y1 = self.y2


    def effacer(self):
        """ Efface la zone graphique """
        self.zone_dessin.delete(ALL)
        self.img = Image.new("RGB", (self.l, self.h), "black")
        self.dessin = ImageDraw.Draw(self.img)
        self.zone_prediction_valeur.config(text='')

    def valider(self):
        """ Valide le dessin et prédit la valeur """
        # self.img.save('dessin.png')
        self.image = self.img.resize((28, 28), Image.ANTIALIAS)
        self.image = np.array(self.image)/255.0
        self.image = np.expand_dims(self.image, axis=0)
        print(self.image.shape)
        
        proba = self.model.predict(self.image)
        prediction = proba.argmax(axis = -1)

        self.zone_prediction_valeur.config(text=prediction)



if __name__ == '__main__':
    main()


# ##----- Création de la fenêtre -----##
# fen = Tk()
# fen.title('Dessin')

# ##----- Création de l'image -----##
# img = Image.new("RGB", (l, h), "black")
# dessin = ImageDraw.Draw(img)

# ##----- Création des boutons -----##
# bouton_valider = Button(fen, text='Valider', command=valider)
# bouton_valider.grid(row=1, column = 1, padx = 5, pady = 5)
# bouton_effacer = Button(fen, text='Effacer', command=effacer)
# bouton_effacer.grid(row=1, column = 0, padx = 5, pady = 5)

# ##----- Création du canevas -----##
# zone_dessin = Canvas(fen, width = l, height = h, bg ='black')
# zone_dessin.grid(row=0, column = 0, columnspan = 2, padx =5, pady =5)

# ##----- Programme principal -----##
# zone_dessin.bind('<Button-1>', clic)            # pour le premier clic
# zone_dessin.bind('<B1-Motion>', deplace_clic)   # pour le déplacement bouton enfoncé

# fen.mainloop()                  # Boucle d'attente des événements
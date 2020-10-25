# -*- coding: utf-8 -*-
from tempfile import NamedTemporaryFile

import face_recognition as fr
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from preprocessors import Preprocessor
from utils import visualize_detections

# header
st.title('COVID-19: Détection de masque')
# sidebar
pages = ['Accueil', 'Essayer']
choice = st.sidebar.selectbox('Menu', pages)
# processor
pp = Preprocessor(224, 224)


# mask detection
def mask_recognition(image):
    preds = []
    img = np.array(image.convert('RGB'))
    # detect and encode faces
    boxes = fr.face_locations(img)
    for top, right, bottom, left in boxes:
        if (right - left) >= 50 and (bottom - top) >= 50:
            face = img[top:bottom, left:right]
            face = pp.preprocess(face)
            face = face.reshape(1, 224, 224, 3)
            # prediction
            preds.append(model.predict(face)[0][0])
    img = visualize_detections(img, boxes, preds)
    return img


# load model
@st.cache
def load_data():
    return tf.keras.models.load_model('./model/best_nasnet.h5')


# about
def about():
    # description
    st.write(
        'Cette application est un détecteur de masque dans une image. Dans un'
        'contexte de COVID-19 où le port du masque est vivement recommander. '
        "Ceci démontre l'utilisation du deep learning pour détecter des masques"
        'dans une image.'
    )
    st.write(
        "Vous pouvez trouver le code sur [GitLab](https://gitlab.com/schalappe/mask-detector)."
    )
    # function
    st.header('Fonctionnement')
    st.write(
        '''
            Cette application a été bâtie en deux étapes: 
            \n1. Présence d'un masque dans l'image: Oui ou Non ?
            \n2. Détecter le masque dans l'image.
        '''
    )
    # step 1
    st.subheader('Etape 1: Présence')
    st.write(
        'Dans la première étape, il faut bâtir un modèle de classification '
        'pour savoir si oui ou non une image contient un masque. '
        'Voir le code [ici](https://gitlab.com/schalappe/spot-the-mask-challenge)'
    )
    # step 2
    st.subheader('Etape 2: Détection')
    st.write(
        'Avec un modèle permettant de dire si une image contient un masque. '
        'On peut bâtir un application pour détecter des masques sur les visages '
        'des personne. Cette application fonction en 3 phases:\n'
    )
    st.write(
        '''
            1. utiliser un algoritme pour détecter des visages
            2. extraire les visages détectés
            3. determiner si l'image (visage) contient un masque
        '''
    )
    st.write(
        'Pour la détection de visage, on utilise la bibliothèque: '
        '[Face Recognition](https://github.com/ageitgey/face_recognition). '
        "Ce qui peut poser certains problème. Car le succès de l'application "
        "dépend dès lors du bon fonctionnement de ce package."
    )
    # examples
    st.header('Exemples')
    st.image('./img/5.jpg', width=708, caption='Exemple 1: Masque détecté')
    st.image('./img/3.jpg', width=708, caption='Exemple 2: Sans masque')
    # go further
    st.header('Aller plus loin')
    st.write("L'utilisation de l'application de detection de visage peut engendrer des erreurs. "
             "Si le visage n'est pas reconnu, il n'y a pas de détection de "
             "masque. Pour parler aller, on peut entraîner spécifiquement un "
             "modèle pour détecter les masques.")
    st.image('./img/0.jpg', width=708,
             caption="Exemple 3: Visage non détecté par le package bien que le "
                     "modèle est détecté la présence de masque")


# main
def main():

    if choice == 'Accueil':
        about()
    elif choice == 'Essayer':
        st.write("Uploder une image avec des visages pour tester l'application.")
        file = st.file_uploader(" ", type=['jpeg', 'png', 'jpg'])
        temp_file = NamedTemporaryFile(delete=False)
        if file is not None:
            temp_file.write(file.getvalue())
            image = Image.open(temp_file.name)
            if st.button("Lancer"):
                result_img = mask_recognition(image)
                st.image(result_img, use_column_width=True)


if __name__ == '__main__':
    model = load_data()
    main()

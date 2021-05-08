# -*- coding: utf-8 -*-
from tempfile import NamedTemporaryFile

import mtcnn
import logging
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils import visualize_detections

# logging level
logging.basicConfig(filename="app.log", filemode="w", format="%(name)s - %(levelname)s - %(message)s")

# header
st.title("COVID-19: Détection de masque")

# sidebar
pages = ["Accueil", "Essayer"]
choice = st.sidebar.selectbox("Menu", pages)


# load model
@st.cache
def load_data():
    logging.info("Load model")
    return [tf.keras.models.load_model("./model/best_nasnet.h5"), mtcnn.MTCNN()]


# mask detection
def mask_recognition(image: np.ndarray) -> np.ndarray:
    """
    Take an image, search face then search mask
    Args:
        image (ndarray): Image to process

    Returns:
        ndarray: Image processed
    """
    preds = []
    img = np.array(image.convert("RGB"))
    # detect and encode faces
    boxes = detector.detect_faces(img)
    for box in boxes:
        x, y, width, height = box["box"]
        if width >= 30 and height >= 30 and box["confidence"] > 0.97:
            face = img[y : y + height, x : x + width]
            face = Image.fromarray(face)
            face = face.resize((224, 224))
            face = tf.keras.preprocessing.image.img_to_array(face) / 255.0
            face = np.expand_dims(face, axis=0)
            # prediction
            preds.append(model.predict(face)[0][0])
    img = visualize_detections(img, boxes, preds)
    return img


# about
def about():
    # description
    st.write(
        "Cette application est un détecteur de masque dans une image. Dans un "
        "contexte de COVID-19 où le port du masque est vivement recommander. "
        "Ceci démontre l'utilisation du deep learning pour détecter des masques "
        "dans une image."
    )
    st.write("Vous pouvez trouver le code sur [GitHub](https://github.com/schalappe/Mask-Detector).")
    # function
    st.header("Fonctionnement")
    st.write(
        """
            Cette application a été bâtie en deux étapes:
            \n1. Présence d'un masque dans l'image: Oui ou Non ?
            \n2. Détecter le masque dans l'image.
        """
    )
    # step 1
    st.subheader("Etape 1: Présence")
    st.write(
        "Dans la première étape, il faut bâtir un modèle de classification "
        "pour savoir si oui ou non une image contient un masque. "
        "Voir le code [ici](https://github.com/schalappe/Mask-Challenge)"
    )
    # step 2
    st.subheader("Etape 2: Détection")
    st.write(
        "Avec un modèle permettant de dire si une image contient un masque. "
        "On peut bâtir un application pour détecter des masques sur les visages "
        "des personne. Cette application fonction en 3 phases:\n"
    )
    st.write(
        """
            1. utiliser un modèle pour détecter des visages
            2. extraire les visages détectés
            3. determiner si l'image (visage) contient un masque
        """
    )
    st.write(
        "Pour la détection de visage, on utilise la bibliothèque: "
        "[MTCNN](https://github.com/ipazc/mtcnn). "
        "Ce qui peut poser certains problème. Car le succès de l'application "
        "dépend dès lors du bon fonctionnement de ce package."
    )
    # examples
    st.header("Exemples")
    st.image("./img/5.jpg", width=708, caption="Exemple 1: Masque détecté")
    st.image("./img/3.jpeg", width=708, caption="Exemple 2: Sans masque")
    # go further
    st.header("Aller plus loin")
    st.write(
        "L'utilisation de l'application de detection de visage peut engendrer des erreurs. "
        "Si le visage n'est pas reconnu, il n'y a pas de détection de "
        "masque. Pour aller plus loin, on peut entraîner spécifiquement un "
        "modèle pour détecter les masques."
    )
    st.image(
        "./img/0.jpg",
        width=708,
        caption="Exemple 3: Visage non détecté par le package bien que le "
        "modèle est détecté la présence d'un masque",
    )


# main
def main():

    if choice == "Accueil":
        about()
    elif choice == "Essayer":
        st.write("Uploder une image avec des visages pour tester l'application.")
        file = st.file_uploader(" ", type=["jpeg", "png", "jpg"])
        temp_file = NamedTemporaryFile(delete=False)
        if file is not None:
            temp_file.write(file.getvalue())
            image = Image.open(temp_file.name)
            if st.button("Lancer"):
                try:
                    result_img = mask_recognition(image)
                    st.image(result_img, use_column_width=True)
                    logging.debug("Mask find in image")
                except Exception as e:
                    logging.warning(f"Error in processing {e}")
                    st.write("Nous avons rencontré certains problèmes avec cette image ...")
                    st.image(image, use_column_width=True)


if __name__ == "__main__":
    model, detector = load_data()
    main()

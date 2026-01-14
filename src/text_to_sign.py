import cv2
import numpy as np

alphabet_dict = {
    # Your dictionary mappings...
    '1':r"C:\Users\Admin\Documents\Btechproject\Indian\1\4.jpg",
    '2':r"C:\Users\Admin\Documents\Btechproject\Indian\2\4.jpg",
    '3':r"C:\Users\Admin\Documents\Btechproject\Indian\3\4.jpg",
    '4':r"C:\Users\Admin\Documents\Btechproject\Indian\4\4.jpg",
    '5':r"C:\Users\Admin\Documents\Btechproject\Indian\5\4.jpg",
    '6':r"C:\Users\Admin\Documents\Btechproject\Indian\6\4.jpg",
    '7':r"C:\Users\Admin\Documents\Btechproject\Indian\7\4.jpg",
    '8':r"C:\Users\Admin\Documents\Btechproject\Indian\8\4.jpg",
    '9':r"C:\Users\Admin\Documents\Btechproject\Indian\9\4.jpg",
    'A':r"C:\Users\Admin\Documents\Btechproject\Indian\A\4.jpg",
    'B':r"C:\Users\Admin\Documents\Btechproject\Indian\B\4.jpg",
    'C':r"C:\Users\Admin\Documents\Btechproject\Indian\C\4.jpg",
    'D':r"C:\Users\Admin\Documents\Btechproject\Indian\D\4.jpg",
    'E':r"C:\Users\Admin\Documents\Btechproject\Indian\E\4.jpg",
    'F':r"C:\Users\Admin\Documents\Btechproject\Indian\F\4.jpg",
    'G':r"C:\Users\Admin\Documents\Btechproject\Indian\G\4.jpg",
    'H':r"C:\Users\Admin\Documents\Btechproject\Indian\H\4.jpg",
    'I':r"C:\Users\Admin\Documents\Btechproject\Indian\I\4.jpg",
    'J':r"C:\Users\Admin\Documents\Btechproject\Indian\J\4.jpg",
    'K':r"C:\Users\Admin\Documents\Btechproject\Indian\K\4.jpg",
    'L':r"C:\Users\Admin\Documents\Btechproject\Indian\L\4.jpg",
    'M':r"C:\Users\Admin\Documents\Btechproject\Indian\M\4.jpg",
    'N':r"C:\Users\Admin\Documents\Btechproject\Indian\N\4.jpg",
    'O':r"C:\Users\Admin\Documents\Btechproject\Indian\O\4.jpg",
    'P':r"C:\Users\Admin\Documents\Btechproject\Indian\P\4.jpg",
    'Q':r"C:\Users\Admin\Documents\Btechproject\Indian\Q\4.jpg",
    'R':r"C:\Users\Admin\Documents\Btechproject\Indian\R\4.jpg",
    'S':r"C:\Users\Admin\Documents\Btechproject\Indian\S\4.jpg",
    'T':r"C:\Users\Admin\Documents\Btechproject\Indian\T\4.jpg",
    'U':r"C:\Users\Admin\Documents\Btechproject\Indian\U\4.jpg",
    'V':r"C:\Users\Admin\Documents\Btechproject\Indian\V\4.jpg",
    'W':r"C:\Users\Admin\Documents\Btechproject\Indian\W\4.jpg",
    'X':r"C:\Users\Admin\Documents\Btechproject\Indian\X\4.jpg",
    'Y':r"C:\Users\Admin\Documents\Btechproject\Indian\Y\4.jpg",
    'Z':r"C:\Users\Admin\Documents\Btechproject\Indian\Z\4.jpg"

    # ...
}

def text_to_sign(text):
    words = text.split()
    line_images = []
    max_width = 0

    for word in words:
        images = []
        for char in word:
            if char.isalpha() or char.isdigit():
                char_upper = char.upper()
                if char_upper in alphabet_dict:
                    img_path = alphabet_dict[char_upper]
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                    else:
                        print(f"Image not loaded for {char_upper}: {img_path}")
                else:
                    print(f"No mapping for {char_upper}")

        if images:  # Process only if images aren't empty
            try:
                word_image = np.hstack(images)
                line_images.append(word_image)
                max_width = max(max_width, word_image.shape[1])
            except Exception as e:
                print(f"Error stacking images for word {word}: {e}")

    if line_images:  # <- now this is correctly paired
        padded_line_images = []
        for line_image in line_images:
            padded_image = np.zeros((line_image.shape[0], max_width, 3), dtype=np.uint8)
            padded_image[:, :line_image.shape[1], :] = line_image
            padded_line_images.append(padded_image)

        combined_image = np.vstack(padded_line_images)
        cv2.imshow('Sign Language', combined_image)
        cv2.waitKey(0)
    else: 
        print("No valid images to display.")


text = input("Enter text: ")
text_to_sign(text)
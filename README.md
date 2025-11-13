# DM_noreference

This is the code of the paper "Learning Underwater Image Enhancement Iteratively without Reference Images" accepted by AAAI 2026

usage steps:

    Putting your data into the dataset folder. (There is initial data in this folder now)(Ours is here:).
    Download the pre-trained model, the link is . Then, put the model in the pre-train folder.
    Execute infer.py to get the inference results in a new folder called experiments_val folder.
    Users can also change the code in the config/underwater.json to change for the dataset root.
    label_clip.py is used to generate the label.txt, namely CLIP text, for text feature extraction.(This is simple way, manually labeling is the best)
    color_filter.py is used to filter the 'cold' color and retain the 'warm' color in the underwater images.
    nethook.py is a package to modify the neural units, extracted from 'Gan dissection: Visualizing and understanding generative adversarial networks, ICLR, 2018'. Based on this package, visualize.py is used to find out which unit controls the 'red color'.
  

P.S. The author is lazy again that he still doesn't want to write more information.

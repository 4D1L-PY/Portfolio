# Visualize decision tree
import os
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image


def display_decision_tree(model, feature_labels, target_labels, tree_nb=5, autodelete_dot_file=True):
    # Export decision tree as dot file
    tree_dot_file_label = f'{tree_nb}th_tree.dot'
    tree_png_file_label = f'{tree_nb}th_tree.png'
    export_graphviz(model.estimators_[tree_nb+1],
                    out_file=tree_dot_file_label, 
                    feature_names=feature_labels,
                    class_names=target_labels,
                    rounded=True,
                    proportion=False, 
                    precision=2,
                    filled=True)
    # Convert to png using system command (requires Graphviz)
    call(['dot', '-Tpng', tree_dot_file_label, '-o', tree_png_file_label, '-Gdpi=600'])
    if autodelete_dot_file:
        os.remove(tree_dot_file_label)
    # Display in jupyter notebook
    return Image(filename=tree_png_file_label)
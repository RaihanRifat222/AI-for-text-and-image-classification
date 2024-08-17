
import os
from collections import Counter
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def call_counts(path):
    """Return a dictionary with the counts of samples from each label name
    >>> call_counts('sample_cifar10')
    {'ship': 8, 'frog': 6, 'cat': 6, 'automobile': 6, 'deer': 6, 'bird': 9, 'airplane': 10, 'truck': 4, 'dog': 8, 'horse': 3}
    """
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    labels = [ label_names[int(file_name[0])] for file_name in os.listdir(path)]
    
    counts = Counter(labels)
    
    return dict(counts)



def sample_images(path, label_name):
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return [(path + '/' + file_name, label_name) for file_name in os.listdir(path) if label_names[int(file_name[0])] == label_name]




def sample_image_folder(path, selected_label_names, sample_numbers, output_filenames):
    """For each number n listed in sample_numbers, return the name of a CSV file that stores the first n 
    samples from each label name specified by exploring the image files stored in path.
    >>> sample_image_folder('sample_cifar10', ('deer', 'airplane', 'truck'), (2, 1), ('file1.csv', 'file2.csv'))
    ['file1.csv', 'file2.csv']
    """
    result = []
    start = 0
    for i, n in enumerate(sample_numbers):
        selected_image_files = []
        for l in selected_label_names:
            image_files = sample_images(path, l)
            selected_image_files += image_files[start:start+n]

        start = start+n
        filename = output_filenames[i]
        lines = ["%s,%s\n" % fname for fname in selected_image_files]
        with open(filename, "w") as f:
            f.writelines(lines)
        result.append(filename)
    return result

if __name__ == "__main__":
    import doctest
    doctest.testmod()

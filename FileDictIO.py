# ELEC475 Lab 5
# Nicholas Chivaran - 18nc34
# Samantha Hawco - 18srh5


def file_to_dict(label_file):

    img_dict = {}
    f = open(label_file)
    for line in f.readlines():
        split_line = line.split(',')

        image_name = str(split_line[0])
        x_label = int(split_line[1].strip()[2:])
        y_label = int(split_line[2].strip()[:-2])
        image_labels = (x_label, y_label)

        img_dict[image_name] = image_labels

    return img_dict


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
    f.close()
    return img_dict


def dict_to_file(input_dict, file_name):
    # generate lines to write into scaled labels file
    lines = []
    for key in input_dict.keys():
        line = f'{key},"{input_dict[key]}"\n'
        lines += [line]

    # write lines
    f = open(file_name, 'w')
    f.writelines(lines)
    f.close()


def get_image_from_coord(coord, label_file):
    file_dict = file_to_dict(label_file)
    for key in file_dict.keys():
        if file_dict[key] == coord:
            return key

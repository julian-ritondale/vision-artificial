def label_to_int(string_label):
    if string_label == 'cleveland_z': return 0
    if string_label == 'orange_ricky': return 1
    if string_label == 'hero': return 2
    if string_label == 'smashboy': return 3
    if string_label == 'teewee': return 4

    else:
        raise Exception('unkown class_label')


def int_to_label(string_label):
    if string_label == 0: return 'cleveland_z'
    if string_label == 1: return 'orange_ricky'
    if string_label == 2: return 'hero'
    if string_label == 3: return 'smashboy'
    if string_label == 4: return 'teewee'
    
    else:
        raise Exception('unkown class_label')

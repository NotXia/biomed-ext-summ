"""
    Pads a list to a given size
"""
def padToSize(to_pad_list, pad_size, filler):
    return to_pad_list + [filler]*(pad_size-len(to_pad_list))
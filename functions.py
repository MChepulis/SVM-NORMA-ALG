def print_dict(dictionary):
    for k, v in dictionary.items():
        print("\'%s\': %s" % (format(k, ""), format(v, "")), end=",   ")

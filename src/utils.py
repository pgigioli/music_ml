def print_and_log(string, filename):
    print(string)
    with open(filename, 'a') as write_file:
        write_file.write(string + '\n')
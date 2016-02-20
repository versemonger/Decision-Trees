import numpy as np


class Data:
    attri_option_list = ["bcxfks", "fgys", "nbcgrpuewy",
                         "tf", "alcyfmnps", "adfn",
                         "cwd", "bn", "knbhgropuewy",
                         "et", "bcuezr?", "fyks",
                         "fyks", "nbcgopewy", "nbcgopewy",
                         "pu", "nowy", "not",
                         "ceflnpsz", "knbhrouwy", "acnsvy",
                         "glmpuwd"]

    chi_table = []
    train = []
    test = []
    validation = []
    validation_flag = False
    display_tree_flag = False
    mode = 'i'  # i for info gain, m for mis classification error
    conf_level = 1  # 0 for 50, 1 for 95, 2 for 99, 3 for 0
    @staticmethod
    def dinput(filename):
        """
        Read data set and store it in a matrix t
        :param filename: the name of the data file
        :return: a matrix containing all data
        """
        f = open(filename, 'r')
        t = []
        for line in f:
            t.append(line.strip().split(','))
        t = np.array(t)
        return t

    def __init__(self):
        Data.train = self.dinput("training.txt")
        Data.test = self.dinput("testing.txt")
        Data.validation = self.dinput("validation.txt")
        chi_table_raw = self.dinput("chi_test_table.txt")
        for row in chi_table_raw:
            temp = []
            for i in range(1, len(row)):
                temp.append(float(row[i]))
            Data.chi_table.append(temp)

